"""
Map your 645 drugs to KEGG-documented human drug-metabolism products.

Source: KEGG PATHWAY maps hsa00982 (Drug metabolism - cytochrome P450) and
hsa00983 (Drug metabolism - other enzymes), via the free KEGG REST API
(https://rest.kegg.jp). No predictions -- these are curated reaction
diagrams; a "metabolite" here is a compound appearing as a REACTION PRODUCT
where one of your drugs (or a compound resolving to the same structure)
appears as the REACTION SUBSTRATE, within these two pathway maps.

Pipeline:
  1. Pull KGML for both pathway maps -> parse <reaction> substrate/product
     edges -> build a directed substrate -> product compound graph.
  2. Pull the KEGG compound <-> PubChem crosswalk (conv endpoint). Note:
     KEGG's own conv table is SID-keyed, not CID-keyed, so each SID is
     resolved to CID(s) via PubChem PUG-REST as a second step.
  3. Match your drug CIDs against resolved KEGG compound IDs that appear as
     a SUBSTRATE in the reaction graph.
  4. For each match, pull the corresponding PRODUCT compound's structure
     (MOL block via KEGG) and convert to SMILES/InChIKey with RDKit.
  5. Write drugs_kegg_metabolites.csv with per-drug results, honestly
     marking has_metabolite=0 where no match was found.

Coverage expectation: KEGG's two drug-metabolism maps are curated,
general-purpose diagrams (not a per-drug library), so expect a modest
match rate against an arbitrary 645-drug benchmark list -- this is
real/curated data, not exhaustive.

Rate limiting: KEGG's terms ask for no more than ~3 requests/second from
an individual academic user; this script sleeps between calls accordingly.
"""

import time
import re
import requests
import pandas as pd
from lxml import etree
from rdkit import Chem

DRUGS_CSV = "drugs.csv"
OUT_CSV = "drugs_kegg_metabolites.csv"

PATHWAY_MAPS = ["hsa00982", "hsa00983"]

KEGG_REST = "https://rest.kegg.jp"
PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

SLEEP = 0.35  # be polite to both free APIs


def normalize_cid(raw_id):
    nums = "".join(filter(str.isdigit, str(raw_id)))
    return int(nums) if nums else None


def load_drugs(path):
    df = pd.read_csv(path, dtype={"drug_id": str})
    df["cid"] = df["drug_id"].apply(normalize_cid)
    return df


# ---------- 1. Parse KGML reaction graphs for the two pathway maps ----------

def fetch_kgml(pathway_id):
    url = f"{KEGG_REST}/get/{pathway_id}/kgml"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    time.sleep(SLEEP)
    return r.text


def parse_reaction_edges(kgml_text):
    """
    Returns a list of (substrate_kegg_cid, product_kegg_cid, reaction_id)
    tuples. KGML <reaction> elements list substrate/product children as
    KEGG entry IDs referencing <entry> elements; those <entry> elements'
    'name' attribute holds the actual compound ID(s) (e.g. 'cpd:C00082').
    """
    root = etree.fromstring(kgml_text.encode("utf-8"))

    # entry id -> list of compound IDs (an entry can represent >1 compound)
    entry_compounds = {}
    for entry in root.findall("entry"):
        if entry.get("type") == "compound":
            names = entry.get("name", "").split()
            cids = [n.replace("cpd:", "") for n in names if n.startswith("cpd:")]
            entry_compounds[entry.get("id")] = cids

    edges = []
    for reaction in root.findall("reaction"):
        rxn_id = reaction.get("name")
        substrate_entries = [s.get("id") for s in reaction.findall("substrate")]
        product_entries = [p.get("id") for p in reaction.findall("product")]

        sub_cids = [c for eid in substrate_entries for c in entry_compounds.get(eid, [])]
        prod_cids = [c for eid in product_entries for c in entry_compounds.get(eid, [])]

        for s in sub_cids:
            for p in prod_cids:
                edges.append((s, p, rxn_id))
    return edges


def build_reaction_graph():
    all_edges = []
    for pw in PATHWAY_MAPS:
        print(f"Fetching KGML for {pw}...")
        kgml = fetch_kgml(pw)
        edges = parse_reaction_edges(kgml)
        print(f"  -> {len(edges)} substrate->product edges")
        all_edges.extend(edges)

    substrate_to_products = {}
    for s, p, rxn in all_edges:
        substrate_to_products.setdefault(s, set()).add(p)
    return substrate_to_products


# ---------- 2. KEGG compound <-> PubChem crosswalk (SID, then resolve to CID) ----------

def fetch_kegg_pubchem_sid_map():
    """Returns dict: kegg_compound_id ('C#####') -> list of pubchem SIDs."""
    url = f"{KEGG_REST}/conv/pubchem/compound"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    time.sleep(SLEEP)

    mapping = {}
    for line in r.text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        kegg_id, pubchem_ref = parts
        kegg_cid = kegg_id.replace("cpd:", "")
        sid = pubchem_ref.replace("pubchem:", "")
        mapping.setdefault(kegg_cid, []).append(sid)
    return mapping


def resolve_sid_to_cids(sid, cache={}):
    """PubChem PUG-REST: substance ID -> associated compound CID(s)."""
    if sid in cache:
        return cache[sid]
    url = f"{PUBCHEM_REST}/substance/sid/{sid}/cids/JSON"
    try:
        r = requests.get(url, timeout=10)
        time.sleep(SLEEP)
        if r.status_code != 200:
            cache[sid] = []
            return []
        data = r.json()
        cids = data.get("InformationList", {}).get("Information", [{}])[0].get("CID", [])
        cache[sid] = cids
        return cids
    except (requests.RequestException, KeyError, IndexError):
        cache[sid] = []
        return []


def build_kegg_cid_to_pubchem_cid(kegg_compound_ids):
    """Only resolves SIDs for the KEGG compound IDs we actually need
    (those appearing as substrates in the reaction graph), to avoid
    resolving the entire KEGG compound database unnecessarily."""
    sid_map = fetch_kegg_pubchem_sid_map()
    kegg_to_pubchem_cid = {}
    for kegg_cid in kegg_compound_ids:
        sids = sid_map.get(kegg_cid, [])
        pubchem_cids = set()
        for sid in sids:
            pubchem_cids.update(resolve_sid_to_cids(sid))
        if pubchem_cids:
            kegg_to_pubchem_cid[kegg_cid] = pubchem_cids
    return kegg_to_pubchem_cid


# ---------- 3. Fetch structures for matched KEGG product compounds ----------

def fetch_kegg_mol(kegg_cid, cache={}):
    if kegg_cid in cache:
        return cache[kegg_cid]
    url = f"{KEGG_REST}/get/{kegg_cid}/mol"
    r = requests.get(url, timeout=15)
    time.sleep(SLEEP)
    if r.status_code != 200 or not r.text.strip():
        cache[kegg_cid] = None
        return None
    cache[kegg_cid] = r.text
    return r.text


def mol_to_smiles_inchikey(mol_block):
    mol = Chem.MolFromMolBlock(mol_block)
    if mol is None:
        return None, None
    smiles = Chem.MolToSmiles(mol)
    inchikey = Chem.InchiToInchiKey(Chem.MolToInchi(mol))
    return smiles, inchikey


# ---------- 4. Main ----------

def main():
    drugs_df = load_drugs(DRUGS_CSV)

    print("Building KEGG drug-metabolism reaction graph (hsa00982 + hsa00983)...")
    substrate_to_products = build_reaction_graph()
    all_substrate_kegg_ids = set(substrate_to_products.keys())
    print(f"Total distinct substrate compounds across both maps: {len(all_substrate_kegg_ids)}")

    print("Resolving KEGG compound IDs -> PubChem CIDs for substrates only...")
    kegg_to_pubchem = build_kegg_cid_to_pubchem_cid(all_substrate_kegg_ids)

    # Invert: pubchem_cid -> set of kegg substrate ids that resolve to it
    pubchem_cid_to_kegg = {}
    for kegg_cid, pc_cids in kegg_to_pubchem.items():
        for pc_cid in pc_cids:
            pubchem_cid_to_kegg.setdefault(int(pc_cid), set()).add(kegg_cid)

    print(f"Resolved {len(pubchem_cid_to_kegg)} distinct PubChem CIDs among substrates.")

    rows = []
    n_matched = 0
    for _, row in drugs_df.iterrows():
        cid = row["cid"]
        kegg_hits = pubchem_cid_to_kegg.get(cid, set())

        if not kegg_hits:
            rows.append({
                "drug_id": row["drug_id"], "cid": cid, "smiles": row["smiles"],
                "has_metabolite": 0, "source": "kegg_hsa00982_hsa00983",
                "metabolite_kegg_id": None, "metabolite_smiles": None,
                "metabolite_inchikey": None,
            })
            continue

        # Gather all product compounds reachable from any matched substrate id
        product_kegg_ids = set()
        for kegg_sub_id in kegg_hits:
            product_kegg_ids.update(substrate_to_products.get(kegg_sub_id, set()))

        if not product_kegg_ids:
            rows.append({
                "drug_id": row["drug_id"], "cid": cid, "smiles": row["smiles"],
                "has_metabolite": 0, "source": "kegg_hsa00982_hsa00983",
                "metabolite_kegg_id": None, "metabolite_smiles": None,
                "metabolite_inchikey": None,
            })
            continue

        n_matched += 1
        for prod_id in product_kegg_ids:
            mol_block = fetch_kegg_mol(prod_id)
            smiles, inchikey = (None, None) if mol_block is None else mol_to_smiles_inchikey(mol_block)
            rows.append({
                "drug_id": row["drug_id"], "cid": cid, "smiles": row["smiles"],
                "has_metabolite": 1, "source": "kegg_hsa00982_hsa00983",
                "metabolite_kegg_id": prod_id,
                "metabolite_smiles": smiles, "metabolite_inchikey": inchikey,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    total = len(drugs_df)
    print("\n=== KEGG drug-metabolism coverage ===")
    print(f"Total drugs:   {total}")
    print(f"Matched:       {n_matched} ({100*n_matched/total:.1f}%)")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
