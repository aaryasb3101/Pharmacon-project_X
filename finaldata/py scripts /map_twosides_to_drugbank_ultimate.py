import csv
import sys
import re
from pathlib import Path
from lxml import etree
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
import difflib

remover = SaltRemover()

def local(tag):
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def clean_drug_name(name):
    if not name:
        return ""
    salts = ['hydrochloride', 'sodium', 'sulfate', 'acetate', 'chloride', 
             'maleate', 'tartrate', 'citrate', 'phosphate', 'bromide', 'iodide', 'calcium', 'potassium']
    name_lower = name.lower()
    for s in salts:
        name_lower = re.sub(r'\b' + s + r'\b', '', name_lower)
    name_clean = re.sub(r'[^a-z0-9\s]', '', name_lower)
    return " ".join(name_clean.split())

def process_smiles(smiles_str):
    """Compute canonical SMILES, InChIKey, and connectivity InChIKey using RDKit."""
    if not smiles_str:
        return None, None, None
    try:
        mol = Chem.MolFromSmiles(str(smiles_str))
        if not mol:
            return None, None, None
        
        # Standard InChIKey
        full_ikey = Chem.MolToInchiKey(mol)
        conn_ikey = full_ikey.split('-')[0] if full_ikey and '-' in full_ikey else None

        # Salt-stripped canonical SMILES
        stripped = remover.StripMol(mol)
        if stripped is None or stripped.GetNumAtoms() == 0:
            stripped = mol
        canonical_smiles = Chem.MolToSmiles(stripped)

        return canonical_smiles, full_ikey, conn_ikey
    except Exception:
        return None, None, None

def load_twosides(csv_path):
    print("Loading TWOSIDES drugs.csv and computing chemical features...", file=sys.stderr)
    drugs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug_id = row.get("drug_id") or row.get("id") or list(row.values())[0]
            smiles = row.get("smiles") or row.get("SMILES") or row.get("smile")
            
            # Extract CID from drug_id if it starts with CID
            clean_cid = None
            if drug_id:
                drug_id_str = str(drug_id).strip()
                if drug_id_str.upper().startswith("CID"):
                    clean_cid = drug_id_str[3:].lstrip("0")
                elif drug_id_str.isdigit():
                    clean_cid = str(int(drug_id_str))

            can_smiles, full_ikey, conn_ikey = process_smiles(smiles)
            
            drugs.append({
                "drug_id": drug_id,
                "cid": clean_cid,
                "smiles": smiles,
                "canonical_smiles": can_smiles,
                "inchikey": full_ikey,
                "connectivity_ikey": conn_ikey
            })
            
    print(f"  Loaded and processed {len(drugs)} TWOSIDES drugs\n", file=sys.stderr)
    return drugs

def build_drugbank_indexes(drugbank_path):
    print("Scanning DrugBank for comprehensive indexes (CIDs, InChIKeys, Salt-stripped SMILES)...", file=sys.stderr)
    context = etree.iterparse(drugbank_path, events=("end",), huge_tree=True)
    
    cid_index = {}
    full_ikey_index = {}
    conn_ikey_index = {}
    smiles_index = {}
    name_index = {}
    n = 0

    for event, elem in context:
        if local(elem.tag) != "drug":
            continue
        parent = elem.getparent()
        if parent is None or local(parent.tag) != "drugbank":
            continue

        n += 1
        primary_id = None
        drug_name = None
        synonyms = []
        smiles = None
        inchikey = None
        pubchem_cid = None

        for child in elem:
            tag = local(child.tag)
            if tag == "drugbank-id" and primary_id is None:
                primary_id = child.text
            if tag == "name" and drug_name is None:
                drug_name = child.text
            if tag == "synonyms":
                for syn in child:
                    if syn.text:
                        synonyms.append(syn.text)
            if tag == "calculated-properties":
                for prop in child:
                    if local(prop.tag) != "property":
                        continue
                    kind, value = None, None
                    for pc in prop:
                        if local(pc.tag) == "kind":
                            kind = pc.text
                        if local(pc.tag) == "value":
                            value = pc.text
                    if kind == "SMILES":
                        smiles = value
                    if kind == "InChIKey":
                        inchikey = value
            if tag == "external-identifiers":
                for ext_id in child:
                    if local(ext_id.tag) != "external-identifier":
                        continue
                    resource, identifier = None, None
                    for ec in ext_id:
                        if local(ec.tag) == "resource":
                            resource = ec.text
                        if local(ec.tag) == "identifier":
                            identifier = ec.text
                    if resource and resource.strip() == "PubChem Compound" and identifier:
                        pubchem_cid = identifier.strip().lstrip("0")

        drug_record = (primary_id, drug_name, smiles, inchikey)
        
        if pubchem_cid and primary_id:
            cid_index[pubchem_cid] = drug_record
        if inchikey and primary_id:
            ik_clean = inchikey.strip()
            full_ikey_index[ik_clean] = drug_record
            conn_ikey = ik_clean.split('-')[0]
            if conn_ikey not in conn_ikey_index:
                conn_ikey_index[conn_ikey] = drug_record
        
        if smiles and primary_id:
            std_smiles, _, _ = process_smiles(smiles)
            if std_smiles:
                smiles_index[std_smiles] = drug_record

        if drug_name and primary_id:
            clean_n = clean_drug_name(drug_name)
            if clean_n:
                name_index[clean_n] = drug_record
        for syn in synonyms:
            clean_syn = clean_drug_name(syn)
            if clean_syn and clean_syn not in name_index:
                name_index[clean_syn] = drug_record

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if n % 3000 == 0:
            print(f"  ...scanned {n} DrugBank drugs so far", file=sys.stderr)

    print(f"DrugBank scan complete: {n} drugs indexed.\n", file=sys.stderr)
    return cid_index, full_ikey_index, conn_ikey_index, smiles_index, name_index

def main():
    script_dir = Path(__file__).resolve().parent
    drugs_csv_path = script_dir / "drugs.csv"
    
    drugbank_path = None
    search_roots = [script_dir.parent.parent, Path("/Users/jaanya/Library/Mobile Documents/com~apple~CloudDocs")]
    for root in search_roots:
        if root.exists():
            for p in root.rglob("drugbank_full_database.xml"):
                drugbank_path = p
                break
        if drugbank_path:
            break
            
    if not drugbank_path:
        print(f"Error: Could not find drugbank_full_database.xml.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found DrugBank XML at: {drugbank_path}", file=sys.stderr)
    output_path = script_dir / "twosides_drugbank_mapping_ultimate.csv"

    twosides_drugs = load_twosides(str(drugs_csv_path))
    cid_idx, full_ikey_idx, conn_ikey_idx, smiles_idx, name_idx = build_drugbank_indexes(str(drugbank_path))

    results = []
    match_stats = {"PubChem CID": 0, "Full InChIKey": 0, "Connectivity InChIKey": 0, "Salt-Stripped SMILES": 0, "Unmatched": 0}

    for drug in twosides_drugs:
        db_id, db_name, db_smiles, db_inchikey = (None, None, None, None)
        method = "Unmatched"

        # Pass 1: PubChem CID extracted from drug_id
        if drug["cid"] and drug["cid"] in cid_idx:
            db_id, db_name, db_smiles, db_inchikey = cid_idx[drug["cid"]]
            method = "PubChem CID"
        
        # Pass 2: Full InChIKey computed from SMILES
        elif drug["inchikey"] and drug["inchikey"] in full_ikey_idx:
            db_id, db_name, db_smiles, db_inchikey = full_ikey_idx[drug["inchikey"]]
            method = "Full InChIKey"

        # Pass 3: Connectivity-Only InChIKey (First 14 chars)
        elif drug["connectivity_ikey"] and drug["connectivity_ikey"] in conn_ikey_idx:
            db_id, db_name, db_smiles, db_inchikey = conn_ikey_idx[drug["connectivity_ikey"]]
            method = "Connectivity InChIKey"

        # Pass 4: Salt-Stripped SMILES Match
        elif drug["canonical_smiles"] and drug["canonical_smiles"] in smiles_idx:
            db_id, db_name, db_smiles, db_inchikey = smiles_idx[drug["canonical_smiles"]]
            method = "Salt-Stripped SMILES"

        match_stats[method] += 1

        results.append({
            "twosides_drug_id": drug["drug_id"],
            "matched_drugbank_id": db_id or "",
            "matched_drugbank_name": db_name or "",
            "match_method": method
        })

    with open(output_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    total = len(twosides_drugs)
    matched_total = total - match_stats["Unmatched"]
    pct = (matched_total / total) * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print(f"Total TWOSIDES drugs:                    {total}")
    print(f"Total Successfully Matched:              {matched_total} ({pct:.1f}%)")
    print("-" * 60)
    for m, count in match_stats.items():
        print(f"  - {m}: {count}")
    print(f"\nFull mapping written to: {output_path}")

if __name__ == "__main__":
    main()