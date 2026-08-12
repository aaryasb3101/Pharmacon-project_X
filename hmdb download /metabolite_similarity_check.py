"""
Computes structural (Tanimoto) similarity between each parent drug's SMILES
and its HMDB-matched candidate metabolite's SMILES, using RDKit Morgan
fingerprints.

Real metabolites are typically structurally SIMILAR to their parent (small
modifications: demethylation, hydroxylation, glucuronidation, etc.) -- so a
high similarity score is supporting evidence for a genuine match, while a low
score suggests the drugbank_id match was likely just a coincidental identity
cross-reference, not a real parent->metabolite relationship.

Usage:
    python3 metabolite_similarity_check.py /path/to/hmdb_metabolites.xml /path/to/drugbank_full_database.xml
"""

import sys
from lxml import etree
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  # silence RDKit's verbose warnings

def local(tag):
    return tag.split('}')[-1] if '}' in tag else tag


def get_drugbank_smiles(drugbank_path):
    """Returns dict: drugbank_id -> (drug_name, smiles)"""
    print("Pass 1/2: collecting parent drug SMILES from DrugBank...", file=sys.stderr)
    context = etree.iterparse(drugbank_path, events=("end",), huge_tree=True)
    id_to_smiles = {}
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
        smiles = None

        for child in elem:
            tag = local(child.tag)
            if tag == "drugbank-id" and primary_id is None:
                primary_id = child.text
            if tag == "name" and drug_name is None:
                drug_name = child.text
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

        if primary_id and smiles:
            id_to_smiles[primary_id] = (drug_name, smiles)

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if n % 3000 == 0:
            print(f"  ...scanned {n} DrugBank drugs so far", file=sys.stderr)

    print(f"Pass 1 complete: {len(id_to_smiles)} drugs with SMILES\n", file=sys.stderr)
    return id_to_smiles


def compute_similarity(smiles_a, smiles_b, fpgen):
    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return None
        fp_a = fpgen.GetFingerprint(mol_a)
        fp_b = fpgen.GetFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except Exception:
        return None


def scan_and_score(hmdb_path, id_to_smiles, sample_print=20):
    print("Pass 2/2: scanning HMDB, computing similarity for drugbank_id matches...", file=sys.stderr)
    context = etree.iterparse(hmdb_path, events=("end",), huge_tree=True)

    fpgen = GetMorganGenerator(radius=2, fpSize=2048)

    n_total = 0
    n_checked = 0
    n_failed_parse = 0
    results = []  # (hmdb_name, db_id, parent_name, similarity)

    for event, elem in context:
        if local(elem.tag) != "metabolite":
            continue
        parent = elem.getparent()
        if parent is None:
            continue

        n_total += 1

        db_id = None
        name = None
        smiles = None

        for child in elem:
            tag = local(child.tag)
            if tag == "drugbank_id":
                db_id = child.text
            if tag == "name":
                name = child.text
            if tag == "smiles":
                smiles = child.text

        if db_id and smiles and db_id in id_to_smiles:
            parent_name, parent_smiles = id_to_smiles[db_id]
            sim = compute_similarity(parent_smiles, smiles, fpgen)
            n_checked += 1
            if sim is None:
                n_failed_parse += 1
            else:
                results.append((name, db_id, parent_name, sim))

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if n_total % 20000 == 0:
            print(f"  ...scanned {n_total} HMDB entries so far", file=sys.stderr)

    results.sort(key=lambda x: -x[3])  # highest similarity first

    print("\n" + "=" * 65)
    print(f"Total HMDB entries scanned:                     {n_total}")
    print(f"Checked (had matching parent SMILES + own SMILES): {n_checked}")
    print(f"Failed to parse (invalid SMILES):                {n_failed_parse}")
    print(f"Valid similarity scores computed:                {len(results)}")

    if results:
        sims = [r[3] for r in results]
        print(f"\nSimilarity score stats:")
        print(f"  Mean:   {sum(sims)/len(sims):.3f}")
        print(f"  Max:    {max(sims):.3f}")
        print(f"  Min:    {min(sims):.3f}")

        # bucket into rough confidence tiers
        high = [r for r in results if r[3] >= 0.5]
        medium = [r for r in results if 0.2 <= r[3] < 0.5]
        low = [r for r in results if r[3] < 0.2]
        print(f"\n  High similarity (>=0.5, likely genuine metabolite):  {len(high)}")
        print(f"  Medium similarity (0.2-0.5, plausible but uncertain): {len(medium)}")
        print(f"  Low similarity (<0.2, likely NOT a real metabolite):  {len(low)}")

    print(f"\nTop {sample_print} highest-similarity matches:")
    for name, db_id, parent_name, sim in results[:sample_print]:
        print(f"  [{sim:.3f}] {name}  <-- metabolite of {parent_name} ({db_id})")

    print(f"\nBottom {sample_print} lowest-similarity matches (likely false positives):")
    for name, db_id, parent_name, sim in results[-sample_print:]:
        print(f"  [{sim:.3f}] {name}  <-- claimed metabolite of {parent_name} ({db_id})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 metabolite_similarity_check.py /path/to/hmdb_metabolites.xml /path/to/drugbank_full_database.xml")
        sys.exit(1)
    hmdb_path = sys.argv[1]
    drugbank_path = sys.argv[2]

    id_to_smiles = get_drugbank_smiles(drugbank_path)
    scan_and_score(hmdb_path, id_to_smiles)
