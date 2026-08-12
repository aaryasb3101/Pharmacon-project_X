"""
Scans the full HMDB 'All Metabolites' XML file, extracting every entry that
has a populated <drugbank_id> field, along with its SMILES/InChIKey.

Then cross-references those DrugBank IDs against your own DrugBank XML's list
of parent drug IDs (specifically the "proper" organic ones with SMILES) to see
how many actually overlap.

Usage:
    python3 hmdb_drugbank_overlap.py /path/to/hmdb_metabolites.xml /path/to/drugbank_full_database.xml
"""

import sys
from lxml import etree

def local(tag):
    return tag.split('}')[-1] if '}' in tag else tag


def get_drugbank_organic_ids(drugbank_path):
    """Returns set of DrugBank IDs for drugs that have a SMILES (i.e. 'proper' organic drugs)."""
    print("Scanning DrugBank file for organic parent drug IDs...", file=sys.stderr)
    context = etree.iterparse(drugbank_path, events=("end",), huge_tree=True)
    organic_ids = set()
    all_ids = set()
    n = 0

    for event, elem in context:
        if local(elem.tag) != "drug":
            continue
        parent = elem.getparent()
        if parent is None or local(parent.tag) != "drugbank":
            continue

        n += 1
        primary_id = None
        has_smiles = False

        for child in elem:
            tag = local(child.tag)
            if tag == "drugbank-id" and primary_id is None:
                primary_id = child.text
            if tag == "calculated-properties":
                for prop in child:
                    if local(prop.tag) != "property":
                        continue
                    kind = None
                    for pc in prop:
                        if local(pc.tag) == "kind":
                            kind = pc.text
                    if kind == "SMILES":
                        has_smiles = True

        if primary_id:
            all_ids.add(primary_id)
            if has_smiles:
                organic_ids.add(primary_id)

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if n % 3000 == 0:
            print(f"  ...scanned {n} DrugBank drugs so far", file=sys.stderr)

    print(f"DrugBank scan complete: {n} total drugs, {len(organic_ids)} organic (SMILES) drugs\n", file=sys.stderr)
    return organic_ids, all_ids


def scan_hmdb(hmdb_path, organic_ids, all_ids, sample_print=5):
    print("Scanning HMDB file for drugbank_id cross-references...", file=sys.stderr)
    context = etree.iterparse(hmdb_path, events=("end",), huge_tree=True)

    n_total = 0
    n_with_drugbank_id = 0
    n_matched_organic = 0
    n_matched_any = 0
    n_with_smiles = 0
    samples = []

    for event, elem in context:
        if local(elem.tag) != "metabolite":
            continue
        parent = elem.getparent()
        if parent is None:
            continue

        n_total += 1

        db_id = None
        smiles = None
        inchikey = None
        name = None

        for child in elem:
            tag = local(child.tag)
            if tag == "drugbank_id":
                db_id = child.text
            if tag == "smiles":
                smiles = child.text
            if tag == "inchikey":
                inchikey = child.text
            if tag == "name":
                name = child.text

        if db_id:
            n_with_drugbank_id += 1
            if smiles:
                n_with_smiles += 1
            if db_id in all_ids:
                n_matched_any += 1
            if db_id in organic_ids:
                n_matched_organic += 1
                if len(samples) < sample_print:
                    samples.append((name, db_id, smiles, inchikey))

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if n_total % 20000 == 0:
            print(f"  ...scanned {n_total} HMDB entries so far", file=sys.stderr)

    print("\n" + "=" * 60)
    print(f"Total HMDB entries scanned:                      {n_total}")
    print(f"HMDB entries with a drugbank_id populated:        {n_with_drugbank_id}")
    print(f"  ...of those, with SMILES also present:          {n_with_smiles}")
    print(f"  ...matched to ANY DrugBank id (organic or not): {n_matched_any}")
    print(f"  ...matched to an ORGANIC DrugBank parent drug:  {n_matched_organic}")
    if len(organic_ids) > 0:
        pct = 100 * n_matched_organic / len(organic_ids)
        print(f"\n% of organic DrugBank parent drugs with a matched HMDB structure: {pct:.1f}%")

    print("\nSample matches:")
    for name, db_id, smiles, inchikey in samples:
        print(f"  {name} -> {db_id}")
        print(f"    SMILES: {smiles}")
        print(f"    InChIKey: {inchikey}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 hmdb_drugbank_overlap.py /path/to/hmdb_metabolites.xml /path/to/drugbank_full_database.xml")
        sys.exit(1)
    hmdb_path = sys.argv[1]
    drugbank_path = sys.argv[2]

    organic_ids, all_ids = get_drugbank_organic_ids(drugbank_path)
    scan_hmdb(hmdb_path, organic_ids, all_ids)