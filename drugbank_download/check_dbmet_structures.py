"""
Checks whether DBMET-prefixed metabolite IDs (referenced inside <reactions>
as <right-element>) exist ELSEWHERE in the file as their own entries with
structural data (SMILES / InChIKey) — e.g. as a top-level <drug> with
type="metabolite", or some other dedicated metabolite record.

This tells you whether you can actually get a metabolite's own SMILES/InChIKey
for your gating logic, or whether DrugBank only gives you the metabolite's
name/ID with no structure attached.

Usage:
    python3 check_dbmet_structures.py /path/to/drugbank_full_database.xml [max_drugs]
"""

import sys
from lxml import etree

def local(tag):
    return tag.split('}')[-1] if '}' in tag else tag

def main(path, max_drugs=None, sample_print=5):
    context = etree.iterparse(path, events=("end",), huge_tree=True)

    n_total = 0
    dbmet_ids_referenced = set()   # collected from <reactions><right-element>
    dbmet_ids_found_as_entry = set()  # DBMET ids that appear as their OWN <drug> element
    dbmet_with_smiles = set()
    dbmet_with_inchikey = set()
    samples_printed = 0

    for event, elem in context:
        if local(elem.tag) != "drug":
            continue
        parent = elem.getparent()
        if parent is None or local(parent.tag) != "drugbank":
            continue

        n_total += 1
        if max_drugs and n_total > max_drugs:
            elem.clear()
            break

        # get this drug's own id(s) -- primary + any secondary drugbank-ids
        own_ids = []
        has_smiles = False
        has_inchikey = False

        for child in elem:
            tag = local(child.tag)

            if tag == "drugbank-id":
                if child.text:
                    own_ids.append(child.text)

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
                    if kind == "InChIKey":
                        has_inchikey = True

            if tag == "reactions":
                for reaction in child:
                    if local(reaction.tag) != "reaction":
                        continue
                    for rc in reaction:
                        if local(rc.tag) == "right-element":
                            for rec in rc:
                                if local(rec.tag) == "drugbank-id" and rec.text and rec.text.startswith("DBMET"):
                                    dbmet_ids_referenced.add(rec.text)

        # if THIS drug entry's own id happens to be a DBMET id, record it
        for oid in own_ids:
            if oid.startswith("DBMET"):
                dbmet_ids_found_as_entry.add(oid)
                if has_smiles:
                    dbmet_with_smiles.add(oid)
                if has_inchikey:
                    dbmet_with_inchikey.add(oid)

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if n_total % 2000 == 0:
            print(f"  ...scanned {n_total} drugs so far", file=sys.stderr)

    overlap = dbmet_ids_referenced & dbmet_ids_found_as_entry

    print("=" * 55)
    print(f"Total drugs scanned:                          {n_total}")
    print(f"Unique DBMET ids referenced in <reactions>:   {len(dbmet_ids_referenced)}")
    print(f"DBMET ids that ALSO appear as own <drug> entry:{len(dbmet_ids_found_as_entry)}")
    print(f"  ...of those, overlap with referenced ones:  {len(overlap)}")
    print(f"  ...of those, with SMILES:                   {len(dbmet_with_smiles)}")
    print(f"  ...of those, with InChIKey:                 {len(dbmet_with_inchikey)}")

    if len(dbmet_ids_referenced) > 0:
        pct_found = 100 * len(overlap) / len(dbmet_ids_referenced)
        print(f"\n% of referenced metabolites found as full entries: {pct_found:.1f}%")

    print("\nSample DBMET ids referenced but check first few:")
    for i, mid in enumerate(list(dbmet_ids_referenced)[:sample_print]):
        found = "FOUND as own entry" if mid in dbmet_ids_found_as_entry else "NOT found as own entry"
        print(f"  {mid}: {found}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 check_dbmet_structures.py /path/to/drugbank_full_database.xml [max_drugs]")
        sys.exit(1)
    max_d = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(sys.argv[1], max_d)
