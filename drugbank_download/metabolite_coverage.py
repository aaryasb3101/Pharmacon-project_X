"""
Counts how many "proper" (small-molecule, organic) parent drugs have at least
one metabolite reaction, based on the <reactions> block pattern discovered
in the DrugBank XML (metabolite = right-element with a DBMET-prefixed id).

A drug is considered "proper" / organic here if it has a SMILES string in its
calculated-properties (i.e. it's a small molecule with real structure, not a
biologic/peptide/protein entry like Lepirudin which has no SMILES).

Usage:
    python3 metabolite_coverage.py /path/to/drugbank_full_database.xml [max_drugs]
"""

import sys
from lxml import etree

def local(tag):
    return tag.split('}')[-1] if '}' in tag else tag

def main(path, max_drugs=None):
    context = etree.iterparse(path, events=("end",), huge_tree=True)

    n_total = 0
    n_organic = 0          # has SMILES (i.e. small molecule, not biologic)
    n_organic_with_met = 0 # organic AND has >=1 metabolite reaction
    n_inorganic_skipped = 0

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

        has_smiles = False
        has_metabolite_reaction = False

        for child in elem:
            tag = local(child.tag)

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

            if tag == "reactions":
                for reaction in child:
                    if local(reaction.tag) != "reaction":
                        continue
                    for rc in reaction:
                        if local(rc.tag) == "right-element":
                            for rec in rc:
                                if local(rec.tag) == "drugbank-id" and rec.text and rec.text.startswith("DBMET"):
                                    has_metabolite_reaction = True

        if has_smiles:
            n_organic += 1
            if has_metabolite_reaction:
                n_organic_with_met += 1
        else:
            n_inorganic_skipped += 1

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if n_total % 1000 == 0:
            print(f"  ...scanned {n_total} drugs so far", file=sys.stderr)

    print("=" * 55)
    print(f"Total drugs scanned:                     {n_total}")
    print(f"Skipped (no SMILES / biologic-like):     {n_inorganic_skipped}")
    print(f"'Proper' organic drugs (has SMILES):     {n_organic}")
    print(f"  ...of those, with >=1 metabolite rxn:  {n_organic_with_met}")
    if n_organic > 0:
        pct = 100 * n_organic_with_met / n_organic
        print(f"  ...percentage with metabolites:        {pct:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 metabolite_coverage.py /path/to/drugbank_full_database.xml [max_drugs]")
        sys.exit(1)
    max_d = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(sys.argv[1], max_d)
