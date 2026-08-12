"""
Quick inspector for a DrugBank full_database.xml file.
Checks: how many drugs have metabolites, drug-interactions, SMILES/InChIKey,
and prints a sample of each so you can eyeball the structure.

Usage:
    python3 inspect_drugbank.py /path/to/drugbank_full_database.xml
"""

import sys
from lxml import etree

def local(tag):
    """Strip namespace from a tag, e.g. '{http://www.drugbank.ca}drug' -> 'drug'"""
    return tag.split('}')[-1] if '}' in tag else tag

def main(path, max_drugs_to_scan=2000, sample_print=2):
    print(f"Parsing (streaming): {path}\n")

    context = etree.iterparse(path, events=("end",), huge_tree=True)

    n_drugs = 0
    n_with_metabolites = 0
    n_with_interactions = 0
    n_with_smiles = 0
    n_with_inchikey = 0

    samples_printed = 0

    for event, elem in context:
        if local(elem.tag) != "drug":
            continue

        # only count top-level <drug> entries, not nested ones (e.g. inside <drug-interactions>)
        parent = elem.getparent()
        if parent is None or local(parent.tag) != "drugbank":
            continue

        n_drugs += 1
        if n_drugs > max_drugs_to_scan:
            elem.clear()
            break

        drug_name = None
        has_metabolites = False
        has_interactions = False
        has_smiles = False
        has_inchikey = False
        metabolite_sample = []
        interaction_sample = []

        for child in elem:
            tag = local(child.tag)

            if tag == "name" and drug_name is None:
                drug_name = child.text

            if tag == "metabolites":
                mets = [c for c in child if local(c.tag) == "metabolite"]
                if mets:
                    has_metabolites = True
                    for m in mets[:sample_print]:
                        m_name = None
                        m_inchikey = None
                        for mc in m.iter():
                            mtag = local(mc.tag)
                            if mtag == "name" and m_name is None:
                                m_name = mc.text
                            if mtag == "inchikey":
                                m_inchikey = mc.text
                        metabolite_sample.append((m_name, m_inchikey))

            if tag == "drug-interactions":
                ddis = [c for c in child if local(c.tag) == "drug-interaction"]
                if ddis:
                    has_interactions = True
                    for d in ddis[:sample_print]:
                        d_name, d_desc = None, None
                        for dc in d:
                            dtag = local(dc.tag)
                            if dtag == "name":
                                d_name = dc.text
                            if dtag == "description":
                                d_desc = dc.text
                        interaction_sample.append((d_name, d_desc))

            if tag == "calculated-properties":
                for prop in child:
                    if local(prop.tag) != "property":
                        continue
                    kind, val = None, None
                    for pc in prop:
                        if local(pc.tag) == "kind":
                            kind = pc.text
                        if local(pc.tag) == "value":
                            val = pc.text
                    if kind == "SMILES":
                        has_smiles = True
                    if kind == "InChIKey":
                        has_inchikey = True

        if has_metabolites:
            n_with_metabolites += 1
        if has_interactions:
            n_with_interactions += 1
        if has_smiles:
            n_with_smiles += 1
        if has_inchikey:
            n_with_inchikey += 1

        if samples_printed < sample_print and (has_metabolites or has_interactions):
            print(f"--- Sample drug: {drug_name} ---")
            if metabolite_sample:
                print("  Metabolites:")
                for mn, mk in metabolite_sample:
                    print(f"    - {mn}  (InChIKey: {mk})")
            if interaction_sample:
                print("  Drug interactions:")
                for dn, dd in interaction_sample:
                    desc_short = (dd[:120] + "...") if dd and len(dd) > 120 else dd
                    print(f"    - with {dn}: {desc_short}")
            print()
            samples_printed += 1

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    print("=" * 50)
    print(f"Drugs scanned:              {n_drugs}")
    print(f"  with >=1 metabolite:      {n_with_metabolites}")
    print(f"  with >=1 drug-interaction:{n_with_interactions}")
    print(f"  with SMILES:              {n_with_smiles}")
    print(f"  with InChIKey:            {n_with_inchikey}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_drugbank.py /path/to/drugbank_full_database.xml")
        sys.exit(1)
    main(sys.argv[1])
