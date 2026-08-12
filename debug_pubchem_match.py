"""
Debug script: prints the actual <resource> names found under
<external-identifiers> in DrugBank (so we can see the EXACT string used for
PubChem, e.g. "PubChem Compound" vs "PubChem Substance"), plus a handful of
sample CIDs from both DrugBank and your TWOSIDES drugs.csv, so we can compare
formats directly instead of guessing.

Usage:
    python3 debug_pubchem_match.py /path/to/drugbank_full_database.xml /path/to/drugs.csv
"""

import sys
import csv
import re
from collections import Counter
from lxml import etree

def local(tag):
    return tag.split('}')[-1] if '}' in tag else tag


def debug_drugbank(drugbank_path, max_drugs=500, sample_print=10):
    print(f"Scanning first {max_drugs} DrugBank drugs for external-identifier resource names...\n")
    context = etree.iterparse(drugbank_path, events=("end",), huge_tree=True)

    resource_counter = Counter()
    pubchem_samples = []
    n = 0

    for event, elem in context:
        if local(elem.tag) != "drug":
            continue
        parent = elem.getparent()
        if parent is None or local(parent.tag) != "drugbank":
            continue

        n += 1
        if n > max_drugs:
            elem.clear()
            break

        drug_name = None
        for child in elem:
            tag = local(child.tag)
            if tag == "name" and drug_name is None:
                drug_name = child.text
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
                    if resource:
                        resource_counter[resource] += 1
                        if "PubChem" in resource and len(pubchem_samples) < sample_print:
                            pubchem_samples.append((drug_name, resource, identifier))

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    print("All distinct 'resource' values seen under external-identifiers:")
    for resource, count in resource_counter.most_common(30):
        print(f"  {resource!r:35s} count={count}")

    print(f"\nSample PubChem-related entries (name, resource, identifier):")
    for name, resource, identifier in pubchem_samples:
        print(f"  {name!r:30s} | resource={resource!r} | identifier={identifier!r}")


def debug_twosides(drugs_csv_path, sample_print=10):
    print(f"\n\nSample TWOSIDES drug_id values from drugs.csv:")
    with open(drugs_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        print(f"  Columns found: {reader.fieldnames}")
        for i, row in enumerate(reader):
            if i >= sample_print:
                break
            drug_id = row.get("drug_id")
            m = re.match(r'CID0*(\d+)', drug_id) if drug_id else None
            extracted = m.group(1) if m else None
            print(f"  raw drug_id={drug_id!r}  -> extracted CID={extracted!r}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 debug_pubchem_match.py /path/to/drugbank_full_database.xml /path/to/drugs.csv")
        sys.exit(1)
    debug_drugbank(sys.argv[1])
    debug_twosides(sys.argv[2])
