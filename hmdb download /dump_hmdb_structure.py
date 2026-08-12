"""
Dumps the actual tag structure of the FIRST metabolite element in the HMDB
'All Metabolites' XML file. No guessing -- shows real tag names so we know
exactly how to check for DrugBank cross-references and SMILES/InChIKey.

Usage:
    python3 dump_hmdb_structure.py /path/to/hmdb_metabolites.xml
"""

import sys
from lxml import etree

def local(tag):
    return tag.split('}')[-1] if '}' in tag else tag

def print_tree(elem, depth=0, max_depth=3):
    if depth > max_depth:
        return
    tag = local(elem.tag)
    text_preview = ""
    if elem.text and elem.text.strip():
        t = elem.text.strip()
        text_preview = f'  = "{t[:50]}"' if len(t) <= 50 else f'  = "{t[:50]}..."'
    print("  " * depth + f"<{tag}>{text_preview}")
    seen = set()
    for child in elem:
        ctag = local(child.tag)
        if ctag in seen:
            continue
        seen.add(ctag)
        print_tree(child, depth + 1, max_depth)

def main(path, max_depth=3):
    context = etree.iterparse(path, events=("end",), huge_tree=True)
    count = 0
    for event, elem in context:
        tag = local(elem.tag)
        # HMDB entries are typically <metabolite> elements under a root like <hmdb>
        if tag != "metabolite":
            continue
        parent = elem.getparent()
        # only take top-level metabolite entries, not nested references to other metabolites
        if parent is None:
            continue

        count += 1
        print(f"=== Structure of metabolite entry #{count} (depth-limited to {max_depth}) ===\n")
        print_tree(elem, 0, max_depth)
        if count >= 1:
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 dump_hmdb_structure.py /path/to/hmdb_metabolites.xml")
        sys.exit(1)
    main(sys.argv[1])
