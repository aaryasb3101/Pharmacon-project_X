"""
Dumps the actual tag structure of the FIRST <drug> element in a DrugBank XML file.
No guessing — this shows you every child/grandchild tag that genuinely exists,
so you can see exactly how metabolites/interactions/properties are nested.

Usage:
    python3 dump_structure.py /path/to/drugbank_full_database.xml
"""

import sys
from lxml import etree

def local(tag):
    return tag.split('}')[-1] if '}' in tag else tag

def print_tree(elem, depth=0, max_depth=4):
    if depth > max_depth:
        return
    tag = local(elem.tag)
    text_preview = ""
    if elem.text and elem.text.strip():
        t = elem.text.strip()
        text_preview = f'  = "{t[:40]}"' if len(t) <= 40 else f'  = "{t[:40]}..."'
    print("  " * depth + f"<{tag}>{text_preview}")
    # only recurse into unique child tags once each, to keep output short
    seen = set()
    for child in elem:
        ctag = local(child.tag)
        if ctag in seen:
            continue
        seen.add(ctag)
        print_tree(child, depth + 1, max_depth)

def main(path, max_depth=4):
    context = etree.iterparse(path, events=("end",), huge_tree=True)
    for event, elem in context:
        if local(elem.tag) != "drug":
            continue
        parent = elem.getparent()
        if parent is None or local(parent.tag) != "drugbank":
            continue
        print(f"=== Full structure of first <drug> element (depth-limited to {max_depth}) ===\n")
        print_tree(elem, 0, max_depth)
        break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 dump_structure.py /path/to/drugbank_full_database.xml")
        sys.exit(1)
    main(sys.argv[1])
