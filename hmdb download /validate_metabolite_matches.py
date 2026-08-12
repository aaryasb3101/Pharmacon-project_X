"""
Validates whether HMDB entries with a drugbank_id are GENUINE metabolites of
that parent drug, or just coincidental identity matches (same compound also
separately listed as a drug in DrugBank).

Method: for each parent drug in DrugBank, collect the set of metabolite NAMES
listed in its <reactions> block (the DBMET-referenced names). Then, for each
HMDB entry whose drugbank_id matches that SAME parent drug, check whether the
HMDB entry's name (or synonyms) appears in that parent's DBMET name list.

A match = genuine, independently-confirmed parent->metabolite relationship.
No match = likely just an identity cross-reference, not a real metabolite link.

Usage:
    python3 validate_metabolite_matches.py /path/to/hmdb_metabolites.xml /path/to/drugbank_full_database.xml
"""

import sys
import re
from lxml import etree

def local(tag):
    return tag.split('}')[-1] if '}' in tag else tag

def normalize(name):
    """Loose normalization for name comparison: lowercase, strip brackets/parens/stereo prefixes."""
    if not name:
        return ""
    n = name.lower().strip()
    n = re.sub(r'^\(\+?/?-?\)?-?', '', n)          # strip leading (S)-, (+)-, (R,S)- etc
    n = re.sub(r'\[.*?\]|\(.*?\)', '', n)          # strip anything in [] or ()
    n = re.sub(r'[^a-z0-9]+', ' ', n)              # collapse punctuation to spaces
    n = re.sub(r'\s+', ' ', n).strip()
    return n


def get_drugbank_metabolite_names(drugbank_path):
    """
    Returns: dict mapping parent_drug_id -> set of normalized metabolite names
             (from <reactions><right-element> where id starts with DBMET)
    Also returns: dict mapping parent_drug_id -> parent_drug_name (for readability)
    """
    print("Pass 1/2: collecting DBMET metabolite names per parent drug from DrugBank...", file=sys.stderr)
    context = etree.iterparse(drugbank_path, events=("end",), huge_tree=True)

    parent_to_met_names = {}
    parent_names = {}
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
        met_names = set()

        for child in elem:
            tag = local(child.tag)
            if tag == "drugbank-id" and primary_id is None:
                primary_id = child.text
            if tag == "name" and drug_name is None:
                drug_name = child.text
            if tag == "reactions":
                for reaction in child:
                    if local(reaction.tag) != "reaction":
                        continue
                    for rc in reaction:
                        if local(rc.tag) == "right-element":
                            met_id, met_name = None, None
                            for rec in rc:
                                if local(rec.tag) == "drugbank-id":
                                    met_id = rec.text
                                if local(rec.tag) == "name":
                                    met_name = rec.text
                            if met_id and met_id.startswith("DBMET") and met_name:
                                met_names.add(normalize(met_name))

        if primary_id:
            parent_names[primary_id] = drug_name
            if met_names:
                parent_to_met_names[primary_id] = met_names

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if n % 3000 == 0:
            print(f"  ...scanned {n} DrugBank drugs so far", file=sys.stderr)

    print(f"Pass 1 complete: {len(parent_to_met_names)} parent drugs have >=1 named metabolite reaction\n", file=sys.stderr)
    return parent_to_met_names, parent_names


def validate_hmdb_matches(hmdb_path, parent_to_met_names, parent_names, sample_print=15):
    print("Pass 2/2: scanning HMDB, checking name overlap against DBMET reaction names...", file=sys.stderr)
    context = etree.iterparse(hmdb_path, events=("end",), huge_tree=True)

    n_total = 0
    n_with_drugbank_id = 0
    n_parent_has_reactions_listed = 0   # drugbank_id matches a parent that HAS reaction data at all
    n_genuine_match = 0                  # AND the hmdb name matches one of those reaction names
    genuine_samples = []
    non_genuine_samples = []

    for event, elem in context:
        if local(elem.tag) != "metabolite":
            continue
        parent = elem.getparent()
        if parent is None:
            continue

        n_total += 1

        db_id = None
        name = None
        synonyms = []
        smiles = None
        inchikey = None

        for child in elem:
            tag = local(child.tag)
            if tag == "drugbank_id":
                db_id = child.text
            if tag == "name":
                name = child.text
            if tag == "smiles":
                smiles = child.text
            if tag == "inchikey":
                inchikey = child.text
            if tag == "synonyms":
                for syn in child:
                    if local(syn.tag) == "synonym" and syn.text:
                        synonyms.append(syn.text)

        if db_id:
            n_with_drugbank_id += 1

            if db_id in parent_to_met_names:
                n_parent_has_reactions_listed += 1
                candidate_names = {normalize(name)} | {normalize(s) for s in synonyms}
                met_names_for_parent = parent_to_met_names[db_id]

                if candidate_names & met_names_for_parent:
                    n_genuine_match += 1
                    if len(genuine_samples) < sample_print:
                        genuine_samples.append((name, db_id, parent_names.get(db_id), smiles, inchikey))
                else:
                    if len(non_genuine_samples) < sample_print:
                        non_genuine_samples.append((name, db_id, parent_names.get(db_id)))

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if n_total % 20000 == 0:
            print(f"  ...scanned {n_total} HMDB entries so far", file=sys.stderr)

    print("\n" + "=" * 65)
    print(f"Total HMDB entries scanned:                                {n_total}")
    print(f"HMDB entries with a drugbank_id:                           {n_with_drugbank_id}")
    print(f"  ...where that parent drug HAS reaction/metabolite data:  {n_parent_has_reactions_listed}")
    print(f"  ...AND the name genuinely matches a DBMET reaction name: {n_genuine_match}")
    if n_with_drugbank_id > 0:
        pct = 100 * n_genuine_match / n_with_drugbank_id
        print(f"\n% of drugbank_id matches that are GENUINE metabolite links: {pct:.1f}%")

    print("\n--- Sample GENUINE matches (name confirmed via DrugBank reaction data) ---")
    for name, db_id, parent_name, smiles, inchikey in genuine_samples:
        print(f"  {name}  (metabolite of {parent_name}, {db_id})")
        print(f"    SMILES: {smiles}")
        print(f"    InChIKey: {inchikey}")

    print("\n--- Sample NON-genuine / unconfirmed matches (likely identity cross-ref only) ---")
    for name, db_id, parent_name in non_genuine_samples:
        print(f"  {name}  (drugbank_id points to {parent_name}, {db_id}, but name not found in its reaction list)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 validate_metabolite_matches.py /path/to/hmdb_metabolites.xml /path/to/drugbank_full_database.xml")
        sys.exit(1)
    hmdb_path = sys.argv[1]
    drugbank_path = sys.argv[2]

    parent_to_met_names, parent_names = get_drugbank_metabolite_names(drugbank_path)
    validate_hmdb_matches(hmdb_path, parent_to_met_names, parent_names)
