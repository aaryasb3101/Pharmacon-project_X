import csv
import sys
from pathlib import Path
from lxml import etree

def local(tag):
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def load_twosides(csv_path):
    print("Loading TWOSIDES drugs.csv...", file=sys.stderr)
    drugs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        print(f"  Detected CSV Columns: {reader.fieldnames}", file=sys.stderr)
        
        for i, row in enumerate(reader):
            # Print the first row to see what values are actually mapped
            if i == 0:
                print(f"  Sample first row parsed: {row}", file=sys.stderr)
                
            # Flexible key lookup across different naming conventions
            drug_id = row.get("drug_id") or row.get("id") or row.get("twosides_drug_id") or list(row.values())[0]
            smiles = row.get("smiles") or row.get("SMILES") or row.get("smile")
            inchikey = row.get("inchikey") or row.get("InChIKey") or row.get("inchikey_1")
            cid = row.get("pubchem_cid") or row.get("cid") or row.get("CID") or row.get("pubchem")
            
            clean_cid = None
            if cid:
                cid_str = str(cid).strip()
                if cid_str.lower().startswith("cid"):
                    clean_cid = cid_str[3:].lstrip("0")
                elif cid_str.isdigit():
                    clean_cid = str(int(cid_str))  # strips leading zeros reliably
                    
            clean_ikey = str(inchikey).strip() if inchikey else None
            clean_smiles = str(smiles).strip() if smiles else None
            
            drugs.append((drug_id, clean_cid, clean_smiles, clean_ikey))
            
    print(f"  Loaded {len(drugs)} TWOSIDES drugs\n", file=sys.stderr)
    return drugs

def build_drugbank_indexes(drugbank_path):
    print("Scanning DrugBank for PubChem CIDs and InChIKeys...", file=sys.stderr)
    context = etree.iterparse(drugbank_path, events=("end",), huge_tree=True)
    cid_index = {}
    inchikey_index = {}
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
        inchikey = None
        pubchem_cid = None

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
            inchikey_index[inchikey.strip()] = drug_record
            base_ikey = inchikey.split('-')[0]
            inchikey_index[base_ikey] = drug_record

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if n % 3000 == 0:
            print(f"  ...scanned {n} DrugBank drugs so far", file=sys.stderr)

    print(f"DrugBank scan complete: {n} drugs, {len(cid_index)} CIDs, {len(inchikey_index)} InChIKeys indexed\n", file=sys.stderr)
    return cid_index, inchikey_index

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
        print(f"Error: Could not find drugbank_full_database.xml anywhere on your system.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found DrugBank XML at: {drugbank_path}", file=sys.stderr)
    output_path = script_dir / "twosides_drugbank_mapping.csv"

    twosides_drugs = load_twosides(str(drugs_csv_path))
    cid_index, inchikey_index = build_drugbank_indexes(str(drugbank_path))

    results = []
    n_matched = 0
    n_inchikey_confirmed = 0

    for drug_id, cid, smiles, ikey in twosides_drugs:
        db_id, db_name, db_smiles, db_inchikey = (None, None, None, None)
        matched = False
        inchikey_match = False
        match_method = "None"

        if cid and cid in cid_index:
            db_id, db_name, db_smiles, db_inchikey = cid_index[cid]
            matched = True
            n_matched += 1
            match_method = "PubChem CID"
            if ikey and db_inchikey and ikey.strip() == db_inchikey.strip():
                inchikey_match = True
                n_inchikey_confirmed += 1
        elif ikey and ikey.strip() in inchikey_index:
            db_id, db_name, db_smiles, db_inchikey = inchikey_index[ikey.strip()]
            matched = True
            n_matched += 1
            match_method = "InChIKey Fallback"
            inchikey_match = True
            n_inchikey_confirmed += 1

        results.append({
            "twosides_drug_id": drug_id,
            "twosides_smiles": smiles,
            "matched_drugbank_id": db_id or "",
            "matched_drugbank_name": db_name or "",
            "drugbank_smiles": db_smiles or "",
            "inchikey_cross_check_passed": inchikey_match,
            "match_method": match_method
        })

    with open(output_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print(f"Total TWOSIDES drugs:                          {len(twosides_drugs)}")
    print(f"Matched to a DrugBank entry:                   {n_matched}")
    print(f"  ...of those, InChIKey confirmed/matched:      {n_inchikey_confirmed}")
    if len(twosides_drugs) > 0:
        pct = 100 * n_matched / len(twosides_drugs)
        print(f"\n% of TWOSIDES drugs matched to DrugBank: {pct:.1f}%")
    print(f"\nFull mapping written to: {output_path}")

if __name__ == "__main__":
    main()