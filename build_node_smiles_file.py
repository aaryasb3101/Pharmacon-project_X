"""
Joins gnn_node_mapping.csv (drug_id, node_index) with drugs.csv (drug_id, smiles)
to produce a single file: node_index, drug_id, smiles -- this is the input
your SMILES Transformer (and per-molecule GCN, via RDKit conversion at
data-load time) actually needs.

Usage:
    python3 build_node_smiles_file.py /path/to/gnn_node_mapping.csv /path/to/drugs.csv /path/to/output_node_smiles.csv
"""

import sys
import csv


def main(node_mapping_path, drugs_path, output_path):
    # Load drug_id -> smiles
    id_to_smiles = {}
    with open(drugs_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug_id = row.get("drug_id")
            smiles = row.get("smiles")
            if drug_id and smiles:
                id_to_smiles[drug_id] = smiles

    print(f"Loaded {len(id_to_smiles)} drug_id -> smiles pairs from drugs.csv", file=sys.stderr)

    # Load node mapping and join
    rows_out = []
    n_missing = 0
    with open(node_mapping_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug_id = row.get("drug_id")
            node_index = row.get("node_index")
            smiles = id_to_smiles.get(drug_id)
            if smiles is None:
                n_missing += 1
            rows_out.append({
                "node_index": node_index,
                "drug_id": drug_id,
                "smiles": smiles or ""
            })

    # sort by node_index (as int) so the file is ordered 0, 1, 2, ... -- convenient
    # for directly indexing into a feature matrix later
    rows_out.sort(key=lambda r: int(r["node_index"]))

    with open(output_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["node_index", "drug_id", "smiles"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nTotal nodes in mapping:      {len(rows_out)}")
    print(f"Nodes missing a SMILES:      {n_missing}")
    print(f"Nodes with valid SMILES:     {len(rows_out) - n_missing}")
    print(f"\nOutput written to: {output_path}")

    if n_missing > 0:
        print(f"\nWARNING: {n_missing} drug_ids in your node mapping had no matching "
              f"SMILES in drugs.csv -- check these before feeding into your Transformer, "
              f"since they'll break tokenization if left as empty strings.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 build_node_smiles_file.py /path/to/gnn_node_mapping.csv /path/to/drugs.csv /path/to/output_node_smiles.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
