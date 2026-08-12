"""
STEP 4 of 4: Merges all per-drug single-metabolite files (from step 3, one row
each) into one final combined CSV -- your finished dataset, one row per drug
that successfully produced a primary CYP metabolite.

Also tags each row with the parent drug_id and node_index (parsed from the
filename, e.g. "24_CID000004920.csv" -> node_index=24, drug_id=CID000004920),
matching the same convention used in your original merge() function.

Usage:
    python3 step4_merge_final.py single_metabolite_results final_metabolites.csv
"""

import sys
import csv
import os


def main(single_results_folder, output_path):
    csv_files = sorted(
        f for f in os.listdir(single_results_folder)
        if f.lower().endswith(".csv")
    )
    print(f"Found {len(csv_files)} single-metabolite files to merge", file=sys.stderr)

    writer = None
    n_written = 0

    with open(output_path, "w", newline='', encoding='utf-8') as out_f:
        for fname in csv_files:
            fpath = os.path.join(single_results_folder, fname)
            try:
                node_index, rest = fname.split("_", 1)
                drug_id = rest.rsplit(".csv", 1)[0]
            except ValueError:
                node_index, drug_id = "", fname

            try:
                with open(fpath, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row["parent_node_index"] = node_index
                        row["parent_drug_id"] = drug_id
                        if writer is None:
                            fieldnames = ["parent_node_index", "parent_drug_id"] + [
                                k for k in row.keys() if k not in ("parent_node_index", "parent_drug_id")
                            ]
                            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                            writer.writeheader()
                        writer.writerow(row)
                        n_written += 1
            except Exception as e:
                print(f"  (skipping unreadable {fname}: {e})", file=sys.stderr)

    print(f"\nWrote {n_written} final rows (one per drug) to {output_path}")
    print(f"Coverage: {n_written} / 645 original TWOSIDES drugs have a final primary CYP metabolite")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 step4_merge_final.py single_metabolite_results final_metabolites.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
