"""
Filters BioTransformer output CSV files (one per drug, all in a folder) to
keep only rows where the 'Enzyme(s)' column mentions a CYP enzyme
(e.g. CYP3A4, CYP2D6, CYP2C9, etc.) -- since CYP450 enzymes are the dominant
real-world drug-metabolizing enzymes, and are most relevant for DDI-style
metabolism (e.g. one drug inhibiting the CYP enzyme another drug depends on).

Usage:
    python3 filter_cyp_metabolites.py /path/to/bt_results_folder /path/to/cyp_only_output.csv
"""

import sys
import csv
import os
import re

ENZYME_COL = "Enzyme(s)"
CYP_PATTERN = re.compile(r'\bCYP\w*', re.IGNORECASE)


def main(biotransformer_folder, output_path):
    csv_files = sorted(
        f for f in os.listdir(biotransformer_folder)
        if f.lower().endswith(".csv")
    )
    print(f"Found {len(csv_files)} CSV files in {biotransformer_folder}", file=sys.stderr)

    total_rows = 0
    total_cyp_rows = 0
    n_files_processed = 0
    n_files_skipped = 0
    writer = None
    out_f = None

    try:
        out_f = open(output_path, "w", newline='', encoding='utf-8')

        for i, fname in enumerate(csv_files, 1):
            fpath = os.path.join(biotransformer_folder, fname)
            try:
                with open(fpath, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames

                    if ENZYME_COL not in fieldnames:
                        print(f"  WARNING: {fname} missing '{ENZYME_COL}' column, skipping.", file=sys.stderr)
                        n_files_skipped += 1
                        continue

                    if writer is None:
                        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                        writer.writeheader()

                    for row in reader:
                        total_rows += 1
                        enzyme_field = row.get(ENZYME_COL) or ""
                        if CYP_PATTERN.search(enzyme_field):
                            total_cyp_rows += 1
                            writer.writerow(row)

                n_files_processed += 1
            except Exception as e:
                print(f"  ERROR processing {fname}: {e}", file=sys.stderr)
                n_files_skipped += 1

            if i % 50 == 0:
                print(f"  ...processed {i}/{len(csv_files)} files so far", file=sys.stderr)
    finally:
        if out_f:
            out_f.close()

    print("\n" + "=" * 60)
    print(f"Files processed successfully:    {n_files_processed}")
    print(f"Files skipped:                   {n_files_skipped}")
    print(f"Total rows scanned:              {total_rows}")
    print(f"Rows with a CYP enzyme match:    {total_cyp_rows}")
    if total_rows > 0:
        print(f"  ({100 * total_cyp_rows / total_rows:.1f}% of all rows)")
    print(f"\nCYP-only metabolites written to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 filter_cyp_metabolites.py /path/to/bt_results_folder /path/to/cyp_only_output.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
