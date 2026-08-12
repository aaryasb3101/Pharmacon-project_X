"""
STEP 2 of 4: Filters each per-drug PRIMARY-metabolite file (from step 1) down
to rows where the 'Enzyme(s)' column mentions a CYP enzyme, writing one
filtered file per drug into --outdir.

Usage:
    python3 step2_filter_cyp.py primary_results cyp_results
"""

import sys
import csv
import os
import re

ENZYME_COL = "Enzyme(s)"
CYP_PATTERN = re.compile(r'\bCYP\w*', re.IGNORECASE)


def main(primary_results_folder, outdir):
    os.makedirs(outdir, exist_ok=True)

    csv_files = sorted(
        f for f in os.listdir(primary_results_folder)
        if f.lower().endswith(".csv")
    )
    print(f"Found {len(csv_files)} per-drug primary-metabolite files", file=sys.stderr)

    total_cyp = 0
    total_non_cyp = 0
    n_files_with_cyp = 0
    n_files_no_cyp = 0
    n_files_skipped = 0

    for i, fname in enumerate(csv_files, 1):
        fpath = os.path.join(primary_results_folder, fname)
        try:
            with open(fpath, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                if not fieldnames or ENZYME_COL not in fieldnames:
                    n_files_skipped += 1
                    continue

                kept_rows = []
                for row in reader:
                    enzyme_field = row.get(ENZYME_COL) or ""
                    if CYP_PATTERN.search(enzyme_field):
                        kept_rows.append(row)
                        total_cyp += 1
                    else:
                        total_non_cyp += 1

            if kept_rows:
                out_path = os.path.join(outdir, fname)
                with open(out_path, "w", newline='', encoding='utf-8') as out_f:
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(kept_rows)
                n_files_with_cyp += 1
            else:
                n_files_no_cyp += 1

        except Exception as e:
            print(f"  ERROR processing {fname}: {e}", file=sys.stderr)
            n_files_skipped += 1

        if i % 50 == 0:
            print(f"  ...processed {i}/{len(csv_files)} files so far", file=sys.stderr)

    print("\n" + "=" * 60)
    print(f"Files with >=1 CYP primary metabolite (written): {n_files_with_cyp}")
    print(f"Files with 0 CYP metabolites (no output):        {n_files_no_cyp}")
    print(f"Files skipped:                                   {n_files_skipped}")
    print(f"Total CYP metabolite rows kept:                  {total_cyp}")
    print(f"Total non-CYP rows excluded:                     {total_non_cyp}")
    print(f"\nPer-drug CYP-filtered files written to: {outdir}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 step2_filter_cyp.py primary_results cyp_results")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
