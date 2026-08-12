"""
STEP 1 of 4: Filters each BioTransformer output CSV (one per drug) down to
PRIMARY metabolites only (precursor InChIKey matches the known parent drug),
writing one filtered file per drug into --outdir (same filenames preserved).

Usage:
    python3 step1_filter_primary.py drugs.csv bt_results primary_results
"""

import sys
import csv
import os
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

PRECURSOR_INCHIKEY_COL = "Precursor InChIKey"
METABOLITE_SMILES_COL = "SMILES"


def smiles_to_inchikey(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def load_parent_inchikeys(drugs_csv_path):
    print("Loading known parent drug SMILES and computing InChIKeys...", file=sys.stderr)
    parent_inchikeys = set()
    with open(drugs_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get("smiles")
            if not smiles:
                continue
            ikey = smiles_to_inchikey(smiles)
            if ikey:
                parent_inchikeys.add(ikey)
    print(f"  {len(parent_inchikeys)} unique parent drug InChIKeys loaded", file=sys.stderr)
    return parent_inchikeys


def main(drugs_csv_path, bt_results_folder, outdir):
    os.makedirs(outdir, exist_ok=True)
    parent_inchikeys = load_parent_inchikeys(drugs_csv_path)

    csv_files = sorted(
        f for f in os.listdir(bt_results_folder)
        if f.lower().endswith(".csv") and f != "metabolites_merged.csv"
    )
    print(f"Found {len(csv_files)} per-drug CSV files in {bt_results_folder}", file=sys.stderr)

    total_primary = 0
    total_excluded = 0
    n_files_with_primary = 0
    n_files_empty_after_filter = 0
    n_files_skipped = 0

    for i, fname in enumerate(csv_files, 1):
        fpath = os.path.join(bt_results_folder, fname)
        try:
            with open(fpath, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                if not fieldnames or PRECURSOR_INCHIKEY_COL not in fieldnames or METABOLITE_SMILES_COL not in fieldnames:
                    n_files_skipped += 1
                    continue

                kept_rows = []
                for row in reader:
                    precursor_ikey = row.get(PRECURSOR_INCHIKEY_COL)
                    metabolite_smiles = row.get(METABOLITE_SMILES_COL)
                    if not precursor_ikey or not metabolite_smiles:
                        total_excluded += 1
                        continue
                    if precursor_ikey in parent_inchikeys:
                        kept_rows.append(row)
                        total_primary += 1
                    else:
                        total_excluded += 1

            if kept_rows:
                out_path = os.path.join(outdir, fname)
                with open(out_path, "w", newline='', encoding='utf-8') as out_f:
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(kept_rows)
                n_files_with_primary += 1
            else:
                n_files_empty_after_filter += 1

        except Exception as e:
            print(f"  ERROR processing {fname}: {e}", file=sys.stderr)
            n_files_skipped += 1

        if i % 50 == 0:
            print(f"  ...processed {i}/{len(csv_files)} files so far", file=sys.stderr)

    print("\n" + "=" * 60)
    print(f"Files with >=1 primary metabolite (written):  {n_files_with_primary}")
    print(f"Files with 0 primary metabolites (no output): {n_files_empty_after_filter}")
    print(f"Files skipped (bad/missing columns):          {n_files_skipped}")
    print(f"Total primary metabolite rows kept:           {total_primary}")
    print(f"Total rows excluded:                          {total_excluded}")
    print(f"\nPer-drug filtered files written to: {outdir}/")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 step1_filter_primary.py drugs.csv bt_results primary_results")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
