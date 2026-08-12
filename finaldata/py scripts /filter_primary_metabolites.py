"""
Filters BioTransformer output CSV(s) to keep only PRIMARY metabolites --
defined as rows whose precursor SMILES matches a known parent drug from
drugs.csv (via InChIKey comparison, not raw string matching, since SMILES
notation can differ for the same molecule).

Rows whose precursor does NOT match a known parent drug are secondary/tertiary
metabolites (derived from another metabolite, not the original drug) and are
excluded.

IMPORTANT: adjust PRECURSOR_SMILES_COL and METABOLITE_SMILES_COL below once
you've checked your actual BioTransformer output file's real column names.

Usage:
    python3 filter_primary_metabolites.py /path/to/drugs.csv /path/to/biotransformer_output.csv /path/to/primary_metabolites_output.csv
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


def sniff_column(fieldnames, candidates):
    for c in candidates:
        if c in fieldnames:
            return c
    lower_fields = {f.lower(): f for f in fieldnames}
    for c in candidates:
        for lf, original in lower_fields.items():
            if c.lower() in lf:
                return original
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


def filter_primary_metabolites(biotransformer_csv_path, parent_inchikeys, writer_state):
    with open(biotransformer_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if PRECURSOR_INCHIKEY_COL not in fieldnames or METABOLITE_SMILES_COL not in fieldnames:
            print(f"  WARNING: {biotransformer_csv_path} missing expected columns, skipping.", file=sys.stderr)
            return 0, 0, 0

        if writer_state["writer"] is None:
            writer_state["writer"] = csv.DictWriter(writer_state["file"], fieldnames=fieldnames)
            writer_state["writer"].writeheader()

        n_primary = 0
        n_secondary_or_deeper = 0
        n_invalid = 0

        for row in reader:
            precursor_ikey = row.get(PRECURSOR_INCHIKEY_COL)
            metabolite_smiles = row.get(METABOLITE_SMILES_COL)

            if not precursor_ikey or not metabolite_smiles:
                n_invalid += 1
                continue

            if precursor_ikey in parent_inchikeys:
                n_primary += 1
                writer_state["writer"].writerow(row)
            else:
                n_secondary_or_deeper += 1

        return n_primary, n_secondary_or_deeper, n_invalid


def main(drugs_csv_path, biotransformer_folder, output_path):
    parent_inchikeys = load_parent_inchikeys(drugs_csv_path)

    csv_files = sorted(
        f for f in os.listdir(biotransformer_folder)
        if f.lower().endswith(".csv")
    )
    print(f"Found {len(csv_files)} CSV files in {biotransformer_folder}", file=sys.stderr)

    total_primary = 0
    total_secondary = 0
    total_invalid = 0
    n_files_processed = 0
    n_files_skipped = 0

    with open(output_path, "w", newline='', encoding='utf-8') as out_f:
        writer_state = {"file": out_f, "writer": None}

        for i, fname in enumerate(csv_files, 1):
            fpath = os.path.join(biotransformer_folder, fname)
            try:
                n_p, n_s, n_i = filter_primary_metabolites(fpath, parent_inchikeys, writer_state)
                if writer_state["writer"] is None:
                    n_files_skipped += 1
                else:
                    total_primary += n_p
                    total_secondary += n_s
                    total_invalid += n_i
                    n_files_processed += 1
            except Exception as e:
                print(f"  ERROR processing {fname}: {e}", file=sys.stderr)
                n_files_skipped += 1

            if i % 50 == 0:
                print(f"  ...processed {i}/{len(csv_files)} files so far", file=sys.stderr)

    print("\n" + "=" * 60)
    print(f"Files processed successfully:          {n_files_processed}")
    print(f"Files skipped (missing columns/error): {n_files_skipped}")
    print(f"Total primary metabolites kept:        {total_primary}")
    print(f"Total secondary/deeper excluded:       {total_secondary}")
    print(f"Total invalid/missing rows:            {total_invalid}")
    print(f"\nCombined primary metabolites written to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 filter_primary_metabolites.py /path/to/drugs.csv /path/to/bt_results_folder /path/to/primary_metabolites_output.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])