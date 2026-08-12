"""
STEP 3 of 4: For each drug's CYP-filtered primary-metabolite file (from step 2,
which may still contain multiple metabolites per drug), select exactly ONE
final representative metabolite, using this rule:

  1. Group rows by metabolite InChIKey (same metabolite structure predicted
     via multiple enzymes/reactions counts as stronger evidence).
  2. Pick the metabolite InChIKey with the HIGHEST row count (i.e. predicted
     via the most independent enzyme/reaction rules) -- this is the
     "consensus" pick.
  3. If there's a tie, break it by preferring a row whose Enzyme(s) field
     mentions CYP3A4 specifically (the single most significant human drug
     metabolizing enzyme) over other CYPs.
  4. If still tied, just take the first row (arbitrary but deterministic).

Writes one output file per drug (single row each) into --outdir, keeping the
full original row (not just SMILES) so downstream steps still have Enzyme,
Reaction, etc. available if needed.

Usage:
    python3 step3_select_single.py cyp_results single_metabolite_results
"""

import sys
import csv
import os
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

METABOLITE_INCHIKEY_COL = "InChIKey"
METABOLITE_SMILES_COL = "SMILES"
PRECURSOR_SMILES_COL = "Precursor SMILES"
ENZYME_COL = "Enzyme(s)"
PREFERRED_ENZYME = "CYP3A4"

_fpgen = GetMorganGenerator(radius=2, fpSize=2048)


def tanimoto(smiles_a, smiles_b):
    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return -1
        fp_a = _fpgen.GetFingerprint(mol_a)
        fp_b = _fpgen.GetFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except Exception:
        return -1


def select_one(rows):
    """Returns (chosen_row, selection_method)"""
    groups = defaultdict(list)
    for row in rows:
        ikey = row.get(METABOLITE_INCHIKEY_COL) or "UNKNOWN"
        groups[ikey].append(row)

    if len(groups) == 1:
        return rows[0], "single_candidate"

    sorted_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    top_count = len(sorted_groups[0][1])
    top_groups = [g for g in sorted_groups if len(g[1]) == top_count]

    if len(top_groups) == 1:
        return top_groups[0][1][0], "consensus_count"

    # tie-break 1: prefer CYP3A4
    for ikey, group_rows in top_groups:
        for row in group_rows:
            if PREFERRED_ENZYME in (row.get(ENZYME_COL) or ""):
                return row, "cyp3a4_tiebreak"

    # tie-break 2: prefer the metabolite with the highest Tanimoto similarity
    # to its precursor -- i.e. the most "typical" small-modification metabolite,
    # rather than an arbitrary pick
    best_row = None
    best_sim = -2
    for ikey, group_rows in top_groups:
        row = group_rows[0]
        met_smiles = row.get(METABOLITE_SMILES_COL)
        precursor_smiles = row.get(PRECURSOR_SMILES_COL)
        if met_smiles and precursor_smiles:
            sim = tanimoto(precursor_smiles, met_smiles)
            if sim > best_sim:
                best_sim = sim
                best_row = row

    if best_row is not None and best_sim >= 0:
        return best_row, "similarity_tiebreak"

    # last-resort fallback -- should now rarely/never trigger
    return top_groups[0][1][0], "arbitrary_fallback"


def main(cyp_results_folder, outdir):
    os.makedirs(outdir, exist_ok=True)

    csv_files = sorted(
        f for f in os.listdir(cyp_results_folder)
        if f.lower().endswith(".csv")
    )
    print(f"Found {len(csv_files)} per-drug CYP-filtered files", file=sys.stderr)

    n_written = 0
    n_skipped = 0
    method_counts = defaultdict(int)

    for i, fname in enumerate(csv_files, 1):
        fpath = os.path.join(cyp_results_folder, fname)
        try:
            with open(fpath, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)

            if not rows:
                n_skipped += 1
                continue

            chosen_row, method = select_one(rows)
            method_counts[method] += 1
            chosen_row = dict(chosen_row)
            chosen_row["selection_method"] = method

            out_fieldnames = list(fieldnames) + ["selection_method"]
            out_path = os.path.join(outdir, fname)
            with open(out_path, "w", newline='', encoding='utf-8') as out_f:
                writer = csv.DictWriter(out_f, fieldnames=out_fieldnames)
                writer.writeheader()
                writer.writerow(chosen_row)
            n_written += 1

        except Exception as e:
            print(f"  ERROR processing {fname}: {e}", file=sys.stderr)
            n_skipped += 1

        if i % 50 == 0:
            print(f"  ...processed {i}/{len(csv_files)} files so far", file=sys.stderr)

    print("\n" + "=" * 60)
    print(f"Drugs with a single metabolite chosen (written): {n_written}")
    print(f"Files skipped (empty/error):                     {n_skipped}")
    print(f"\nBreakdown by selection method:")
    for method, count in sorted(method_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {method:25s} {count}")
    print(f"\nSingle-metabolite-per-drug files written to: {outdir}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 step3_select_single.py cyp_results single_metabolite_results")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])