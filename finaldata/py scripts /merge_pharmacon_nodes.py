#!/usr/bin/env python3
"""
Merge gnn_node_mapping.csv, node_smiles.csv, and final_metabolites.csv
into one combined CSV: pharmacon_final_nodes.csv

Output columns:
    node_index, drug_id, parent_smiles, has_metabolite,
    metabolite_smiles, metabolite_inchikey, selection_method

Join logic:
    1. Base: gnn_node_mapping.csv (drug_id, node_index) -> 645 rows
    2. Left-join node_smiles.csv on drug_id -> brings in parent_smiles
    3. Left-join final_metabolites.csv on parent_drug_id -> brings in
       metabolite smiles / inchikey / selection_method
    4. has_metabolite = 1 if metabolite_smiles is non-null else 0
    5. Row count must stay at 645 throughout (left joins only).
       Any drop signals a join-key mismatch and the script aborts.

Usage:
    python3 merge_pharmacon_nodes.py [--indir DIR] [--outdir DIR]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

EXPECTED_ROWS = 645

# Candidate column names to try, in order, for flexible matching against
# real-world files that may use slightly different headers.
DRUG_ID_CANDIDATES = ["drug_id"]
NODE_INDEX_CANDIDATES = ["node_index"]
PARENT_SMILES_CANDIDATES = ["parent_smiles", "smiles"]
PARENT_DRUG_ID_CANDIDATES = ["parent_drug_id", "drug_id"]
PARENT_NODE_INDEX_CANDIDATES = ["parent_node_index", "node_index"]
METABOLITE_SMILES_CANDIDATES = ["metabolite_smiles", "metabolite SMILES", "smiles"]
METABOLITE_INCHIKEY_CANDIDATES = ["metabolite_inchikey", "InChIKey", "inchikey"]
SELECTION_METHOD_CANDIDATES = ["selection_method"]


def find_col(df: pd.DataFrame, candidates, label: str, required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find a column for '{label}' in columns: {list(df.columns)}. "
            f"Tried: {candidates}"
        )
    return None


def check_row_count(df: pd.DataFrame, step: str, expected: int = EXPECTED_ROWS) -> None:
    n = len(df)
    if n < expected:
        raise ValueError(
            f"[ABORT] Row count dropped to {n} after step '{step}' "
            f"(expected {expected}). This means a join-key mismatch occurred — "
            f"check that the key columns line up (dtype, whitespace, casing) "
            f"before trusting the output."
        )
    if n > expected:
        raise ValueError(
            f"[ABORT] Row count grew to {n} after step '{step}' "
            f"(expected {expected}). This means the join key isn't unique on "
            f"one side (duplicate drug_id / parent_drug_id rows), causing a "
            f"fan-out. Check for duplicates before trusting the output."
        )
    print(f"  OK: {n} rows after '{step}'")


def main():
    parser = argparse.ArgumentParser(description="Merge pharmacon node CSVs.")
    parser.add_argument(
        "--indir",
        default="/mnt/user-data/uploads",
        help="Directory containing the three input CSVs (default: %(default)s)",
    )
    parser.add_argument(
        "--outdir",
        default="/mnt/user-data/outputs",
        help="Directory to write pharmacon_final_nodes.csv (default: %(default)s)",
    )
    args = parser.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mapping_path = indir / "gnn_node_mapping.csv"
    smiles_path = indir / "node_smiles.csv"
    metabolites_path = indir / "final_metabolites.csv"

    for p in (mapping_path, smiles_path, metabolites_path):
        if not p.exists():
            print(f"ERROR: expected input file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # ---- Step 1: base ----
    print("Step 1: loading gnn_node_mapping.csv as base ...")
    base = pd.read_csv(mapping_path)
    drug_id_col = find_col(base, DRUG_ID_CANDIDATES, "drug_id")
    node_index_col = find_col(base, NODE_INDEX_CANDIDATES, "node_index")
    base = base.rename(columns={drug_id_col: "drug_id", node_index_col: "node_index"})
    base = base[["drug_id", "node_index"]].drop_duplicates()
    check_row_count(base, "load base (gnn_node_mapping.csv)")

    # ---- Step 2: left-join node_smiles.csv ----
    print("Step 2: left-joining node_smiles.csv on drug_id ...")
    smiles_df = pd.read_csv(smiles_path)
    smiles_drug_id_col = find_col(smiles_df, DRUG_ID_CANDIDATES, "drug_id (node_smiles.csv)")
    smiles_parent_col = find_col(smiles_df, PARENT_SMILES_CANDIDATES, "parent_smiles (node_smiles.csv)")
    smiles_df = smiles_df.rename(
        columns={smiles_drug_id_col: "drug_id", smiles_parent_col: "parent_smiles"}
    )[["drug_id", "parent_smiles"]].drop_duplicates(subset=["drug_id"])

    merged = base.merge(smiles_df, on="drug_id", how="left")
    check_row_count(merged, "join node_smiles.csv")

    # ---- Step 3: left-join final_metabolites.csv ----
    print("Step 3: left-joining final_metabolites.csv on parent_drug_id ...")
    meta_df = pd.read_csv(metabolites_path)
    meta_drug_id_col = find_col(meta_df, PARENT_DRUG_ID_CANDIDATES, "parent_drug_id (final_metabolites.csv)")
    meta_smiles_col = find_col(meta_df, METABOLITE_SMILES_CANDIDATES, "metabolite_smiles (final_metabolites.csv)")
    meta_inchikey_col = find_col(meta_df, METABOLITE_INCHIKEY_CANDIDATES, "metabolite_inchikey (final_metabolites.csv)")
    meta_method_col = find_col(
        meta_df, SELECTION_METHOD_CANDIDATES, "selection_method (final_metabolites.csv)", required=False
    )

    rename_map = {
        meta_drug_id_col: "drug_id",
        meta_smiles_col: "metabolite_smiles",
        meta_inchikey_col: "metabolite_inchikey",
    }
    keep_cols = ["drug_id", "metabolite_smiles", "metabolite_inchikey"]
    if meta_method_col:
        rename_map[meta_method_col] = "selection_method"
        keep_cols.append("selection_method")

    meta_df = meta_df.rename(columns=rename_map)

    # Guard against duplicate metabolite rows per drug (would cause fan-out).
    dup_count = meta_df["drug_id"].duplicated().sum()
    if dup_count > 0:
        print(
            f"  WARNING: final_metabolites.csv has {dup_count} duplicate "
            f"parent_drug_id rows — keeping the first occurrence per drug."
        )
        meta_df = meta_df.drop_duplicates(subset=["drug_id"], keep="first")

    meta_df = meta_df[keep_cols]
    if "selection_method" not in meta_df.columns:
        meta_df["selection_method"] = pd.NA

    merged = merged.merge(meta_df, on="drug_id", how="left")
    check_row_count(merged, "join final_metabolites.csv")

    # ---- Step 4: has_metabolite ----
    print("Step 4: computing has_metabolite ...")
    merged["has_metabolite"] = merged["metabolite_smiles"].notna().astype(int)

    # ---- Step 5: save ----
    out_cols = [
        "node_index",
        "drug_id",
        "parent_smiles",
        "has_metabolite",
        "metabolite_smiles",
        "metabolite_inchikey",
        "selection_method",
    ]
    merged = merged[out_cols]
    check_row_count(merged, "final assembly")

    out_path = outdir / "pharmacon_final_nodes.csv"
    merged.to_csv(out_path, index=False)

    n_with_meta = int(merged["has_metabolite"].sum())
    n_without_meta = len(merged) - n_with_meta
    print("\nDone.")
    print(f"  Total rows: {len(merged)}")
    print(f"  With metabolite: {n_with_meta}")
    print(f"  Without metabolite: {n_without_meta} (expected 88)")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
