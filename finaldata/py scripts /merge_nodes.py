"""
Merge gnn_node_mapping.csv + node_smiles.csv + final_metabolites.csv
into a single pharmacon_final_nodes.csv (one row per drug, 645 rows).

Usage:
    python merge_nodes.py

Expects the three input files in the same directory as this script
(edit the paths below if yours live elsewhere). File names are left
unchanged per instructions.
"""

import pandas as pd

# ---- 1. File paths (edit if needed) ----------------------------------
NODE_MAPPING_FILE   = "gnn_node_mapping.csv"     # drug_id, node_index
NODE_SMILES_FILE    = "node_smiles.csv"          # drug_id/node_index, parent_smiles
METABOLITES_FILE    = "final_metabolites.csv"    # parent_drug_id/parent_node_index, metabolite cols
OUTPUT_FILE          = "pharmacon_final_nodes.csv"

# ---- 2. Load base: one row per drug -----------------------------------
base = pd.read_csv(NODE_MAPPING_FILE)
n_start = len(base)
print(f"[1] Loaded {NODE_MAPPING_FILE}: {n_start} rows")

# ---- 3. Load node_smiles and left-join on drug_id (fallback node_index) ----
smiles = pd.read_csv(NODE_SMILES_FILE)

if "drug_id" in smiles.columns and "drug_id" in base.columns:
    join_key = "drug_id"
elif "node_index" in smiles.columns and "node_index" in base.columns:
    join_key = "node_index"
else:
    raise ValueError(
        f"Cannot find a common join key between {NODE_MAPPING_FILE} "
        f"({list(base.columns)}) and {NODE_SMILES_FILE} ({list(smiles.columns)})"
    )

merged = base.merge(smiles, on=join_key, how="left", suffixes=("", "_smiles"))
print(f"[2] Left-joined {NODE_SMILES_FILE} on '{join_key}': {len(merged)} rows")

if len(merged) != n_start:
    raise AssertionError(
        f"Row count changed after joining {NODE_SMILES_FILE}: "
        f"{n_start} -> {len(merged)}. Check for duplicate keys in that file."
    )

# Standardize the SMILES column name to parent_smiles
if "parent_smiles" not in merged.columns:
    candidates = [c for c in smiles.columns if "smiles" in c.lower()]
    if len(candidates) == 1:
        merged = merged.rename(columns={candidates[0]: "parent_smiles"})
    else:
        raise ValueError(
            f"Could not identify the parent SMILES column in {NODE_SMILES_FILE}. "
            f"Columns found: {list(smiles.columns)}. Please rename it to 'parent_smiles' "
            f"or adjust the script."
        )

# ---- 4. Load final_metabolites and left-join on parent_drug_id/parent_node_index ----
metabolites = pd.read_csv(METABOLITES_FILE)

if "parent_drug_id" in metabolites.columns and "drug_id" in merged.columns:
    left_key, right_key = "drug_id", "parent_drug_id"
elif "parent_node_index" in metabolites.columns and "node_index" in merged.columns:
    left_key, right_key = "node_index", "parent_node_index"
else:
    raise ValueError(
        f"Cannot find a common join key between merged data "
        f"({list(merged.columns)}) and {METABOLITES_FILE} ({list(metabolites.columns)})"
    )

merged = merged.merge(
    metabolites,
    left_on=left_key,
    right_on=right_key,
    how="left",
    suffixes=("", "_meta"),
)
print(f"[3] Left-joined {METABOLITES_FILE} on '{left_key}'='{right_key}': {len(merged)} rows")

if len(merged) != n_start:
    raise AssertionError(
        f"Row count changed after joining {METABOLITES_FILE}: "
        f"{n_start} -> {len(merged)}. This means the metabolites file has "
        f"duplicate/multiple rows per drug (e.g. multiple candidate metabolites "
        f"per parent) -- dedupe or pick one per drug before merging, or check "
        f"for a many-to-one join issue."
    )

# ---- 5. Standardize metabolite column names (exact names, not fuzzy) ----
# final_metabolites.csv has BOTH "SMILES" (the metabolite) and
# "Precursor SMILES" (the parent) -- must match exactly, not by substring.
exact_rename_map = {
    "SMILES": "metabolite_smiles",
    "InChIKey": "metabolite_inchikey",
    "selection_method": "selection_method",  # already correct name
}
missing_exact = [c for c in ["SMILES", "InChIKey", "selection_method"] if c not in metabolites.columns]
if missing_exact:
    raise ValueError(
        f"Expected exact columns {missing_exact} not found in {METABOLITES_FILE}. "
        f"Columns found: {list(metabolites.columns)}"
    )
merged = merged.rename(columns=exact_rename_map)

for required in ["metabolite_smiles", "metabolite_inchikey", "selection_method"]:
    if required not in merged.columns:
        raise ValueError(
            f"Could not identify a '{required}' column in {METABOLITES_FILE}. "
            f"Columns found: {list(metabolites.columns)}. Please rename or adjust the script."
        )

# ---- 6. has_metabolite flag --------------------------------------------
merged["has_metabolite"] = merged["metabolite_smiles"].notna().astype(int)

n_with_metabolite = merged["has_metabolite"].sum()
n_without = len(merged) - n_with_metabolite
print(f"[4] has_metabolite: {n_with_metabolite} yes / {n_without} no")

# ---- 7. Select and order final columns ----------------------------------
final_cols = [
    "node_index",
    "drug_id",
    "parent_smiles",
    "has_metabolite",
    "metabolite_smiles",
    "metabolite_inchikey",
    "selection_method",
]
missing = [c for c in final_cols if c not in merged.columns]
if missing:
    raise ValueError(f"Missing expected final columns: {missing}. Have: {list(merged.columns)}")

final = merged[final_cols]

# ---- 8. Final sanity check + save ---------------------------------------
if len(final) != n_start:
    raise AssertionError(f"Final row count {len(final)} != expected {n_start}")

final.to_csv(OUTPUT_FILE, index=False)
print(f"[5] Saved {OUTPUT_FILE}: {len(final)} rows, columns: {list(final.columns)}")
print("Done.")
