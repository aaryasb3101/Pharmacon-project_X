"""
Join has_metabolite flags (from drugs_flagged.csv) onto ddis.csv pairs and
report AND / OR / NEITHER coverage overall and per interaction type.
"""

import pandas as pd

DDIS_CSV = "ddis.csv"
FLAGGED_CSV = "drugs_flagged.csv"


def parse_cid(drug_id_series):
    return (
        drug_id_series.str.replace("CID", "", regex=False)
        .str.lstrip("0")
        .replace("", "0")
        .astype(int)
    )


def main():
    ddis = pd.read_csv(DDIS_CSV, dtype={"d1": str, "d2": str})
    ddis["cid1"] = parse_cid(ddis["d1"])
    ddis["cid2"] = parse_cid(ddis["d2"])

    flagged = pd.read_csv(FLAGGED_CSV)
    flag_map = flagged.set_index("cid")["has_metabolite"].to_dict()

    ddis["flag1"] = ddis["cid1"].map(flag_map)
    ddis["flag2"] = ddis["cid2"].map(flag_map)

    # Any drug in ddis.csv not present in drugs.csv at all (shouldn't happen
    # given your setup, but check rather than assume) -> treat as unmatched.
    n_missing = ddis["flag1"].isna().sum() + ddis["flag2"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} drug references in ddis.csv have no "
              f"entry in drugs_flagged.csv at all (not just unmatched-in-HMDB "
              f"-- literally absent from your drugs.csv). Check ID parsing.")
    ddis["flag1"] = ddis["flag1"].fillna(0).astype(int)
    ddis["flag2"] = ddis["flag2"].fillna(0).astype(int)

    ddis["both"] = (ddis["flag1"] == 1) & (ddis["flag2"] == 1)
    ddis["at_least_one"] = (ddis["flag1"] == 1) | (ddis["flag2"] == 1)
    ddis["neither"] = (ddis["flag1"] == 0) & (ddis["flag2"] == 0)

    # De-duplicate to unique (d1, d2) pairs before the overall pair-level
    # counts -- ddis.csv has one row per (d1, d2, type), so the same pair
    # repeats across types.
    unique_pairs = ddis.drop_duplicates(subset=["cid1", "cid2"])

    total_pairs = len(unique_pairs)
    n_both = unique_pairs["both"].sum()
    n_atleast1 = unique_pairs["at_least_one"].sum()
    n_neither = unique_pairs["neither"].sum()

    print("=== Overall pair-level coverage (unique d1,d2 pairs) ===")
    print(f"Total unique pairs:        {total_pairs}")
    print(f"Both have metabolite:      {n_both} ({100*n_both/total_pairs:.1f}%)")
    print(f"At least one (OR/masked):  {n_atleast1} ({100*n_atleast1/total_pairs:.1f}%)")
    print(f"Neither:                   {n_neither} ({100*n_neither/total_pairs:.1f}%)")

    # Per-type breakdown. Note: rows here are (d1,d2,type) rows, not deduped,
    # since coverage-by-type is inherently about the type-labeled rows.
    per_type = (
        ddis.groupby("type")
        .agg(
            n_pairs=("type", "size"),
            n_both=("both", "sum"),
            n_atleast1=("at_least_one", "sum"),
            n_neither=("neither", "sum"),
        )
        .reset_index()
    )
    per_type["pct_both"] = 100 * per_type["n_both"] / per_type["n_pairs"]
    per_type["pct_atleast1"] = 100 * per_type["n_atleast1"] / per_type["n_pairs"]
    per_type["pct_neither"] = 100 * per_type["n_neither"] / per_type["n_pairs"]

    per_type.to_csv("ddi_coverage_by_type.csv", index=False)
    print("\nWrote ddi_coverage_by_type.csv "
          f"({len(per_type)} interaction types)")

    # Flag types most/least skewed toward coverage, for a quick sanity look
    print("\nTop 5 types by %-both-covered:")
    print(per_type.sort_values("pct_both", ascending=False).head(5)
          [["type", "n_pairs", "pct_both"]])
    print("\nBottom 5 types by %-both-covered:")
    print(per_type.sort_values("pct_both").head(5)
          [["type", "n_pairs", "pct_both"]])


if __name__ == "__main__":
    main()
