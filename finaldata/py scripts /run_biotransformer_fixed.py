"""
Batch-run BioTransformer3.0 over a CSV of parent drug SMILES.

SETUP (one time):
    conda create -n biotransformer -c bioconda -c conda-forge biotransformer python=3.10 -y
    conda activate biotransformer
    biotransformer -h        # sanity check it works

USAGE:
    conda activate biotransformer
    python run_biotransformer.py --input node_smiles.csv --outdir bt_results --mode allHuman --depth 2

This calls the `biotransformer` CLI (installed by bioconda) once per drug,
writes one CSV per drug into --outdir, then concatenates everything into
a single metabolites.csv at the end. Safe to interrupt and re-run — it
skips drugs that already have an output file.
"""

import argparse
import csv
import os
import subprocess
import sys
import time

def run_one(smiles, drug_id, node_index, mode, depth, outdir, timeout):
    out_path = os.path.join(outdir, f"{node_index}_{drug_id}.csv")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path, "skipped (already done)"

    cmd = [
        "biotransformer",
        "-k", "pred",
        "-b", mode,
        "-ismi", smiles,
        "-ocsv", out_path,
        "-s", str(depth),
    ]

    env = os.environ.copy()
    # Fix for: "WARNING: A restricted method in java.lang.System has been called"
    # NOTE: must use JDK_JAVA_OPTIONS, not JAVA_TOOL_OPTIONS -- the latter
    # rejects --add-opens with "Unrecognized option" on modern JVMs.
    existing_opts = env.get("JDK_JAVA_OPTIONS", "")
    new_opts = "--add-opens java.base/java.lang=ALL-UNNAMED --enable-native-access=ALL-UNNAMED"
    env["JDK_JAVA_OPTIONS"] = (existing_opts + " " + new_opts).strip()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
        if result.returncode != 0:
            return None, f"FAILED (code {result.returncode}): {result.stderr[:300]}"
        if not os.path.exists(out_path):
            return None, f"FAILED: no output file produced. stderr: {result.stderr[:300]}"
        return out_path, "ok"
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT after {timeout}s"
    except FileNotFoundError:
        print("ERROR: `biotransformer` command not found. Did you activate the conda env?")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with node_index,drug_id,smiles")
    ap.add_argument("--outdir", default="bt_results")
    ap.add_argument("--mode", default="allHuman",
                     choices=["cyp450", "phaseII", "ecbased", "hgut", "allHuman", "superbio", "env"])
    ap.add_argument("--depth", type=int, default=2, help="max metabolism steps (-s flag)")
    ap.add_argument("--timeout", type=int, default=180, help="seconds per drug before giving up")
    ap.add_argument("--limit", type=int, default=None, help="only process first N rows (for testing)")
    ap.add_argument("--retry-failed-from-log", default=None,
                     help="path to a previous _run_log.tsv; only re-process drug_ids that FAILED or TIMED OUT in it")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "_run_log.tsv")

    with open(args.input, newline="") as f:
        rows = list(csv.DictReader(f))

    if args.retry_failed_from_log:
        failed_ids = set()
        ok_ids = set()
        with open(args.retry_failed_from_log, newline="") as logf_in:
            logreader = csv.reader(logf_in, delimiter="\t")
            for parts in logreader:
                if len(parts) < 4:
                    continue
                drug_id = parts[1].strip()
                status = parts[3]
                if status.strip() == "ok" or "skipped" in status:
                    ok_ids.add(drug_id)
                elif "FAILED" in status or "TIMEOUT" in status:
                    failed_ids.add(drug_id)
        # only truly-failed: never succeeded in ANY logged attempt
        failed_ids = failed_ids - ok_ids
        rows = [r for r in rows if r["drug_id"].strip() in failed_ids]
        print(f"Retry mode: found {len(failed_ids)} drug_ids that never succeeded in the log, "
              f"{len(rows)} matching rows in input to reprocess.")

    if args.limit:
        rows = rows[: args.limit]

    print(f"Processing {len(rows)} drugs with mode={args.mode}, depth={args.depth} ...")

    with open(log_path, "a", newline="") as logf:
        logw = csv.writer(logf, delimiter="\t")
        for i, row in enumerate(rows, 1):
            smiles = row["smiles"].strip()
            drug_id = row["drug_id"].strip()
            node_index = row["node_index"].strip()

            t0 = time.time()
            out_path, status = run_one(
                smiles, drug_id, node_index, args.mode, args.depth, args.outdir, args.timeout
            )
            elapsed = round(time.time() - t0, 1)
            print(f"[{i}/{len(rows)}] {drug_id} ({node_index}): {status} ({elapsed}s)")
            logw.writerow([node_index, drug_id, smiles, status, elapsed])
            logf.flush()

    print("\nDone with individual runs. Merging into metabolites.csv ...")
    merge(args.outdir, args.input)


def merge(outdir, input_csv):
    """Concatenate all per-drug BioTransformer CSVs into one file,
    tagging each row with the parent node_index / drug_id."""
    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    id_map = {(r["node_index"], r["drug_id"]): r for r in rows}

    merged_path = os.path.join(outdir, "metabolites_merged.csv")
    wrote_header = False
    n_written = 0

    with open(merged_path, "w", newline="") as out_f:
        writer = None
        for fname in sorted(os.listdir(outdir)):
            if not fname.endswith(".csv") or fname == "metabolites_merged.csv":
                continue
            node_index, rest = fname.split("_", 1)
            drug_id = rest.rsplit(".csv", 1)[0]
            fpath = os.path.join(outdir, fname)
            try:
                with open(fpath, newline="") as in_f:
                    reader = csv.DictReader(in_f)
                    for r in reader:
                        r["parent_node_index"] = node_index
                        r["parent_drug_id"] = drug_id
                        if writer is None:
                            fieldnames = ["parent_node_index", "parent_drug_id"] + [
                                k for k in r.keys() if k not in ("parent_node_index", "parent_drug_id")
                            ]
                            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                            writer.writeheader()
                        writer.writerow(r)
                        n_written += 1
            except Exception as e:
                print(f"  (skipping unreadable {fname}: {e})")

    print(f"Wrote {n_written} metabolite rows to {merged_path}")


if __name__ == "__main__":
    main()