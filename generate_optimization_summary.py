"""
generate_optimization_summary.py
=================================
For each CEC benchmark subfolder (CEC-2014, CEC-2017, CEC-2020, CEC-2022),
reads 1_Mean_Std_Rank_Table.csv and produces:
  optimization_summary.txt  - counts how many functions GBFAO-v4 optimised
                              (ranked 1st = best mean fitness) vs total.

'Optimised' definition
  Rank 1 -> GBFAO-v4 achieved the lowest (best) mean fitness on that function.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd

BASE      = os.path.join("results")
SUITES    = ["CEC-2014", "CEC-2017", "CEC-2020", "CEC-2022"]
ALGORITHM = "GBFAO-v4"

RANK_COL     = "Rank"
FUNC_COL     = "Function"
ALGO_COL     = "Algorithm"
MEAN_COL     = "Mean"

SEP = "=" * 60


def analyse_suite(suite_path: str, suite_name: str) -> dict:
    csv_path = os.path.join(suite_path, "1_Mean_Std_Rank_Table.csv")
    df = pd.read_csv(csv_path)

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    all_funcs = df[FUNC_COL].unique().tolist()
    total_funcs = len(all_funcs)

    gbfao_df = df[df[ALGO_COL] == ALGORITHM].copy()

    # Functions where GBFAO-v4 is rank 1 (best)
    rank1_funcs = gbfao_df[gbfao_df[RANK_COL] == 1][FUNC_COL].tolist()

    # Functions where GBFAO-v4 is in top-3
    top3_funcs = gbfao_df[gbfao_df[RANK_COL] <= 3][FUNC_COL].tolist()

    # Per-function detail: rank of GBFAO-v4
    detail = (gbfao_df[[FUNC_COL, MEAN_COL, RANK_COL]]
              .sort_values(FUNC_COL)
              .reset_index(drop=True))

    return {
        "all_funcs"   : all_funcs,
        "total_funcs" : total_funcs,
        "rank1_funcs" : rank1_funcs,
        "top3_funcs"  : top3_funcs,
        "detail"      : detail,
    }


def write_summary(suite_path: str, suite_name: str, info: dict):
    out_path = os.path.join(suite_path, "optimization_summary.txt")

    rank1  = info["rank1_funcs"]
    top3   = info["top3_funcs"]
    total  = info["total_funcs"]
    detail = info["detail"]

    lines = []
    lines.append(SEP)
    lines.append(f"  GBFAO Optimisation Summary  –  {suite_name}")
    lines.append(SEP)
    lines.append("")
    lines.append(f"  Algorithm evaluated : {ALGORITHM}")
    lines.append(f"  Total functions     : {total}")
    lines.append("")
    lines.append(f"  [OPTIMISED]  Functions where GBFAO-v4 ranked #1 (best fitness)")
    lines.append(f"     Count : {len(rank1)} / {total}")
    lines.append(f"     Functions : {', '.join(rank1) if rank1 else 'None'}")
    lines.append("")
    lines.append(f"  [TOP-3]  Functions where GBFAO-v4 ranked 1st, 2nd, or 3rd")
    lines.append(f"     Count : {len(top3)} / {total}")
    lines.append(f"     Functions : {', '.join(top3) if top3 else 'None'}")
    lines.append("")
    lines.append(SEP)
    lines.append("  Per-Function Detail")
    lines.append(SEP)
    lines.append(f"  {'Function':<12} {'Mean Fitness':>20}  {'Rank':>6}  {'Status'}")
    lines.append(f"  {'-'*12}  {'-'*20}  {'-'*6}  {'-'*15}")

    for _, row in detail.iterrows():
        fname = str(row[FUNC_COL])
        mean  = row[MEAN_COL]
        rank  = int(row[RANK_COL])
        if rank == 1:
            status = "[#1] OPTIMISED (Best)"
        elif rank <= 3:
            status = f"[Top-3]"
        else:
            status = ""
        lines.append(f"  {fname:<12}  {mean:>20.6e}  {rank:>6}  {status}")

    lines.append("")
    lines.append(SEP)
    lines.append(f"  RESULT: {ALGORITHM} optimised {len(rank1)} out of {total} functions")
    lines.append(f"          (ranked #1 / best mean fitness)")
    lines.append(SEP)
    lines.append("")

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print(f"[+] Saved -> {out_path}\n")


def main():
    for suite_name in SUITES:
        suite_path = os.path.join(BASE, suite_name)
        if not os.path.isdir(suite_path):
            print(f"[!] Skipping {suite_name} – directory not found: {suite_path}")
            continue
        csv_file = os.path.join(suite_path, "1_Mean_Std_Rank_Table.csv")
        if not os.path.isfile(csv_file):
            print(f"[!] Skipping {suite_name} – CSV not found: {csv_file}")
            continue

        print(f"\nProcessing {suite_name} ...")
        info = analyse_suite(suite_path, suite_name)
        write_summary(suite_path, suite_name, info)


if __name__ == "__main__":
    main()
