"""
=============================================================
 Statistical Analysis – Wilcoxon Rank-Sum Test
 ─────────────────────────────────────────────
 Pairwise comparison of GBFAO v4 vs each competitor algorithm
 across all benchmark functions.

 Output
 ──────
 • Wilcoxon p-values table  (algorithms × functions)
 • Summary win / tie / loss counts
 • CSV export: wilcoxon_results.csv
=============================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

# ─────────────────────────────────────────────────────────────
# Core Wilcoxon wrapper
# ─────────────────────────────────────────────────────────────

def wilcoxon_test(results_a: np.ndarray,
                  results_b: np.ndarray,
                  alpha: float = 0.05) -> dict:
    """
    Perform Wilcoxon rank-sum test between two result vectors.

    Parameters
    ----------
    results_a : 1-D array of fitness values for algorithm A (proposed)
    results_b : 1-D array of fitness values for algorithm B (competitor)
    alpha     : significance level (default 0.05)

    Returns
    -------
    dict with keys:
        p_value    : float
        statistic  : float
        significant: bool      True if p_value < alpha
        result     : str       '+' better, '−' worse, '≈' tie
    """
    try:
        stat, p = stats.mannwhitneyu(results_a, results_b,
                                     alternative='two-sided')
    except ValueError:
        # All values equal
        return {"p_value": 1.0, "statistic": 0.0,
                "significant": False, "result": "≈"}

    significant = p < alpha
    if significant:
        if np.median(results_a) < np.median(results_b):
            outcome = "+"    # proposed is BETTER
        else:
            outcome = "−"    # proposed is WORSE
    else:
        outcome = "≈"        # statistically tied

    return {"p_value": float(p), "statistic": float(stat),
            "significant": significant, "result": outcome}


# ─────────────────────────────────────────────────────────────
# Full comparison table
# ─────────────────────────────────────────────────────────────

def full_wilcoxon_table(proposed_results: dict,
                        competitor_results: dict,
                        func_names: list,
                        alpha: float = 0.05,
                        save_path: str = "results/wilcoxon_results.csv"
                        ) -> pd.DataFrame:
    """
    Build a full pairwise Wilcoxon table.

    Parameters
    ----------
    proposed_results   : {func_name: np.ndarray}  50 runs of GBFAO v4
    competitor_results : {algo_name: {func_name: np.ndarray}}
    func_names         : list of function name strings
    alpha              : significance level
    save_path          : output CSV path

    Returns
    -------
    DataFrame  (index = algorithm names, columns = func names,
                values = '+' / '−' / '≈')
    Also prints win/tie/loss summary.
    """
    algo_names  = list(competitor_results.keys())
    table_sign  = pd.DataFrame(index=algo_names, columns=func_names, dtype=str)
    table_pvals = pd.DataFrame(index=algo_names, columns=func_names, dtype=float)

    for algo in algo_names:
        wins = ties = losses = 0
        for fn in func_names:
            if fn not in proposed_results or fn not in competitor_results[algo]:
                table_sign.loc[algo, fn]  = "N/A"
                table_pvals.loc[algo, fn] = np.nan
                continue

            r = wilcoxon_test(proposed_results[fn],
                              competitor_results[algo][fn],
                              alpha)
            table_sign.loc[algo, fn]  = r["result"]
            table_pvals.loc[algo, fn] = r["p_value"]

            if   r["result"] == "+": wins   += 1
            elif r["result"] == "−": losses += 1
            else:                    ties   += 1

        print(f"  {algo:6s}: W={wins:2d}  T={ties:2d}  L={losses:2d}  "
              f"(vs GBFAO-v4, α={alpha})")

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    table_sign.to_csv(save_path)
    table_pvals.to_csv(save_path.replace(".csv", "_pvalues.csv"))
    print(f"\n[✓] Wilcoxon tables saved → {save_path}")

    return table_sign, table_pvals


# ─────────────────────────────────────────────────────────────
# Summary formatter
# ─────────────────────────────────────────────────────────────

def print_wilcoxon_summary(sign_table: pd.DataFrame) -> None:
    """Print +/−/≈ counts per algorithm."""
    print("\n" + "═"*55)
    print("  Wilcoxon Summary  (GBFAO-v4 vs each competitor)")
    print("  Legend: + = GBFAO-v4 significantly better")
    print("          − = GBFAO-v4 significantly worse")
    print("          ≈ = no significant difference")
    print("═"*55)
    for algo in sign_table.index:
        row  = sign_table.loc[algo]
        wins = (row == "+").sum()
        ties = (row == "≈").sum()
        loss = (row == "−").sum()
        print(f"  {algo:8s}  +{wins:2d}  ≈{ties:2d}  −{loss:2d}")
    print("═"*55 + "\n")


# ─────────────────────────────────────────────────────────────
# Quick self-test  (synthetic data)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    funcs = [f"F{i+1}" for i in range(5)]
    algos = ["PSO", "GWO", "WOA"]

    proposed = {f: rng.normal(100, 10, 50) for f in funcs}
    comps    = {a: {f: rng.normal(110, 15, 50) for f in funcs} for a in algos}

    sign_tbl, p_tbl = full_wilcoxon_table(proposed, comps, funcs,
                                          save_path="results/wilcoxon_results.csv")
    print_wilcoxon_summary(sign_tbl)
    print(sign_tbl.to_string())
