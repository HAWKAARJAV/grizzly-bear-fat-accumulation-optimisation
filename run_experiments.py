"""
=============================================================
 run_experiments.py  –  GBFAO Full Experimental Suite
 ─────────────────────────────────────────────────────────
 HOW TO RUN
 ──────────
 Full assignment run  (30–90 min):
   python run_experiments.py

 Quick test  (2–3 min, verifies pipeline):
   python run_experiments.py --quick

 Custom subset:
   python run_experiments.py --funcs 1,2,3,4,5,6,7,8,9,10

 SETTINGS
   MAX_FES = 60,000 | RUNS = 50 | DIM = 30
   Algorithms: GBFAO-v4/v3/v2/v1 + ARO + PSO+GWO+WOA+DE+SMA+HHO+AO
=============================================================
"""

import argparse, os, time
import numpy as np
import pandas as pd

from benchmark_functions import FUNCTIONS, FUNC_NAMES, BOUNDS
from benchmark_cec2020_2022 import (FUNCTIONS_2020, FUNC_NAMES_2020, BOUNDS_2020,
                                     FUNCTIONS_2022, FUNC_NAMES_2022, BOUNDS_2022)
from competitors          import COMPETITORS
from gbfao_v1             import GBFAO_v1
from gbfao_v2             import GBFAO_v2
from gbfao_v3             import GBFAO_v3
from gbfao_v4             import GBFAO_v4
from engineering_problems import ENGINEERING_PROBLEMS, run_engineering
from statistical_test     import full_wilcoxon_table, print_wilcoxon_summary
from plot_convergence     import (plot_convergence, plot_boxplots,
                                  plot_rank_heatmap, plot_engineering,
                                  plot_version_comparison)

os.makedirs("results", exist_ok=True)

ALL_ALGORITHMS = {
    "GBFAO-v4" : GBFAO_v4,
    "GBFAO-v3" : GBFAO_v3,
    "GBFAO-v2" : GBFAO_v2,
    "GBFAO-v1" : GBFAO_v1,
    **COMPETITORS,          # ARO, PSO, GWO, WOA, DE, SMA, HHO, AO
}

WILCOXON_AGAINST = [a for a in ALL_ALGORITHMS
                    if a not in ("GBFAO-v4","GBFAO-v3","GBFAO-v2","GBFAO-v1")]

def single_run(algo_func, func, lb, ub, dim, max_fes):
    best_fit, _, curve = algo_func(func, lb, ub, dim, max_fes=max_fes)
    return float(best_fit), curve

def run_benchmark(max_fes=60_000, runs=50, dim=30, func_ids=None, verbose=True):
    lb, ub = BOUNDS
    if func_ids is None:
        func_ids = list(range(1, len(FUNCTIONS)+1))
    func_subset = [(FUNC_NAMES[i-1], FUNCTIONS[i-1]) for i in func_ids]
    algo_names  = list(ALL_ALGORITHMS.keys())
    results     = {a: {} for a in algo_names}
    med_curves  = {a: {} for a in algo_names}
    total = len(algo_names)*len(func_subset)*runs
    done  = 0; t0 = time.time()

    for fname, func in func_subset:
        for aname, afunc in ALL_ALGORITHMS.items():
            fits, curves = [], []
            for _ in range(runs):
                f, c = single_run(afunc, func, lb, ub, dim, max_fes)
                fits.append(f); curves.append(c)
                done += 1
            arr = np.array(fits)
            results[aname][fname] = arr
            ml  = min(len(c) for c in curves)
            med_curves[aname][fname] = list(np.median(
                [c[:ml] for c in curves], axis=0))
            if verbose:
                ela = time.time()-t0
                eta = ela/done*(total-done) if done<total else 0
                print(f"  {aname:12s}|{fname:4s}| "
                      f"mean={np.mean(arr):.4e} std={np.std(arr):.4e} "
                      f"[{done}/{total} ETA {eta/60:.1f}min]")
    return results, med_curves

def build_mean_std_rank_table(results, func_names,
                               save_path="results/results_table.csv"):
    algo_names = list(results.keys())
    rows = []
    for fname in func_names:
        means = {a: float(np.mean(results[a].get(fname,[np.nan]))) for a in algo_names}
        stds  = {a: float(np.std(results[a].get(fname,[np.nan])))  for a in algo_names}
        order = sorted(means, key=lambda a: means[a])
        ranks = {a: order.index(a)+1 for a in algo_names}
        for a in algo_names:
            rows.append({"Function":fname,"Algorithm":a,
                         "Mean":means[a],"Std":stds[a],"Rank":ranks[a]})
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"\n[+] Results table -> {save_path}")
    pivot = df.pivot_table(index="Algorithm",columns="Function",
                           values="Rank",aggfunc="mean")
    pivot["Avg Rank"] = pivot.mean(axis=1)
    print(pivot[["Avg Rank"]].sort_values("Avg Rank").round(2).to_string())
    return df

def run_all_engineering(max_fes=60_000, runs=30, verbose=True):
    algo_names = list(ALL_ALGORITHMS.keys())
    eng_means={a:{} for a in algo_names}
    eng_stds ={a:{} for a in algo_names}
    eng_full ={a:{} for a in algo_names}
    for prob in ENGINEERING_PROBLEMS:
        pname = prob["name"]
        if verbose: print(f"\n  [{pname}]")
        for aname, afunc in ALL_ALGORITHMS.items():
            r = run_engineering(afunc, prob, max_fes=max_fes, runs=runs)
            eng_means[aname][pname] = r["mean"]
            eng_stds[aname][pname]  = r["std"]
            eng_full[aname][pname]  = r["results"]
            if verbose:
                print(f"    {aname:12s}: mean={r['mean']:.4e}  "
                      f"std={r['std']:.4e}  best={r['best']:.4e}")
    rows=[]
    for prob in ENGINEERING_PROBLEMS:
        pname=prob["name"]
        for a in algo_names:
            rows.append({"Problem":pname,"Algorithm":a,
                         "Mean":eng_means[a][pname],"Std":eng_stds[a][pname]})
    pd.DataFrame(rows).to_csv("results/engineering_results.csv",index=False)
    print("\n[+] Engineering results -> results/engineering_results.csv")
    return eng_means, eng_full

def run_version_comparison(func_name="F7", dim=30):
    idx  = FUNC_NAMES.index(func_name)
    func = FUNCTIONS[idx]
    lb, ub = BOUNDS
    vers = [("GBFAO-v1",GBFAO_v1,60_000),("GBFAO-v2",GBFAO_v2,45_000),
            ("GBFAO-v3",GBFAO_v3,30_000),("GBFAO-v4",GBFAO_v4,20_000)]
    vc={}
    for label,fn,fes in vers:
        _,_,c = fn(func,lb,ub,dim,max_fes=fes)
        vc[label]=c
        print(f"  {label}: {fes:,} FEs -> best={c[-1]:.4e}")
    return vc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",   action="store_true")
    parser.add_argument("--max_fes", type=int, default=60_000)
    parser.add_argument("--runs",    type=int, default=50)
    parser.add_argument("--dim",     type=int, default=30)
    parser.add_argument("--funcs",   type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        print("="*55+"\n  QUICK TEST (5 funcs, 5 runs, 6000 FEs, dim=10)\n"+"="*55)
        max_fes=6_000; runs=5; dim=10; func_ids=[1,6,7,9,14]
    else:
        max_fes=args.max_fes; runs=args.runs; dim=args.dim
        func_ids=([int(x) for x in args.funcs.split(",")]
                  if args.funcs else None)

    func_names_used=([FUNC_NAMES[i-1] for i in func_ids]
                     if func_ids else FUNC_NAMES)

    print(f"\n  Algorithms : {list(ALL_ALGORITHMS.keys())}")
    print(f"  Functions  : {len(func_names_used)}")
    print(f"  Runs       : {runs}  |  Max FEs : {max_fes:,}  |  Dim : {dim}\n")

    print("[1/5] Benchmark functions ...")
    results, med_curves = run_benchmark(max_fes=max_fes,runs=runs,
                                        dim=dim,func_ids=func_ids)

    print("\n[2/5] Result tables ...")
    build_mean_std_rank_table(results, func_names_used)

    print("\n[3/5] Wilcoxon test (GBFAO-v4 vs ARO + 7 competitors) ...")
    proposed_res   = results["GBFAO-v4"]
    competitor_res = {a: results[a] for a in WILCOXON_AGAINST}
    sign_tbl, _    = full_wilcoxon_table(proposed_res, competitor_res,
                                         func_names_used)
    print_wilcoxon_summary(sign_tbl)

    print("\n[4/5] Plots ...")
    plot_convergence(med_curves, func_names_used)
    mean_dict={a:{f:float(np.mean(results[a].get(f,[np.nan])))
                  for f in func_names_used} for a in results}
    plot_rank_heatmap(mean_dict, func_names_used)
    plot_boxplots(results, func_names_used)
    print("\n  Version comparison ...")
    vc = run_version_comparison(dim=dim)
    plot_version_comparison(vc)

    print("\n[5/5] Engineering problems ...")
    eng_runs = max(10, runs//5)
    eng_means,_ = run_all_engineering(max_fes=max_fes, runs=eng_runs)
    plot_engineering(eng_means, [p["name"] for p in ENGINEERING_PROBLEMS])

    print("\n"+"="*60)
    print("  ALL DONE.  Results in ./results/")
    print("  Paste these CSVs to Claude to generate the final report:")
    print("    results/results_table.csv")
    print("    results/wilcoxon_results.csv")
    print("    results/engineering_results.csv")
    print("="*60)

# (entry point moved to bottom, after CEC-2020/2022 patch)


# ─────────────────────────────────────────────────────────────
# CEC-2020 and CEC-2022 experiment runner
# ─────────────────────────────────────────────────────────────

def run_cec_suite(suite_name, functions, func_names, bounds,
                  max_fes=60_000, runs=50, dim=10, verbose=True):
    """
    Run all algorithms on a CEC-2020 or CEC-2022 function suite.

    Parameters
    ----------
    suite_name : str        e.g. "CEC-2020"
    functions  : list       callable list
    func_names : list       name strings
    bounds     : tuple      (lb, ub)
    dim        : int        recommended 10 for CEC-2020, 10 or 20 for CEC-2022

    Returns
    -------
    results    : {algo: {func_name: np.ndarray(runs,)}}
    """
    lb, ub = bounds
    algo_names = list(ALL_ALGORITHMS.keys())
    results = {a: {} for a in algo_names}
    total = len(algo_names) * len(func_names) * runs
    done = 0; t0 = time.time()

    print(f"\n  Suite: {suite_name}  |  {len(func_names)} functions  |  dim={dim}  |  {runs} runs")

    for fname, func in zip(func_names, functions):
        for aname, afunc in ALL_ALGORITHMS.items():
            fits = []
            for _ in range(runs):
                f, _ = single_run(afunc, func, lb, ub, dim, max_fes)
                fits.append(f)
                done += 1
            arr = np.array(fits)
            results[aname][fname] = arr
            if verbose:
                ela = time.time() - t0
                eta = ela / done * (total - done) if done < total else 0
                print(f"  {aname:12s}|{fname:4s}| mean={np.mean(arr):.4e} "
                      f"std={np.std(arr):.4e} [{done}/{total} ETA {eta/60:.1f}min]")

    return results


def build_cec_table(results, func_names, suite_name,
                    save_path=None):
    """Build and save Mean/Std/Rank table for a CEC suite."""
    algo_names = list(results.keys())
    rows = []
    for fname in func_names:
        means = {a: float(np.mean(results[a].get(fname, [np.nan]))) for a in algo_names}
        stds  = {a: float(np.std( results[a].get(fname, [np.nan]))) for a in algo_names}
        order = sorted(means, key=lambda a: means[a])
        ranks = {a: order.index(a) + 1 for a in algo_names}
        for a in algo_names:
            rows.append({"Suite": suite_name, "Function": fname, "Algorithm": a,
                         "Mean": means[a], "Std": stds[a], "Rank": ranks[a]})
    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[+] {suite_name} table -> {save_path}")
    return df


def run_cec_2020_2022(max_fes=60_000, runs=50):
    """Run full CEC-2020 and CEC-2022 experiment."""
    print("\n" + "="*60)
    print("  CEC-2020 Experiments  (dim=10, 10 functions)")
    print("="*60)
    res2020 = run_cec_suite("CEC-2020", FUNCTIONS_2020, FUNC_NAMES_2020,
                             BOUNDS_2020, max_fes=max_fes, runs=runs, dim=10)
    df2020  = build_cec_table(res2020, FUNC_NAMES_2020, "CEC-2020",
                               "results/cec2020_results.csv")

    print("\n" + "="*60)
    print("  CEC-2022 Experiments  (dim=10, 12 functions)")
    print("="*60)
    res2022 = run_cec_suite("CEC-2022", FUNCTIONS_2022, FUNC_NAMES_2022,
                             BOUNDS_2022, max_fes=max_fes, runs=runs, dim=10)
    df2022  = build_cec_table(res2022, FUNC_NAMES_2022, "CEC-2022",
                               "results/cec2022_results.csv")

    # Wilcoxon for each suite
    print("\n[Wilcoxon] CEC-2020 ...")
    proposed2020   = res2020["GBFAO-v4"]
    competitor2020 = {a: res2020[a] for a in WILCOXON_AGAINST}
    sign2020, _    = full_wilcoxon_table(proposed2020, competitor2020,
                                          FUNC_NAMES_2020,
                                          save_path="results/cec2020_wilcoxon.csv")
    print_wilcoxon_summary(sign2020)

    print("\n[Wilcoxon] CEC-2022 ...")
    proposed2022   = res2022["GBFAO-v4"]
    competitor2022 = {a: res2022[a] for a in WILCOXON_AGAINST}
    sign2022, _    = full_wilcoxon_table(proposed2022, competitor2022,
                                          FUNC_NAMES_2022,
                                          save_path="results/cec2022_wilcoxon.csv")
    print_wilcoxon_summary(sign2022)

    return res2020, res2022, df2020, df2022


# Patch main() to include CEC-2020/2022 as optional step
_original_main = main

def run_cec_2022_only(max_fes=60_000, runs=50):
    """Run only the CEC-2022 suite (for parallel execution on a separate machine)."""
    print("\n" + "="*60)
    print("  CEC-2022 Experiments  (dim=10, 12 functions)")
    print("="*60)
    res2022 = run_cec_suite("CEC-2022", FUNCTIONS_2022, FUNC_NAMES_2022,
                             BOUNDS_2022, max_fes=max_fes, runs=runs, dim=10)
    df2022  = build_cec_table(res2022, FUNC_NAMES_2022, "CEC-2022",
                               "results/cec2022_results.csv")

    print("\n[Wilcoxon] CEC-2022 ...")
    proposed2022   = res2022["GBFAO-v4"]
    competitor2022 = {a: res2022[a] for a in WILCOXON_AGAINST}
    sign2022, _    = full_wilcoxon_table(proposed2022, competitor2022,
                                          FUNC_NAMES_2022,
                                          save_path="results/cec2022_wilcoxon.csv")
    print_wilcoxon_summary(sign2022)
    print("\n[DONE] CEC-2022 complete. Send results/cec2022_results.csv and results/cec2022_wilcoxon.csv")
    return res2022, df2022


def run_cec_2020_only(max_fes=60_000, runs=50):
    """Run only the CEC-2020 suite."""
    print("\n" + "="*60)
    print("  CEC-2020 Experiments  (dim=10, 10 functions)")
    print("="*60)
    res2020 = run_cec_suite("CEC-2020", FUNCTIONS_2020, FUNC_NAMES_2020,
                             BOUNDS_2020, max_fes=max_fes, runs=runs, dim=10)
    df2020  = build_cec_table(res2020, FUNC_NAMES_2020, "CEC-2020",
                               "results/cec2020_results.csv")

    print("\n[Wilcoxon] CEC-2020 ...")
    proposed2020   = res2020["GBFAO-v4"]
    competitor2020 = {a: res2020[a] for a in WILCOXON_AGAINST}
    sign2020, _    = full_wilcoxon_table(proposed2020, competitor2020,
                                          FUNC_NAMES_2020,
                                          save_path="results/cec2020_wilcoxon.csv")
    print_wilcoxon_summary(sign2020)
    print("\n[DONE] CEC-2020 complete. Send results/cec2020_results.csv and results/cec2020_wilcoxon.csv")
    return res2020, df2020


def main():
    import sys
    if "--cec-all" in sys.argv:
        sys.argv.remove("--cec-all")
        _original_main()   # runs CEC-2014/2017 + engineering
        print("\n" + "="*60)
        print("  Now running CEC-2020 and CEC-2022 ...")
        print("="*60)
        run_cec_2020_2022(max_fes=60_000, runs=50)
        print("\n[DONE] All four suites complete. Paste ALL csv files to Claude.")
    elif "--cec2020" in sys.argv:
        # Run ONLY CEC-2020 (for parallel split with --cec2022 on another machine)
        sys.argv.remove("--cec2020")
        run_cec_2020_only(max_fes=60_000, runs=50)
    elif "--cec2022" in sys.argv:
        # Run ONLY CEC-2022 (for parallel split with --cec2020 on another machine)
        sys.argv.remove("--cec2022")
        run_cec_2022_only(max_fes=60_000, runs=50)
    elif "--cec-both" in sys.argv:
        # Run CEC-2020 + CEC-2022 sequentially on one machine
        sys.argv.remove("--cec-both")
        run_cec_2020_2022(max_fes=60_000, runs=50)
    else:
        _original_main()

if __name__ == "__main__":
    main()
