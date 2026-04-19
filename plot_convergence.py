"""
=============================================================
 Plotting Module – Convergence Curves & Result Heatmaps
 ─────────────────────────────────────────────────────────
 Functions
 ─────────
 plot_convergence()   – convergence curves for all algorithms
 plot_boxplots()      – box plots of 50-run fitness distributions
 plot_rank_heatmap()  – rank heatmap across functions
 plot_engineering()   – bar chart for engineering problems
 plot_version_comparison() – shows v1→v4 convergence speedup
=============================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Styling ───────────────────────────────────────────────────
COLORS = {
    "GBFAO-v4": "#E63946",
    "GBFAO-v3": "#FF9F1C",
    "GBFAO-v2": "#2EC4B6",
    "GBFAO-v1": "#8338EC",
    "PSO"     : "#3A86FF",
    "GWO"     : "#06D6A0",
    "WOA"     : "#FFB703",
    "DE"      : "#FB8500",
    "SMA"     : "#8D99AE",
    "HHO"     : "#EF233C",
    "AO"      : "#4CC9F0",
}
LINESTYLES = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,1))]

plt.rcParams.update({
    "figure.dpi"        : 150,
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "legend.fontsize"   : 7,
})

os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. Convergence curves
# ─────────────────────────────────────────────────────────────

def plot_convergence(curves_dict: dict,
                     func_names: list,
                     save_dir: str = "results/convergence",
                     ncols: int = 5):
    """
    Plot convergence curves for all functions.

    Parameters
    ----------
    curves_dict : {algo_name: {func_name: list[float]}}
                  median curve across runs for each algo/function
    func_names  : list of function names
    save_dir    : directory to save figures
    ncols       : subplot columns per figure page
    """
    os.makedirs(save_dir, exist_ok=True)
    algo_names = list(curves_dict.keys())
    nfuncs     = len(func_names)
    nrows      = (nfuncs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.8))
    axes = axes.flatten()

    for fi, fname in enumerate(func_names):
        ax = axes[fi]
        for ai, algo in enumerate(algo_names):
            if fname not in curves_dict[algo]:
                continue
            curve = curves_dict[algo][fname]
            # log-scale safe
            curve_np = np.array(curve, dtype=float)
            curve_np = np.where(curve_np <= 0, 1e-300, curve_np)
            ax.semilogy(curve_np,
                        color=COLORS.get(algo, f"C{ai}"),
                        linestyle=LINESTYLES[ai % len(LINESTYLES)],
                        linewidth=1.2,
                        label=algo)
        ax.set_title(fname, fontsize=9, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=7)
        ax.set_ylabel("Fitness (log)", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide empty subplots
    for fi in range(nfuncs, len(axes)):
        axes[fi].set_visible(False)

    # Shared legend
    handles = [mpatches.Patch(color=COLORS.get(a, f"C{i}"), label=a)
               for i, a in enumerate(algo_names)]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(algo_names)//2 + 1,
               fontsize=7, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(save_dir, "convergence_all.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[✓] Convergence curves  → {out}")


# ─────────────────────────────────────────────────────────────
# 2. Box plots
# ─────────────────────────────────────────────────────────────

def plot_boxplots(results_dict: dict,
                  func_names: list,
                  save_dir: str = "results/boxplots",
                  ncols: int = 5):
    """
    Box-and-whisker plots of 50-run distributions per function.

    Parameters
    ----------
    results_dict : {algo_name: {func_name: np.ndarray(50,)}}
    """
    os.makedirs(save_dir, exist_ok=True)
    algo_names = list(results_dict.keys())
    nfuncs     = len(func_names)
    nrows      = (nfuncs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.0))
    axes = axes.flatten()

    for fi, fname in enumerate(func_names):
        ax   = axes[fi]
        data = [results_dict[algo].get(fname, np.array([np.nan]))
                for algo in algo_names]
        bp   = ax.boxplot(data, patch_artist=True,
                          medianprops={"color": "black", "linewidth": 1.5})
        for patch, algo in zip(bp["boxes"], algo_names):
            patch.set_facecolor(COLORS.get(algo, "steelblue"))
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(algo_names) + 1))
        ax.set_xticklabels(algo_names, rotation=45, ha="right", fontsize=6)
        ax.set_title(fname, fontsize=9, fontweight="bold")
        ax.set_ylabel("Fitness", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)

    for fi in range(nfuncs, len(axes)):
        axes[fi].set_visible(False)

    plt.tight_layout()
    out = os.path.join(save_dir, "boxplots_all.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[✓] Box plots           → {out}")


# ─────────────────────────────────────────────────────────────
# 3. Rank heatmap
# ─────────────────────────────────────────────────────────────

def plot_rank_heatmap(mean_dict: dict,
                      func_names: list,
                      save_path: str = "results/rank_heatmap.pdf"):
    """
    Heatmap of algorithm ranks per function (lower = better).

    Parameters
    ----------
    mean_dict : {algo_name: {func_name: float}}  mean fitness values
    """
    algo_names = list(mean_dict.keys())
    data = np.array([[mean_dict[a].get(f, np.nan)
                      for f in func_names]
                     for a in algo_names])          # (n_algos, n_funcs)

    ranks = np.zeros_like(data, dtype=float)
    for fi in range(data.shape[1]):
        col = data[:, fi]
        valid = ~np.isnan(col)
        ranked = np.empty_like(col)
        ranked[:] = np.nan
        ranked[valid] = np.argsort(np.argsort(col[valid])) + 1
        ranks[:, fi] = ranked

    fig, ax = plt.subplots(figsize=(max(12, len(func_names)*0.5),
                                    max(4,  len(algo_names)*0.6)))
    im = ax.imshow(ranks, aspect="auto", cmap="RdYlGn_r",
                   vmin=1, vmax=len(algo_names))
    plt.colorbar(im, ax=ax, label="Rank (1=best)")

    ax.set_xticks(range(len(func_names)))
    ax.set_xticklabels(func_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(algo_names)))
    ax.set_yticklabels(algo_names, fontsize=9)
    ax.set_title("Algorithm Rank per Benchmark Function (CEC-2014 style)",
                 fontsize=11, fontweight="bold", pad=10)

    for ai in range(len(algo_names)):
        for fi in range(len(func_names)):
            v = ranks[ai, fi]
            if not np.isnan(v):
                ax.text(fi, ai, f"{v:.0f}", ha="center", va="center",
                        fontsize=6, color="black" if v < len(algo_names)//2 else "white")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] Rank heatmap        → {save_path}")


# ─────────────────────────────────────────────────────────────
# 4. Engineering problem bar chart
# ─────────────────────────────────────────────────────────────

def plot_engineering(eng_results: dict,
                     problem_names: list,
                     save_path: str = "results/engineering_results.pdf"):
    """
    Grouped bar chart of mean fitness on engineering problems.

    Parameters
    ----------
    eng_results : {algo_name: {prob_name: float}}  mean fitness
    """
    algo_names = list(eng_results.keys())
    n_algos    = len(algo_names)
    n_probs    = len(problem_names)
    x          = np.arange(n_probs)
    width      = 0.8 / n_algos

    fig, ax = plt.subplots(figsize=(max(10, n_probs * 2), 5))
    for ai, algo in enumerate(algo_names):
        vals = [eng_results[algo].get(p, np.nan) for p in problem_names]
        ax.bar(x + ai*width - 0.4 + width/2,
               vals, width, label=algo,
               color=COLORS.get(algo, f"C{ai}"), alpha=0.8, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(problem_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean Fitness (lower = better)", fontsize=9)
    ax.set_title("Engineering Problem Benchmark – Mean Best Fitness",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=n_algos//2 + 1)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] Engineering chart   → {save_path}")


# ─────────────────────────────────────────────────────────────
# 5. Version comparison (v1 → v4 convergence)
# ─────────────────────────────────────────────────────────────

def plot_version_comparison(version_curves: dict,
                            func_name: str = "F7",
                            save_path: str = "results/version_comparison.pdf"):
    """
    Show how each GBFAO version converges faster than the previous.

    Parameters
    ----------
    version_curves : {version_label: list[float]}
    func_name      : function used for comparison
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for vi, (label, curve) in enumerate(version_curves.items()):
        curve_np = np.array(curve, dtype=float)
        curve_np = np.where(curve_np <= 0, 1e-300, curve_np)
        # normalise x-axis to 0–1 for fair comparison
        x = np.linspace(0, 1, len(curve_np))
        ax.semilogy(x, curve_np,
                    color=COLORS.get(label, f"C{vi}"),
                    linewidth=2.0,
                    linestyle=LINESTYLES[vi % len(LINESTYLES)],
                    label=label)

    ax.set_xlabel("Normalised Budget (0 = start, 1 = end)", fontsize=10)
    ax.set_ylabel("Best Fitness (log scale)", fontsize=10)
    ax.set_title(f"GBFAO Version Comparison on {func_name}\n"
                 f"Each version achieves better results with fewer iterations",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] Version comparison  → {save_path}")
