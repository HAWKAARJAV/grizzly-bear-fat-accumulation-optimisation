# GBFAO v4 – Elite-Den GBFAO (Final Version)
# Enhancements over v3:
#   E1: Sigmoid fat-accumulation schedule (keeps exploration longer)
#   E2: Elite archive – foraging bears attracted to top-k individuals
#   E3: Cauchy mutation during hibernation to escape local optima
#   E4: Dimensional mask (binomial crossover) per bear per iteration
# Default budget 20,000 FEs.

import numpy as np
from math import gamma, sin, pi
from typing import Optional

def _levy_matrix(pop, dim, beta=1.5):
    sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) /
               (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(pop, dim) * sigma_u
    v = np.random.randn(pop, dim)
    return u / (np.abs(v) ** (1 / beta))

def _sigmoid_alpha(t, T, k=10.0):
    """Sigmoid-shaped fat-accumulation schedule."""
    return 1.0 / (1.0 + np.exp(-k * (t / T - 0.5)))

def _cauchy_step(dim, scale):
    return np.random.standard_cauchy(dim) * scale

def _dim_mask(dim, CR=0.9):
    """Binomial crossover mask; at least one dimension always updated."""
    mask = np.random.rand(dim) < CR
    if not mask.any():
        mask[np.random.randint(dim)] = True
    return mask

def _update_elite(pos, fit, elite_size):
    return pos[np.argsort(fit)[:elite_size]].copy()

def _phase_name(alpha):
    if alpha < 0.33: return "Foraging"
    if alpha < 0.67: return "Hyperphagia"
    return "Hibernation"

def _snapshot(pos, best_pos, best_fit, alpha, iteration, fes):
    return {"positions": pos.copy(), "best_pos": best_pos.copy(),
            "best_fit": float(best_fit), "alpha": float(alpha),
            "phase": _phase_name(alpha), "iteration": int(iteration), "fes": int(fes)}


def GBFAO_v4(func, lb, ub, dim, max_fes=20_000, pop=50,
             elite_frac=0.20, obl_interval=15, obl_rate=0.30,
             CR=0.90, k_sig=10.0, return_history=False):
    """Returns (best_fit, best_pos, curve) or with history if return_history=True."""
    elite_size = max(2, int(pop * elite_frac))

    # OBL initialisation
    pos = np.random.uniform(lb, ub, (pop, dim))
    fit = np.array([func(pos[i]) for i in range(pop)], dtype=float)
    fes = pop

    opp     = np.clip(lb + ub - pos, lb, ub)
    fit_opp = np.array([func(opp[i]) for i in range(pop)])
    fes    += pop
    mask    = fit_opp < fit
    pos     = np.where(mask[:, None], opp, pos)
    fit     = np.where(mask, fit_opp, fit)

    best_idx = int(np.argmin(fit))
    best_pos = pos[best_idx].copy()
    best_fit = float(fit[best_idx])
    elite    = _update_elite(pos, fit, elite_size)
    curve    = [best_fit]
    history  = []

    if return_history:
        history.append(_snapshot(pos, best_pos, best_fit, 0.0, 0, fes))

    max_iter = max_fes // pop

    for t in range(1, max_iter + 1):
        if fes >= max_fes:
            break

        alpha   = _sigmoid_alpha(t, max_iter, k_sig)  # E1
        L       = _levy_matrix(pop, dim)
        R       = np.random.rand(pop, dim)
        new_pos = pos.copy()

        if alpha < 0.33:    # Foraging – use random elite member (E2)
            e_ref   = elite[np.random.randint(elite_size)]
            idx_A   = np.random.randint(0, pop, pop)
            idx_B   = np.random.randint(0, pop, pop)
            cand    = pos + L * (e_ref - pos) + R * (pos[idx_A] - pos[idx_B])

        elif alpha < 0.67:  # Hyperphagia
            decay   = np.exp(-alpha)
            cand    = best_pos + R * decay * np.abs(L) * (best_pos - pos)

        else:               # Hibernation + Cauchy mutation (E3)
            sigma   = np.maximum((1 - alpha) * (np.array(ub) - np.array(lb)) * 0.3, 1e-8).mean()
            cand    = np.array([best_pos + _cauchy_step(dim, sigma) for _ in range(pop)])

        # E4: dimensional mask per bear
        for i in range(pop):
            mask_d     = _dim_mask(dim, CR)
            new_pos[i] = np.where(mask_d, cand[i], pos[i])

        new_pos = np.clip(new_pos, lb, ub)

        for i in range(pop):
            if fes >= max_fes:
                break
            f_new = func(new_pos[i]); fes += 1
            if f_new < fit[i]:
                pos[i] = new_pos[i]; fit[i] = f_new
                if f_new < best_fit:
                    best_fit = f_new; best_pos = new_pos[i].copy()

        elite = _update_elite(pos, fit, elite_size)

        # Opposition-Based Jump
        if t % obl_interval == 0 and fes < max_fes:
            n_jump   = max(1, int(pop * obl_rate))
            worst_id = np.argsort(fit)[::-1][:n_jump]
            for idx in worst_id:
                if fes >= max_fes:
                    break
                opp   = np.clip(lb + ub - pos[idx], lb, ub)
                f_opp = func(opp); fes += 1
                if f_opp < fit[idx]:
                    pos[idx] = opp; fit[idx] = f_opp
                    if f_opp < best_fit:
                        best_fit = f_opp; best_pos = opp.copy()

        curve.append(best_fit)
        if return_history:
            history.append(_snapshot(pos, best_pos, best_fit, alpha, t, fes))

    if return_history:
        return best_fit, best_pos, curve, history
    return best_fit, best_pos, curve


def animate_gbfao_v4(func, lb, ub, dim=2, max_fes=4_000, pop=25,
                     grid_points=120, interval=80, repeat=False,
                     save_path=None, show=True):
    """Visualise GBFAO-v4 on a 2D landscape (dim must be 2)."""
    if dim != 2:
        raise ValueError("animate_gbfao_v4 requires dim=2.")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    best_fit, best_pos, curve, history = GBFAO_v4(
        func, lb, ub, dim, max_fes=max_fes, pop=pop, return_history=True)

    x = np.linspace(lb, ub, grid_points)
    y = np.linspace(lb, ub, grid_points)
    xx, yy = np.meshgrid(x, y)
    surface = np.empty_like(xx, dtype=float)
    for i in range(grid_points):
        for j in range(grid_points):
            surface[i, j] = func(np.array([xx[i, j], yy[i, j]], dtype=float))

    fig, (ax_landscape, ax_curve) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1.0]})
    fig.suptitle("GBFAO-v4 Population Animation", fontsize=14, fontweight="bold")

    contour = ax_landscape.contourf(xx, yy, surface, levels=40, cmap="viridis")
    fig.colorbar(contour, ax=ax_landscape, fraction=0.046, pad=0.04, label="Fitness")
    ax_landscape.set_title("Search Landscape")
    ax_landscape.set_xlabel("x1"); ax_landscape.set_ylabel("x2")
    ax_landscape.set_xlim(lb, ub); ax_landscape.set_ylim(lb, ub)

    scatter     = ax_landscape.scatter([], [], s=45, c="#ff6b35", edgecolors="white",
                                       linewidths=0.6, label="Bears")
    best_marker, = ax_landscape.plot([], [], marker="*", markersize=14,
                                     color="#e63946", linestyle="None", label="Best den")
    phase_text  = ax_landscape.text(
        0.02, 0.98, "", transform=ax_landscape.transAxes, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax_landscape.legend(loc="lower right")

    ax_curve.set_title("Convergence"); ax_curve.set_xlabel("Iteration")
    ax_curve.set_ylabel("Best fitness"); ax_curve.grid(True, alpha=0.3)
    ax_curve.set_xlim(0, max(1, len(curve) - 1))
    curve_min = min(curve); curve_max = max(curve)
    pad = max((curve_max - curve_min) * 0.1, 1e-9)
    ax_curve.set_ylim(curve_min - pad, curve_max + pad)
    curve_line,  = ax_curve.plot([], [], color="#1d3557", linewidth=2.2)
    curve_point, = ax_curve.plot([], [], "o", color="#e63946", markersize=6)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        best_marker.set_data([], [])
        curve_line.set_data([], []); curve_point.set_data([], [])
        phase_text.set_text("")
        return scatter, best_marker, curve_line, curve_point, phase_text

    def update(frame_idx):
        frame = history[frame_idx]
        scatter.set_offsets(frame["positions"][:, :2])
        best_marker.set_data([frame["best_pos"][0]], [frame["best_pos"][1]])
        curve_line.set_data(np.arange(frame_idx + 1), curve[:frame_idx + 1])
        curve_point.set_data([frame_idx], [curve[frame_idx]])
        phase_text.set_text(
            f"Phase: {frame['phase']}\nIter: {frame['iteration']}\n"
            f"FEs: {frame['fes']}\nBest: {frame['best_fit']:.4e}")
        return scatter, best_marker, curve_line, curve_point, phase_text

    anim = None
    if show or save_path:
        anim = FuncAnimation(fig, update, frames=len(history), init_func=init,
                             interval=interval, blit=False, repeat=repeat)
        fig._gbfao_anim = anim
        if save_path:
            anim.save(save_path, dpi=140)
        if show:
            plt.tight_layout(); plt.show()
        else:
            plt.close(fig)
    else:
        init(); update(len(history) - 1); plt.close(fig)

    return {"best_fit": best_fit, "best_pos": best_pos, "curve": curve,
            "history": history, "animation": anim, "figure": fig}


if __name__ == "__main__":
    from benchmark_functions import F7
    fit, pos, curve = GBFAO_v4(F7, -100, 100, 30, max_fes=20_000, pop=50)
    print(f"[GBFAO v4] F7 best: {fit:.6e}  |  iters: {len(curve) - 1}")
