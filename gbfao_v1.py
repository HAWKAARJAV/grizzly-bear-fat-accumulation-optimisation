"""
=============================================================
 GBFAO – v1 : Basic Grizzly Bear Fat Accumulation Optimiser
 ─────────────────────────────────────────────────────────
 Budget : 60 000 FEs  (pop=50, max_iter=1200)

 Biological inspiration
 ──────────────────────
 Grizzly bears cycle through three seasonal behaviours:
   Phase 1 – Spring/Summer FORAGING     → Global exploration
   Phase 2 – Autumn       HYPERPHAGIA   → Transition (balanced)
   Phase 3 – Winter       HIBERNATION   → Local exploitation

 Fat-accumulation ratio : α(t) = t / T
   α ∈ [0, 1/3)  → Foraging     (explore widely)
   α ∈ [1/3,2/3) → Hyperphagia  (intensify near food)
   α ∈ [2/3, 1]  → Hibernation  (fine-tune around best den)

 Position update equations
 ─────────────────────────
 Foraging:
   X_new = X + r1*(X_best − X) + r2*(X_rnd1 − X_rnd2)

 Hyperphagia:
   X_new = X_best + r * exp(−α) * (X_best − X)

 Hibernation:
   σ  = (1 − α) * (ub − lb)
   X_new = X_best + N(0, σ²)
=============================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# Core GBFAO v1
# ─────────────────────────────────────────────────────────────

def GBFAO_v1(func, lb, ub, dim,
             max_fes=60_000,
             pop=50):
    """
    Basic GBFAO (Version 1).

    Parameters
    ----------
    func    : callable   objective function f(x) → float
    lb, ub  : float      search-space bounds
    dim     : int        problem dimensionality
    max_fes : int        max function evaluations
    pop     : int        population size

    Returns
    -------
    best_fit  : float           best fitness found
    best_pos  : np.ndarray      corresponding position
    curve     : list[float]     convergence curve (per iteration)
    """
    # ── Initialisation ──────────────────────────────────────
    rng = np.random.default_rng()
    pos = rng.uniform(lb, ub, (pop, dim))
    fit = np.array([func(pos[i]) for i in range(pop)], dtype=float)
    fes = pop

    best_idx = int(np.argmin(fit))
    best_pos = pos[best_idx].copy()
    best_fit = float(fit[best_idx])
    curve    = [best_fit]

    max_iter = max_fes // pop

    # ── Main loop ────────────────────────────────────────────
    for t in range(1, max_iter + 1):
        if fes >= max_fes:
            break

        alpha = t / max_iter          # fat-accumulation ratio [0,1]

        for i in range(pop):
            x = pos[i].copy()

            # ── Phase selection ──────────────────────────────
            if alpha < 1/3:           # FORAGING – Exploration
                r1  = rng.random(dim)
                r2  = rng.random(dim)
                idx = rng.integers(pop, size=2)
                while idx[0] == i: idx[0] = rng.integers(pop)
                while idx[1] == i or idx[1] == idx[0]: idx[1] = rng.integers(pop)
                x_new = x + r1*(best_pos - x) + r2*(pos[idx[0]] - pos[idx[1]])

            elif alpha < 2/3:         # HYPERPHAGIA – Balanced
                r     = rng.random(dim)
                x_new = best_pos + r * np.exp(-alpha) * (best_pos - x)

            else:                     # HIBERNATION – Exploitation
                sigma = float(np.mean((1 - alpha) * (np.array(ub) - np.array(lb))))
                x_new = best_pos + rng.normal(0, sigma, dim)

            # ── Boundary handling ────────────────────────────
            x_new = np.clip(x_new, lb, ub)

            # ── Greedy selection ─────────────────────────────
            f_new = func(x_new); fes += 1
            if f_new < fit[i]:
                pos[i] = x_new
                fit[i] = f_new
                if f_new < best_fit:
                    best_fit = f_new
                    best_pos = x_new.copy()

            if fes >= max_fes:
                break

        curve.append(best_fit)

    return best_fit, best_pos, curve


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from benchmark_functions import F7   # Rastrigin
    fit, pos, curve = GBFAO_v1(F7, -100, 100, 30, max_fes=60_000, pop=50)
    print(f"[GBFAO v1] Best fitness on F7 (Rastrigin): {fit:.6e}")
    print(f"           Total iterations   : {len(curve)}")
    print(f"           Curve[0]  → Curve[-1]: {curve[0]:.4e} → {curve[-1]:.4e}")
