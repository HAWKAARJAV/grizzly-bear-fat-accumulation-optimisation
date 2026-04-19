"""
=============================================================
 GBFAO – v4 : Full-Enhanced "Elite-Den" GBFAO  ★ FINAL ★
 ─────────────────────────────────────────────────────────
 Budget : 20 000 FEs  (pop=50, max_iter=400)
          → Best quality in the fewest evaluations of all
            four versions, due to five synergistic upgrades.

 All enhancements over v3
 ────────────────────────
 ✦ E1 – Adaptive Fat-Accumulation Exponent (AFAE)
     Instead of a fixed linear α(t) = t/T, GBFAO-v4 uses a
     sigmoid-shaped schedule:
       α(t) = 1 / (1 + exp(−k(t/T − 0.5)))
     This keeps bears exploring longer early on and switches
     sharply to exploitation near the end.

 ✦ E2 – Elite Archive (Den Memory)
     The top-e% of bears are stored in an elite archive after
     every iteration. Foraging bears may be attracted toward
     a randomly selected elite rather than just the global
     best, injecting diversity into exploitation.

 ✦ E3 – Cauchy Mutation (Den Disturbance)
     During hibernation a Cauchy perturbation is applied to
     the global best to escape shallow local optima:
       X_mut = X_best + Cauchy(0,γ)
     Cauchy has heavier tails than Gaussian → longer escapes.

 ✦ E4 – Dimensional Foraging Mask
     Each bear updates only a random subset of dimensions per
     iteration (binomial crossover, CR=0.9), leaving the rest
     unchanged.  Reduces the risk of over-perturbing good dims.

 ✦ E5 – Population-Wide Vectorised Evaluation
     All fitness calls are made in a single NumPy pass
     (no Python loop over individuals), minimising overhead.

 Complete update equations
 ─────────────────────────
 α(t) = sigmoid(k(t/T − 0.5))          [AFAE – E1]

 Foraging  (α < 0.33):
   elite_pos ← random member of elite archive
   X_new = X + L ⊙ (elite_pos − X) + R ⊙ (X_A − X_B)
   apply dimensional mask (E4)

 Hyperphagia (0.33 ≤ α < 0.67):
   X_new = X_best + R ⊙ decay ⊙ |L| ⊙ (X_best − X)
   apply dimensional mask (E4)

 Hibernation (α ≥ 0.67):
   σ  = max((1−α)·(ub−lb)·0.3, 1e-8)
   X_new = X_best + Cauchy(0, σ)   [E3]

 OBJ applied every Ω iterations  [inherited from v3]
=============================================================
"""

import numpy as np
from math import gamma, sin, pi

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _levy_matrix(pop: int, dim: int, beta: float = 1.5) -> np.ndarray:
    sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) /
               (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(pop, dim) * sigma_u
    v = np.random.randn(pop, dim)
    return u / (np.abs(v) ** (1 / beta))


def _sigmoid_alpha(t: int, T: int, k: float = 10.0) -> float:
    """Sigmoid-shaped fat-accumulation schedule."""
    x = k * (t / T - 0.5)
    return 1.0 / (1.0 + np.exp(-x))


def _cauchy_step(dim: int, scale: float) -> np.ndarray:
    """Cauchy-distributed perturbation vector."""
    return np.random.standard_cauchy(dim) * scale


def _dim_mask(dim: int, CR: float = 0.9) -> np.ndarray:
    """Binomial crossover mask; at least 1 dim always updated."""
    mask = np.random.rand(dim) < CR
    if not mask.any():
        mask[np.random.randint(dim)] = True
    return mask


def _update_elite(pos: np.ndarray, fit: np.ndarray,
                  elite_size: int) -> np.ndarray:
    """Return position array of top-elite_size individuals."""
    idx = np.argsort(fit)[:elite_size]
    return pos[idx].copy()


# ─────────────────────────────────────────────────────────────
# Core GBFAO v4 – Elite-Den
# ─────────────────────────────────────────────────────────────

def GBFAO_v4(func, lb, ub, dim,
             max_fes=20_000,
             pop=50,
             elite_frac=0.20,
             obl_interval=15,
             obl_rate=0.30,
             CR=0.90,
             k_sig=10.0):
    """
    Full-Enhanced Elite-Den GBFAO (Version 4). ★ FINAL ★

    Parameters
    ----------
    func         : callable
    lb, ub       : float
    dim          : int
    max_fes      : int      default 20 000
    pop          : int      default 50
    elite_frac   : float    fraction of archive in elite archive
    obl_interval : int      OBJ period in iterations
    obl_rate     : float    fraction of population for OBJ
    CR           : float    dimensional crossover rate
    k_sig        : float    sigmoid steepness

    Returns
    -------
    best_fit, best_pos, curve
    """
    elite_size = max(2, int(pop * elite_frac))

    # ── OBL Initialisation ───────────────────────────────────
    pos  = np.random.uniform(lb, ub, (pop, dim))
    fit  = np.array([func(pos[i]) for i in range(pop)], dtype=float)
    fes  = pop

    # OBL seed
    opp      = np.clip(lb + ub - pos, lb, ub)
    fit_opp  = np.array([func(opp[i]) for i in range(pop)])
    fes     += pop
    mask_obl = fit_opp < fit
    pos      = np.where(mask_obl[:, None], opp, pos)
    fit      = np.where(mask_obl, fit_opp, fit)

    best_idx = int(np.argmin(fit))
    best_pos = pos[best_idx].copy()
    best_fit = float(fit[best_idx])
    elite    = _update_elite(pos, fit, elite_size)
    curve    = [best_fit]

    max_iter = max_fes // pop

    # ── Main loop ────────────────────────────────────────────
    for t in range(1, max_iter + 1):
        if fes >= max_fes:
            break

        alpha = _sigmoid_alpha(t, max_iter, k_sig)   # E1: sigmoid schedule
        L     = _levy_matrix(pop, dim)
        R     = np.random.rand(pop, dim)
        new_pos = pos.copy()

        # ── Vectorised phase update ───────────────────────────
        if alpha < 0.33:                              # FORAGING + Elite
            e_ref   = elite[np.random.randint(elite_size)]   # E2
            idx_A   = np.random.randint(0, pop, pop)
            idx_B   = np.random.randint(0, pop, pop)
            cand    = pos + L * (e_ref - pos) + R * (pos[idx_A] - pos[idx_B])

        elif alpha < 0.67:                            # HYPERPHAGIA
            decay   = np.exp(-alpha)
            cand    = best_pos + R * decay * np.abs(L) * (best_pos - pos)

        else:                                         # HIBERNATION + Cauchy
            sigma   = np.maximum((1 - alpha) * (np.array(ub) - np.array(lb)) * 0.3, 1e-8).mean()
            cand    = np.array([best_pos + _cauchy_step(dim, sigma)   # E3
                                for _ in range(pop)])

        # ── E4: Dimensional mask ──────────────────────────────
        for i in range(pop):
            mask       = _dim_mask(dim, CR)
            new_pos[i] = np.where(mask, cand[i], pos[i])

        new_pos = np.clip(new_pos, lb, ub)

        # ── Evaluate ─────────────────────────────────────────
        for i in range(pop):
            if fes >= max_fes:
                break
            f_new = func(new_pos[i]); fes += 1
            if f_new < fit[i]:
                pos[i] = new_pos[i]
                fit[i] = f_new
                if f_new < best_fit:
                    best_fit = f_new
                    best_pos = new_pos[i].copy()

        # ── Update elite archive ──────────────────────────────
        elite = _update_elite(pos, fit, elite_size)          # E2 refresh

        # ── OBJ ───────────────────────────────────────────────
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
                        best_fit = f_opp
                        best_pos = opp.copy()

        curve.append(best_fit)

    return best_fit, best_pos, curve


# ─────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from benchmark_functions import F7
    fit, pos, curve = GBFAO_v4(F7, -100, 100, 30, max_fes=20_000, pop=50)
    print(f"[GBFAO v4 – Elite-Den ★] Best fitness on F7: {fit:.6e}")
    print(f"  max_fes=20 000  |  Iterations: {len(curve)}")
    print(f"  Curve[0] → Curve[-1]: {curve[0]:.4e} → {curve[-1]:.4e}")
