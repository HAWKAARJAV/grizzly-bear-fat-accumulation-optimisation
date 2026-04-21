# GBFAO v2 – Lévy-Flight Enhanced GBFAO
# Adds Lévy-flight steps to foraging and hyperphagia phases for better exploration.
# Default budget 45,000 FEs (achieves comparable quality to v1 in fewer evaluations).

import numpy as np
from math import gamma, sin, pi

def _levy(dim, beta=1.5):
    """Lévy-distributed random step vector."""
    sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) /
               (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma_u
    v = np.random.randn(dim)
    return u / (np.abs(v) ** (1 / beta))


def GBFAO_v2(func, lb, ub, dim, max_fes=45_000, pop=50):
    """Returns (best_fit, best_pos, convergence_curve)."""
    rng = np.random.default_rng()
    pos = rng.uniform(lb, ub, (pop, dim))
    fit = np.array([func(pos[i]) for i in range(pop)], dtype=float)
    fes = pop

    best_idx = int(np.argmin(fit))
    best_pos = pos[best_idx].copy()
    best_fit = float(fit[best_idx])
    curve    = [best_fit]
    max_iter = max_fes // pop

    for t in range(1, max_iter + 1):
        if fes >= max_fes:
            break
        alpha = t / max_iter

        for i in range(pop):
            x   = pos[i].copy()
            lev = _levy(dim)  # Lévy step

            if alpha < 1/3:       # Foraging + Lévy
                r   = rng.random(dim)
                idx = rng.integers(0, pop, size=2)
                while idx[0] == i: idx[0] = rng.integers(pop)
                while idx[1] == i or idx[1] == idx[0]: idx[1] = rng.integers(pop)
                x_new = x + lev * (best_pos - x) + r * (pos[idx[0]] - pos[idx[1]])

            elif alpha < 2/3:     # Hyperphagia + Lévy
                r     = rng.random(dim)
                decay = np.exp(-alpha)
                x_new = best_pos + r * decay * np.abs(lev) * (best_pos - x)

            else:                 # Hibernation (adaptive sigma)
                sigma = float(np.mean((1 - alpha) * (np.array(ub) - np.array(lb)) * 0.5))
                x_new = best_pos + rng.normal(0, max(sigma, 1e-8), dim)

            x_new = np.clip(x_new, lb, ub)
            f_new = func(x_new); fes += 1
            if f_new < fit[i]:
                pos[i] = x_new; fit[i] = f_new
                if f_new < best_fit:
                    best_fit = f_new; best_pos = x_new.copy()

            if fes >= max_fes:
                break

        curve.append(best_fit)

    return best_fit, best_pos, curve


if __name__ == "__main__":
    from benchmark_functions import F7
    fit, pos, curve = GBFAO_v2(F7, -100, 100, 30, max_fes=45_000, pop=50)
    print(f"[GBFAO v2] F7 best: {fit:.6e}  |  iters: {len(curve)}")
