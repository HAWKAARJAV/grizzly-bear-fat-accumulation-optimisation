# GBFAO v1 – Basic Grizzly Bear Fat Accumulation Optimiser
# Three phases driven by fat-accumulation ratio alpha = t/T:
#   alpha < 1/3  -> Foraging (exploration)
#   alpha < 2/3  -> Hyperphagia (balanced)
#   alpha >= 2/3 -> Hibernation (exploitation)

import numpy as np

def GBFAO_v1(func, lb, ub, dim, max_fes=60_000, pop=50):
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
            x = pos[i].copy()

            if alpha < 1/3:       # Foraging
                r1  = rng.random(dim)
                r2  = rng.random(dim)
                idx = rng.integers(pop, size=2)
                while idx[0] == i: idx[0] = rng.integers(pop)
                while idx[1] == i or idx[1] == idx[0]: idx[1] = rng.integers(pop)
                x_new = x + r1*(best_pos - x) + r2*(pos[idx[0]] - pos[idx[1]])

            elif alpha < 2/3:     # Hyperphagia
                r     = rng.random(dim)
                x_new = best_pos + r * np.exp(-alpha) * (best_pos - x)

            else:                 # Hibernation
                sigma = float(np.mean((1 - alpha) * (np.array(ub) - np.array(lb))))
                x_new = best_pos + rng.normal(0, sigma, dim)

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
    fit, pos, curve = GBFAO_v1(F7, -100, 100, 30, max_fes=60_000, pop=50)
    print(f"[GBFAO v1] F7 best: {fit:.6e}  |  iters: {len(curve)}")
