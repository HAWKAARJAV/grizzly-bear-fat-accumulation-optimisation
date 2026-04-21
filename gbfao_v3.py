# GBFAO v3 – Lévy + Opposition-Based Learning (OBL)
# Adds OBL population seeding and periodic OBL jumps to escape local optima.
# Vectorised population updates for speed. Default budget 30,000 FEs.

import numpy as np
from math import gamma, sin, pi

def _levy_matrix(pop, dim, beta=1.5):
    """Lévy step matrix of shape (pop, dim)."""
    sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) /
               (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(pop, dim) * sigma_u
    v = np.random.randn(pop, dim)
    return u / (np.abs(v) ** (1 / beta))


def _obl_population(pos, fits, lb, ub, func):
    """Generate opposites; keep better of each pair. Returns updated pop + FEs used."""
    opp      = np.clip(lb + ub - pos, lb, ub)
    fits_opp = np.array([func(opp[i]) for i in range(len(pos))])
    mask     = fits_opp < fits
    pos      = np.where(mask[:, None], opp, pos)
    fits     = np.where(mask, fits_opp, fits)
    return pos, fits, len(pos)


def _reflect(x, lb, ub):
    span = ub - lb
    x[x < lb] = lb + np.abs(x[x < lb] - lb) % span
    x[x > ub] = ub - np.abs(x[x > ub] - ub) % span
    return np.clip(x, lb, ub)


def GBFAO_v3(func, lb, ub, dim, max_fes=30_000, pop=50,
             obl_interval=20, obl_rate=0.30):
    """Returns (best_fit, best_pos, convergence_curve)."""
    pos = np.random.uniform(lb, ub, (pop, dim))
    fit = np.array([func(pos[i]) for i in range(pop)], dtype=float)
    fes = pop

    # OBL initialisation
    pos, fit, extra = _obl_population(pos, fit, lb, ub, func)
    fes += extra

    best_idx = int(np.argmin(fit))
    best_pos = pos[best_idx].copy()
    best_fit = float(fit[best_idx])
    curve    = [best_fit]
    max_iter = max_fes // pop

    for t in range(1, max_iter + 1):
        if fes >= max_fes:
            break

        alpha = t / max_iter
        L     = _levy_matrix(pop, dim)
        R     = np.random.rand(pop, dim)

        if alpha < 1/3:    # Foraging + Lévy
            idx_A   = np.random.randint(0, pop, pop)
            idx_B   = np.random.randint(0, pop, pop)
            new_pos = pos + L * (best_pos - pos) + R * (pos[idx_A] - pos[idx_B])

        elif alpha < 2/3:  # Hyperphagia + Lévy
            decay   = np.exp(-alpha)
            new_pos = best_pos + R * decay * np.abs(L) * (best_pos - pos)

        else:              # Hibernation
            sigma   = max(float(np.mean((1 - alpha) * (np.array(ub) - np.array(lb)) * 0.5)), 1e-8)
            new_pos = best_pos + np.random.normal(0, sigma, (pop, dim))

        new_pos = np.clip(new_pos, lb, ub)

        for i in range(pop):
            if fes >= max_fes:
                break
            f_new = func(new_pos[i]); fes += 1
            if f_new < fit[i]:
                pos[i] = new_pos[i]; fit[i] = f_new
                if f_new < best_fit:
                    best_fit = f_new; best_pos = new_pos[i].copy()

        # Opposition-Based Jump for worst individuals
        if t % obl_interval == 0 and fes < max_fes:
            n_jump   = max(1, int(pop * obl_rate))
            worst_ids = np.argsort(fit)[::-1][:n_jump]
            for idx in worst_ids:
                if fes >= max_fes:
                    break
                opp   = np.clip(lb + ub - pos[idx], lb, ub)
                f_opp = func(opp); fes += 1
                if f_opp < fit[idx]:
                    pos[idx] = opp; fit[idx] = f_opp
                    if f_opp < best_fit:
                        best_fit = f_opp; best_pos = opp.copy()

        curve.append(best_fit)

    return best_fit, best_pos, curve


if __name__ == "__main__":
    from benchmark_functions import F7
    fit, pos, curve = GBFAO_v3(F7, -100, 100, 30, max_fes=30_000, pop=50)
    print(f"[GBFAO v3] F7 best: {fit:.6e}  |  iters: {len(curve)}")
