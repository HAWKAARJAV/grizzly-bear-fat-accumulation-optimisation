# Competitor algorithms used as baselines in the benchmark experiments.
# All follow the same interface: best_fit, best_pos, curve = algo(func, lb, ub, dim, max_fes)

import numpy as np
import math

def _clip(x, lb, ub):
    return np.clip(x, lb, ub)

def _init_pop(n, dim, lb, ub):
    return np.random.uniform(lb, ub, (n, dim))


def PSO(func, lb, ub, dim, max_fes, pop=30, w=0.7, c1=2.0, c2=2.0):
    """Particle Swarm Optimization."""
    pos = _init_pop(pop, dim, lb, ub)
    vel = np.zeros((pop, dim))
    pbest_pos = pos.copy()
    pbest_fit = np.array([func(pos[i]) for i in range(pop)])
    fes = pop
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    curve = [gbest_fit]

    while fes < max_fes:
        w_cur = w - (w - 0.4) * fes / max_fes
        r1, r2 = np.random.rand(pop, dim), np.random.rand(pop, dim)
        vel = w_cur*vel + c1*r1*(pbest_pos - pos) + c2*r2*(gbest_pos - pos)
        pos = _clip(pos + vel, lb, ub)
        fits = np.array([func(pos[i]) for i in range(pop)])
        fes += pop
        better = fits < pbest_fit
        pbest_pos[better] = pos[better].copy()
        pbest_fit[better] = fits[better]
        best_idx = np.argmin(pbest_fit)
        if pbest_fit[best_idx] < gbest_fit:
            gbest_fit = pbest_fit[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()
        curve.append(gbest_fit)

    return gbest_fit, gbest_pos, curve


def GWO(func, lb, ub, dim, max_fes, pop=30):
    """Grey Wolf Optimizer."""
    pos  = _init_pop(pop, dim, lb, ub)
    fits = np.array([func(pos[i]) for i in range(pop)])
    fes  = pop
    idx  = np.argsort(fits)
    alpha_pos, beta_pos, delta_pos = pos[idx[0]].copy(), pos[idx[1]].copy(), pos[idx[2]].copy()
    alpha_fit = fits[idx[0]]
    curve = [alpha_fit]

    while fes < max_fes:
        a = 2 - 2*(fes/max_fes)
        for i in range(pop):
            for leader in [alpha_pos, beta_pos, delta_pos]:
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A = 2*a*r1 - a; C = 2*r2
                pos[i] = _clip(pos[i] + A*(leader - C*pos[i]), lb, ub)
        fits = np.array([func(pos[i]) for i in range(pop)])
        fes += pop
        idx = np.argsort(fits)
        alpha_pos, beta_pos, delta_pos = pos[idx[0]].copy(), pos[idx[1]].copy(), pos[idx[2]].copy()
        alpha_fit = fits[idx[0]]
        curve.append(alpha_fit)

    return alpha_fit, alpha_pos, curve


def WOA(func, lb, ub, dim, max_fes, pop=30):
    """Whale Optimization Algorithm."""
    pos  = _init_pop(pop, dim, lb, ub)
    fits = np.array([func(pos[i]) for i in range(pop)])
    fes  = pop
    best_idx = np.argmin(fits)
    best_pos = pos[best_idx].copy(); best_fit = fits[best_idx]
    curve = [best_fit]

    while fes < max_fes:
        a = 2 - 2*(fes/max_fes)
        for i in range(pop):
            r = np.random.rand(); A = 2*a*r - a; C = 2*np.random.rand(dim)
            p = np.random.rand()
            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C*best_pos - pos[i])
                    pos[i] = _clip(best_pos - A*D, lb, ub)
                else:
                    rand_idx = np.random.randint(pop)
                    D = np.abs(C*pos[rand_idx] - pos[i])
                    pos[i] = _clip(pos[rand_idx] - A*D, lb, ub)
            else:
                l = np.random.uniform(-1, 1); b = 1
                D = np.abs(best_pos - pos[i])
                pos[i] = _clip(D*np.exp(b*l)*np.cos(2*np.pi*l) + best_pos, lb, ub)
        fits = np.array([func(pos[i]) for i in range(pop)])
        fes += pop
        best_idx = np.argmin(fits)
        if fits[best_idx] < best_fit:
            best_fit = fits[best_idx]; best_pos = pos[best_idx].copy()
        curve.append(best_fit)

    return best_fit, best_pos, curve


def DE(func, lb, ub, dim, max_fes, pop=30, F=0.8, CR=0.9):
    """Differential Evolution (DE/rand/1/bin)."""
    pos  = _init_pop(pop, dim, lb, ub)
    fits = np.array([func(pos[i]) for i in range(pop)])
    fes  = pop
    best_idx = np.argmin(fits)
    best_pos = pos[best_idx].copy(); best_fit = fits[best_idx]
    curve = [best_fit]

    while fes < max_fes:
        for i in range(pop):
            idxs = list(range(pop)); idxs.remove(i)
            a, b, c = pos[np.random.choice(idxs, 3, replace=False)]
            mutant = _clip(a + F*(b - c), lb, ub)
            cross  = np.random.rand(dim) < CR
            if not cross.any(): cross[np.random.randint(dim)] = True
            trial   = np.where(cross, mutant, pos[i])
            f_trial = func(trial); fes += 1
            if f_trial < fits[i]:
                pos[i] = trial; fits[i] = f_trial
                if f_trial < best_fit:
                    best_fit = f_trial; best_pos = trial.copy()
        curve.append(best_fit)

    return best_fit, best_pos, curve


def SMA(func, lb, ub, dim, max_fes, pop=30):
    """Slime Mould Algorithm."""
    pos  = _init_pop(pop, dim, lb, ub)
    fits = np.array([func(pos[i]) for i in range(pop)])
    fes  = pop
    best_idx  = np.argmin(fits)
    best_pos  = pos[best_idx].copy(); best_fit = fits[best_idx]
    worst_fit = np.max(fits)
    W = np.ones(pop)
    curve = [best_fit]

    while fes < max_fes:
        a = np.arctanh(-(fes/max_fes) + 1)
        b = 1 - fes/max_fes
        idx_sorted = np.argsort(fits)
        for rank, idx in enumerate(idx_sorted):
            if rank < pop//2:
                W[idx] = 1 + np.random.rand()*np.log10((best_fit - fits[idx])/(best_fit - worst_fit + 1e-10) + 1)
            else:
                W[idx] = 1 - np.random.rand()*np.log10((best_fit - fits[idx])/(best_fit - worst_fit + 1e-10) + 1)
        for i in range(pop):
            if np.random.rand() < 0.03:
                pos[i] = np.random.uniform(lb, ub, dim)
            else:
                p  = np.tanh(abs(fits[i] - best_fit))
                vb = np.random.uniform(-a, a, dim)
                vc = np.random.uniform(-b, b, dim)
                r  = np.random.rand(dim)
                A  = np.random.randint(pop); B = np.random.randint(pop)
                pos[i] = np.where(r < p, best_pos + vb*(W[i]*pos[A] - pos[B]), vc*pos[i])
                pos[i] = _clip(pos[i], lb, ub)
        fits = np.array([func(pos[i]) for i in range(pop)])
        fes += pop
        best_idx = np.argmin(fits)
        if fits[best_idx] < best_fit:
            best_fit = fits[best_idx]; best_pos = pos[best_idx].copy()
        worst_fit = np.max(fits)
        curve.append(best_fit)

    return best_fit, best_pos, curve


def HHO(func, lb, ub, dim, max_fes, pop=30):
    """Harris Hawks Optimization."""
    pos  = _init_pop(pop, dim, lb, ub)
    fits = np.array([func(pos[i]) for i in range(pop)])
    fes  = pop
    best_idx = np.argmin(fits)
    best_pos = pos[best_idx].copy(); best_fit = fits[best_idx]
    curve = [best_fit]

    def levy(d):
        beta  = 1.5
        sigma = (math.gamma(1+beta)*np.sin(np.pi*beta/2) /
                 (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = np.random.randn(d)*sigma; v = np.random.randn(d)
        return u / np.abs(v)**(1/beta)

    while fes < max_fes:
        E1 = 2*(1 - fes/max_fes)
        for i in range(pop):
            E0 = 2*np.random.rand() - 1
            E  = E1*E0
            r  = np.random.rand()
            if abs(E) >= 1:  # Exploration
                rand_idx = np.random.randint(pop)
                if r >= 0.5:
                    pos[i] = _clip(pos[rand_idx] - np.random.rand()*np.abs(pos[rand_idx] - 2*np.random.rand()*pos[i]), lb, ub)
                else:
                    mu = np.mean(pos, axis=0)
                    pos[i] = _clip((best_pos - mu) - np.random.rand()*(lb + np.random.rand()*(ub-lb)), lb, ub)
            else:             # Exploitation
                J = 2*(1 - np.random.rand()); delta = best_pos - pos[i]
                if r >= 0.5 and abs(E) < 0.5:
                    pos[i] = _clip(best_pos - E*np.abs(delta), lb, ub)
                elif r >= 0.5 and abs(E) >= 0.5:
                    pos[i] = _clip(delta - E*np.abs(J*best_pos - pos[i]), lb, ub)
                elif r < 0.5 and abs(E) < 0.5:
                    lv = levy(dim)
                    Y = best_pos - E*np.abs(delta)
                    Z = Y + np.random.rand(dim)*lv
                    fY = func(Y); fZ = func(Z); fes += 2
                    pos[i] = _clip(Y if fY < fits[i] else (Z if fZ < fits[i] else pos[i]), lb, ub)
                else:
                    lv = levy(dim)
                    Y = best_pos - E*np.abs(J*best_pos - np.mean(pos, axis=0))
                    Z = Y + np.random.rand(dim)*lv
                    fY = func(Y); fZ = func(Z); fes += 2
                    pos[i] = _clip(Y if fY < fits[i] else (Z if fZ < fits[i] else pos[i]), lb, ub)

        fits = np.array([func(pos[i]) for i in range(pop)])
        fes += pop
        best_idx = np.argmin(fits)
        if fits[best_idx] < best_fit:
            best_fit = fits[best_idx]; best_pos = pos[best_idx].copy()
        curve.append(best_fit)

    return best_fit, best_pos, curve


def AO(func, lb, ub, dim, max_fes, pop=30):
    """Aquila Optimizer."""
    pos  = _init_pop(pop, dim, lb, ub)
    fits = np.array([func(pos[i]) for i in range(pop)])
    fes  = pop
    best_idx = np.argmin(fits)
    best_pos = pos[best_idx].copy(); best_fit = fits[best_idx]
    curve = [best_fit]

    def levy(d, beta=1.5):
        sigma = (math.gamma(1+beta)*np.sin(np.pi*beta/2) /
                 (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = np.random.randn(d)*sigma; v = np.random.randn(d)
        return u / np.abs(v)**(1/beta)

    t = 0; T = max_fes // pop
    while fes < max_fes:
        t += 1
        alpha_ctrl = 0.1; delta_ctrl = 0.1
        for i in range(pop):
            r1, r2   = np.random.rand(), np.random.rand()
            rand_pos = np.random.uniform(lb, ub, dim)
            mean_pos = np.mean(pos, axis=0)
            if t/T <= 2/3:         # Exploration
                if r1 < 0.5:
                    pos[i] = best_pos*(1 - t/T) + (mean_pos - best_pos)*np.random.rand()
                else:
                    lev    = levy(dim)
                    pos[i] = best_pos + alpha_ctrl*lev*(best_pos - pos[np.random.randint(pop)])
            else:                  # Exploitation
                if r2 < 0.5:
                    pos[i] = (best_pos - mean_pos)*alpha_ctrl - np.random.rand() + rand_pos*delta_ctrl
                else:
                    pos[i] = best_pos + np.random.rand()*(mean_pos - pos[i])
            pos[i] = _clip(pos[i], lb, ub)
        fits = np.array([func(pos[i]) for i in range(pop)])
        fes += pop
        best_idx = np.argmin(fits)
        if fits[best_idx] < best_fit:
            best_fit = fits[best_idx]; best_pos = pos[best_idx].copy()
        curve.append(best_fit)

    return best_fit, best_pos, curve


def ARO(func, lb, ub, dim, max_fes, pop=30):
    """Artificial Rabbit Optimisation – base algorithm GBFAO is derived from."""
    pos  = _init_pop(pop, dim, lb, ub)
    fits = np.array([func(pos[i]) for i in range(pop)], dtype=float)
    fes  = pop

    best_idx = int(np.argmin(fits))
    best_pos = pos[best_idx].copy()
    best_fit = float(fits[best_idx])
    curve    = [best_fit]
    max_iter = max_fes // pop

    for t in range(1, max_iter + 1):
        if fes >= max_fes:
            break
        A     = 4 * (1 - t / max_iter) * np.log(1 / (np.random.rand() + 1e-10))
        theta = 2 * np.pi * np.random.rand()

        for i in range(pop):
            r = np.random.rand()
            if r < 0.5:  # Random walk
                rand_idx = np.random.randint(pop)
                L        = np.random.rand(dim)
                new_pos  = pos[i] + A * L * (pos[rand_idx] - pos[i])
            else:        # Detour foraging toward best
                d       = abs(best_pos - pos[i])
                B       = np.random.rand(dim)
                new_pos = best_pos + d * np.exp(B * np.cos(theta)) * np.cos(2 * np.pi * B)

            new_pos = _clip(new_pos, lb, ub)
            f_new   = func(new_pos); fes += 1
            if f_new < fits[i]:
                pos[i]  = new_pos; fits[i] = f_new
                if f_new < best_fit:
                    best_fit = f_new; best_pos = new_pos.copy()

            if fes >= max_fes:
                break

        curve.append(best_fit)

    return best_fit, best_pos, curve


COMPETITORS = {
    "ARO": ARO,
    "PSO": PSO,
    "GWO": GWO,
    "WOA": WOA,
    "DE" : DE,
    "SMA": SMA,
    "HHO": HHO,
    "AO" : AO,
}
