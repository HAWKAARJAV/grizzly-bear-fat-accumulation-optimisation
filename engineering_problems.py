"""
=============================================================
 Engineering Constrained Optimisation Problems
 ─────────────────────────────────────────────
 1. Pressure Vessel Design (PVD)
 2. Tension/Compression Spring (TCS)
 3. Welded Beam Design (WBD)
 4. Speed Reducer (SR)
 5. Three-Bar Truss (TBT)

 Each problem is wrapped as an unconstrained penalty function:
   F(x) = f(x) + Σ λ·max(0, g_i(x))²

 Interface:  result = problem["func"](x)
             bounds = (problem["lb"], problem["ub"])
             dim    = problem["dim"]
=============================================================
"""

import numpy as np

PENALTY = 1e6   # penalty coefficient

# ══════════════════════════════════════════════════════════════
# 1. Pressure Vessel Design
#    Variables : x = [Ts, Th, R, L]
#    Minimise  : total cost (material + forming + welding)
# ══════════════════════════════════════════════════════════════

def pressure_vessel(x):
    Ts, Th, R, L = x
    # Objective
    cost = (0.6224*Ts*R*L + 1.7781*Th*R**2
            + 3.1661*Ts**2*L + 19.84*Ts**2*R)
    # Constraints
    g1 = -Ts + 0.0193*R
    g2 = -Th + 0.00954*R
    g3 = -np.pi*R**2*L - (4/3)*np.pi*R**3 + 1296000
    g4 = L - 240
    penalty = PENALTY*(max(0,g1)**2 + max(0,g2)**2
                       + max(0,g3)**2 + max(0,g4)**2)
    return cost + penalty

PVD = {
    "name" : "Pressure Vessel Design",
    "func" : pressure_vessel,
    "dim"  : 4,
    "lb"   : np.array([0.0625, 0.0625, 10.0,  10.0]),
    "ub"   : np.array([6.1875, 6.1875, 200.0, 200.0]),
    "known_best": 6059.714,
}


# ══════════════════════════════════════════════════════════════
# 2. Tension/Compression Spring
#    Variables : x = [d, D, N]  (wire diameter, coil diameter, active coils)
#    Minimise  : weight
# ══════════════════════════════════════════════════════════════

def tension_spring(x):
    d, D, N = x
    weight = (N + 2)*D*d**2
    # Constraints
    g1 = 1 - D**3*N / (71785*d**4)
    g2 = (4*D**2 - d*D) / (12566*(D*d**3 - d**4)) + 1/(5108*d**2) - 1
    g3 = 1 - 140.45*d / (D**2*N)
    g4 = (d + D)/1.5 - 1
    penalty = PENALTY*(max(0,g1)**2 + max(0,g2)**2
                       + max(0,g3)**2 + max(0,g4)**2)
    return weight + penalty

TCS = {
    "name" : "Tension/Compression Spring",
    "func" : tension_spring,
    "dim"  : 3,
    "lb"   : np.array([0.05, 0.25, 2.0]),
    "ub"   : np.array([2.00, 1.30, 15.0]),
    "known_best": 0.012665,
}


# ══════════════════════════════════════════════════════════════
# 3. Welded Beam Design
#    Variables : x = [h, l, t, b]
#    Minimise  : fabrication cost
# ══════════════════════════════════════════════════════════════

def welded_beam(x):
    h, l, t, b = x
    P=6000; L=14; E=30e6; G=12e6
    tau_max=13600; sigma_max=30000; delta_max=0.25
    Pc_min=6000

    M    = P*(L + l/2)
    R    = np.sqrt(l**2/4 + ((h+t)/2)**2)
    J    = 2*(np.sqrt(2)*h*l*(l**2/12 + ((h+t)/2)**2))
    tau1 = P/(np.sqrt(2)*h*l)
    tau2 = M*R/J
    tau  = np.sqrt(tau1**2 + 2*tau1*tau2*l/(2*R) + tau2**2)
    sigma= 6*P*L/(b*t**2)
    delta= 4*P*L**3/(E*b*t**3)
    Pc   = 4.013*np.sqrt(E*G*t**2*b**6/36)/L**2 * (1 - t/(2*L)*np.sqrt(E/(4*G)))

    cost = 1.10471*h**2*l + 0.04811*t*b*(14+l)

    g1 = tau - tau_max
    g2 = sigma - sigma_max
    g3 = h - b
    g4 = 0.10471*h**2 + 0.04811*t*b*(14+l) - 5
    g5 = 0.125 - h
    g6 = delta - delta_max
    g7 = Pc_min - Pc

    penalty = PENALTY*sum(max(0,g)**2 for g in [g1,g2,g3,g4,g5,g6,g7])
    return cost + penalty

WBD = {
    "name" : "Welded Beam Design",
    "func" : welded_beam,
    "dim"  : 4,
    "lb"   : np.array([0.1, 0.1, 0.1, 0.1]),
    "ub"   : np.array([2.0, 10.0, 10.0, 2.0]),
    "known_best": 1.7248,
}


# ══════════════════════════════════════════════════════════════
# 4. Speed Reducer
#    Variables : x = [b, m, z, l1, l2, d1, d2]
#    Minimise  : weight of the speed reducer
# ══════════════════════════════════════════════════════════════

def speed_reducer(x):
    b,m,z,l1,l2,d1,d2 = x
    f = (0.7854*b*m**2*(3.3333*z**2 + 14.9334*z - 43.0934)
         - 1.508*b*(d1**2 + d2**2)
         + 7.477*(d1**3 + d2**3)
         + 0.7854*(l1*d1**2 + l2*d2**2))

    g1 = 27/(b*m**2*z) - 1
    g2 = 397.5/(b*m**2*z**2) - 1
    g3 = 1.93*l1**3/(m*z*d1**4) - 1
    g4 = 1.93*l2**3/(m*z*d2**4) - 1
    A1 = np.sqrt((745*l1/(m*z))**2 + 16.9e6)
    g5 = A1/(0.1*d1**3) - 1100
    A2 = np.sqrt((745*l2/(m*z))**2 + 157.5e6)
    g6 = A2/(0.1*d2**3) - 850
    g7 = m*z/40 - 1
    g8 = 5*m/b - 1
    g9 = b/(12*m) - 1
    g10= 1.5*d1 + 1.9 - l1
    g11= 1.1*d2 + 1.9 - l2

    penalty = PENALTY*sum(max(0,g)**2 for g in
                          [g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11])
    return f + penalty

SR = {
    "name" : "Speed Reducer",
    "func" : speed_reducer,
    "dim"  : 7,
    "lb"   : np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0]),
    "ub"   : np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]),
    "known_best": 2994.471,
}


# ══════════════════════════════════════════════════════════════
# 5. Three-Bar Truss
#    Variables : x = [A1, A2]  (cross-sectional areas)
#    Minimise  : weight of truss
# ══════════════════════════════════════════════════════════════

def three_bar_truss(x):
    A1, A2 = x
    P=2; L=100; sigma=2
    sq2 = np.sqrt(2)
    f = L*(2*sq2*A1 + A2)

    g1 = sq2*A1 + A2
    g1 = P/(g1) - sigma if g1 != 0 else PENALTY

    g2_denom = sq2*A1**2 + 2*A1*A2
    g2 = P*A1/g2_denom - sigma if g2_denom != 0 else PENALTY

    g3 = P/(sq2*A2 + A1) - sigma if (sq2*A2 + A1) != 0 else PENALTY

    penalty = PENALTY*(max(0,g1)**2 + max(0,g2)**2 + max(0,g3)**2)
    return f + penalty

TBT = {
    "name" : "Three-Bar Truss",
    "func" : three_bar_truss,
    "dim"  : 2,
    "lb"   : np.array([0.001, 0.001]),
    "ub"   : np.array([1.0,   1.0  ]),
    "known_best": 263.8958,
}


# ─── Registry ─────────────────────────────────────────────────
ENGINEERING_PROBLEMS = [PVD, TCS, WBD, SR, TBT]


# ─── Convenience runner ───────────────────────────────────────
def run_engineering(algorithm_func, problem, max_fes=60_000, runs=30):
    """
    Run an algorithm on a constrained engineering problem.

    Parameters
    ----------
    algorithm_func : callable  with signature (func, lb, ub, dim, max_fes)
    problem        : dict      from ENGINEERING_PROBLEMS
    max_fes        : int
    runs           : int

    Returns
    -------
    dict with keys: mean, std, best, results
    """
    results = []
    for _ in range(runs):
        best_fit, _, _ = algorithm_func(
            problem["func"],
            problem["lb"],
            problem["ub"],
            problem["dim"],
            max_fes=max_fes,
        )
        results.append(best_fit)
    results = np.array(results)
    return {
        "mean"   : float(np.mean(results)),
        "std"    : float(np.std(results)),
        "best"   : float(np.min(results)),
        "results": results,
    }


# ─── Self-test ────────────────────────────────────────────────
if __name__ == "__main__":
    from gbfao_v4 import GBFAO_v4
    for prob in ENGINEERING_PROBLEMS:
        r = run_engineering(GBFAO_v4, prob, max_fes=10_000, runs=5)
        print(f"{prob['name']:35s}  "
              f"Mean={r['mean']:.4e}  Std={r['std']:.4e}  "
              f"Known≈{prob['known_best']:.4f}")
