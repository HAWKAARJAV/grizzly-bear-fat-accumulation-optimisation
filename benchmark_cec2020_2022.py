"""
=============================================================
 CEC-2020 Benchmark Functions  (10 functions, G1–G10)
 CEC-2022 Benchmark Functions  (12 functions, H1–H12)
 ─────────────────────────────────────────────────────────
 CEC-2020: Yue et al. "Problem Definitions and Evaluation
   Criteria for the CEC 2020 Special Session on Single
   Objective Bound Constrained Numerical Optimization"
   Recommended dim = 5 or 10

 CEC-2022: Abhishek et al. "Problem Definitions and
   Evaluation Criteria for the CEC 2022 Special Session"
   Recommended dim = 10 or 20

 Both suites use shifted + rotated landscapes.
 Here we use dim-consistent shifted versions seeded for
 reproducibility (full rotation matrices require the
 official .mat files — these are high-fidelity proxies).
=============================================================
"""

import numpy as np

# ── Reproducible shift vectors ─────────────────────────────────────────────
def _s(dim, seed):
    rng = np.random.default_rng(seed + 1000)
    return rng.uniform(-100, 100, dim)

def _clip(x, lb=-100, ub=100):
    return np.clip(x, lb, ub)

# ══════════════════════════════════════════════════════════════════════════════
# CEC-2020  (G1 – G10)   Bounds: [-100, 100]^D
# Bias offsets follow the convention: 100 * function_number
# ══════════════════════════════════════════════════════════════════════════════
BIAS_2020 = [100 * (i + 1) for i in range(10)]

def G1(x):
    """CEC2020-F1: Shifted Bent Cigar"""
    z = x - _s(len(x), 101)
    return float(z[0]**2 + 1e6 * np.sum(z[1:]**2)) + BIAS_2020[0]

def G2(x):
    """CEC2020-F2: Shifted Sum of Different Powers"""
    z = x - _s(len(x), 102)
    D = len(z)
    i = np.arange(1, D + 1)
    return float(np.sum(np.abs(z) ** (i + 1))) + BIAS_2020[1]

def G3(x):
    """CEC2020-F3: Shifted Zakharov"""
    z = x - _s(len(x), 103)
    i = np.arange(1, len(z) + 1)
    t = np.sum(0.5 * i * z)
    return float(np.sum(z**2) + t**2 + t**4) + BIAS_2020[2]

def G4(x):
    """CEC2020-F4: Shifted Rosenbrock"""
    z = x - _s(len(x), 104) + 1
    return float(np.sum(100 * (z[1:] - z[:-1]**2)**2 + (z[:-1] - 1)**2)) + BIAS_2020[3]

def G5(x):
    """CEC2020-F5: Shifted Rastrigin"""
    z = x - _s(len(x), 105)
    return float(np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)) + BIAS_2020[4]

def G6(x):
    """CEC2020-F6: Shifted Expanded Scaffer F6"""
    z = x - _s(len(x), 106)
    def sc(a, b):
        t = a**2 + b**2
        return 0.5 + (np.sin(np.sqrt(t))**2 - 0.5) / (1 + 0.001 * t)**2
    val = sum(sc(z[i], z[(i + 1) % len(z)]) for i in range(len(z)))
    return float(val) + BIAS_2020[5]

def G7(x):
    """CEC2020-F7: Shifted Lunacek bi-Rastrigin"""
    z = x - _s(len(x), 107)
    D = len(z); mu0 = 2.5; s = 1 - 1 / (2 * np.sqrt(D + 20) - 8.2)
    mu1 = -np.sqrt((mu0**2 - 1) / s)
    z1 = 2 * np.sign(z) * z - mu0
    z2 = 2 * np.sign(z) * z - mu1
    sphere1 = np.sum((z1 - mu0)**2)
    sphere2 = D + s * np.sum((z2 - mu1)**2)
    ras = 10 * (D - np.sum(np.cos(2 * np.pi * z1)))
    return float(min(sphere1, sphere2) + ras) + BIAS_2020[6]

def G8(x):
    """CEC2020-F8: Shifted Non-Continuous Rastrigin"""
    z = x - _s(len(x), 108)
    y = np.where(np.abs(z) < 0.5, z, np.round(2 * z) / 2)
    return float(np.sum(y**2 - 10 * np.cos(2 * np.pi * y) + 10)) + BIAS_2020[7]

def G9(x):
    """CEC2020-F9: Hybrid (Bent Cigar + Rastrigin + Griewank)"""
    z = x - _s(len(x), 109)
    D = len(z)
    d1, d2 = max(1, D // 3), max(1, D // 3)
    d3 = D - d1 - d2
    z1, z2, z3 = z[:d1], z[d1:d1+d2], z[d1+d2:]
    f1 = z1[0]**2 + 1e6 * np.sum(z1[1:]**2) if len(z1) > 1 else z1[0]**2
    f2 = np.sum(z2**2 - 10 * np.cos(2 * np.pi * z2) + 10) if len(z2) > 0 else 0
    i  = np.arange(1, len(z3) + 1)
    f3 = np.sum(z3**2) / 4000 - np.prod(np.cos(z3 / np.sqrt(i))) + 1 if len(z3) > 0 else 0
    return float(f1 + f2 + f3) + BIAS_2020[8]

def G10(x):
    """CEC2020-F10: Composition (Rastrigin + Griewank + Schwefel + Rosenbrock)"""
    z = x - _s(len(x), 110)
    D = len(z); i = np.arange(1, D + 1)
    sigmas = [10, 20, 30, 40]
    weights_raw = np.array([np.exp(-np.sum(z**2) / (2 * D * s**2)) for s in sigmas])
    if weights_raw.sum() < 1e-300:
        weights = np.ones(4) / 4
    else:
        weights = weights_raw / weights_raw.sum()
    # sub-functions
    ras = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
    gri = np.sum(z**2) / 4000 - np.prod(np.cos(z / np.sqrt(i))) + 1
    sch = -np.sum(z * np.sin(np.sqrt(np.abs(z))))
    ros = np.sum(100 * (z[1:] - z[:-1]**2)**2 + (z[:-1] - 1)**2)
    val = weights[0] * ras + weights[1] * gri + weights[2] * sch + weights[3] * ros
    return float(val) + BIAS_2020[9]

# ── CEC-2020 Registry ─────────────────────────────────────────────────────────
FUNCTIONS_2020 = [G1, G2, G3, G4, G5, G6, G7, G8, G9, G10]
FUNC_NAMES_2020 = [f"G{i+1}" for i in range(10)]
BOUNDS_2020 = (-100.0, 100.0)
NUM_FUNCS_2020 = 10


# ══════════════════════════════════════════════════════════════════════════════
# CEC-2022  (H1 – H12)   Bounds: [-100, 100]^D
# Bias offsets: 300 * function_number
# ══════════════════════════════════════════════════════════════════════════════
BIAS_2022 = [300 * (i + 1) for i in range(12)]

def H1(x):
    """CEC2022-F1: Shifted Schwefel 1.2"""
    z = x - _s(len(x), 201)
    return float(sum(np.sum(z[:i+1])**2 for i in range(len(z)))) + BIAS_2022[0]

def H2(x):
    """CEC2022-F2: Shifted High-Conditioned Elliptic"""
    z = x - _s(len(x), 202)
    D = len(z); i = np.arange(1, D + 1)
    return float(np.sum((1e6)**((i-1)/(max(D-1,1))) * z**2)) + BIAS_2022[1]

def H3(x):
    """CEC2022-F3: Shifted Bent Cigar + Discus"""
    z = x - _s(len(x), 203)
    D = len(z)
    if D == 1:
        return float(1e6 * z[0]**2) + BIAS_2022[2]
    return float(1e6 * z[0]**2 + np.sum(z[1:]**2)) + BIAS_2022[2]

def H4(x):
    """CEC2022-F4: Shifted Rosenbrock"""
    z = x - _s(len(x), 204) + 1
    return float(np.sum(100*(z[1:] - z[:-1]**2)**2 + (z[:-1] - 1)**2)) + BIAS_2022[3]

def H5(x):
    """CEC2022-F5: Shifted Ackley"""
    z = x - _s(len(x), 205); D = len(z)
    a = -20 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / D))
    b = -np.exp(np.sum(np.cos(2 * np.pi * z)) / D)
    return float(a + b + 20 + np.e) + BIAS_2022[4]

def H6(x):
    """CEC2022-F6: Shifted Weierstrass"""
    z = x - _s(len(x), 206)
    a, b, kmax = 0.5, 3.0, 20
    k = np.arange(kmax + 1)
    s = sum(np.sum(a**k * np.cos(2*np.pi*b**k*(zi+0.5))) for zi in z)
    s -= len(z) * np.sum(a**k * np.cos(np.pi * b**k))
    return float(s) + BIAS_2022[5]

def H7(x):
    """CEC2022-F7: Shifted Griewank"""
    z = x - _s(len(x), 207); D = len(z)
    i = np.arange(1, D + 1)
    return float(np.sum(z**2)/4000 - np.prod(np.cos(z/np.sqrt(i))) + 1) + BIAS_2022[6]

def H8(x):
    """CEC2022-F8: Shifted Rastrigin"""
    z = x - _s(len(x), 208)
    return float(np.sum(z**2 - 10*np.cos(2*np.pi*z) + 10)) + BIAS_2022[7]

def H9(x):
    """CEC2022-F9: Shifted Levy"""
    z = x - _s(len(x), 209)
    w = 1 + (z - 1) / 4
    t1 = np.sin(np.pi * w[0])**2
    t2 = np.sum((w[:-1]-1)**2 * (1 + 10*np.sin(np.pi*w[:-1]+1)**2))
    t3 = (w[-1]-1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
    return float(t1 + t2 + t3) + BIAS_2022[8]

def H10(x):
    """CEC2022-F10: Hybrid (Schwefel + Rastrigin + Elliptic)"""
    z = x - _s(len(x), 210)
    D = len(z)
    d1 = max(1, int(0.3 * D))
    d2 = max(1, int(0.3 * D))
    z1, z2, z3 = z[:d1], z[d1:d1+d2], z[d1+d2:]
    f1 = (np.sum(np.abs(z1)) + np.prod(np.abs(z1))) if len(z1) > 0 else 0
    f2 = (np.sum(z2**2 - 10*np.cos(2*np.pi*z2) + 10)) if len(z2) > 0 else 0
    Dz3 = max(len(z3), 1)
    i3 = np.arange(1, Dz3+1)
    f3 = float(np.sum((1e6)**((i3-1)/max(Dz3-1,1)) * z3**2)) if len(z3) > 0 else 0
    return float(f1 + f2 + f3) + BIAS_2022[9]

def H11(x):
    """CEC2022-F11: Hybrid (Bent Cigar + Rosenbrock + Ackley)"""
    z = x - _s(len(x), 211)
    D = len(z)
    d1 = max(1, int(0.4 * D))
    d2 = max(1, int(0.4 * D))
    z1, z2, z3 = z[:d1], z[d1:d1+d2], z[d1+d2:]
    f1 = (z1[0]**2 + 1e6*np.sum(z1[1:]**2)) if len(z1) > 1 else (z1[0]**2 if len(z1) > 0 else 0)
    z2r = z2 + 1
    f2 = float(np.sum(100*(z2r[1:]-z2r[:-1]**2)**2 + (z2r[:-1]-1)**2)) if len(z2) > 1 else 0
    Dz3 = max(len(z3), 1)
    a3 = -20*np.exp(-0.2*np.sqrt(np.sum(z3**2)/Dz3)) if len(z3) > 0 else -20
    b3 = -np.exp(np.sum(np.cos(2*np.pi*z3))/Dz3) if len(z3) > 0 else -1
    f3 = a3 + b3 + 20 + np.e
    return float(f1 + f2 + f3) + BIAS_2022[10]

def H12(x):
    """CEC2022-F12: Composition (Rastrigin+Ackley+Schwefel+Rosenbrock+Elliptic+Griewank)"""
    z = x - _s(len(x), 212)
    D = len(z); i = np.arange(1, D + 1)
    sigmas = [10, 20, 20, 30, 30, 40]
    weights_raw = np.array([np.exp(-np.sum(z**2)/(2*D*s**2)) for s in sigmas])
    weights = weights_raw / (weights_raw.sum() + 1e-300)
    # 6 sub-functions
    ras = np.sum(z**2 - 10*np.cos(2*np.pi*z) + 10)
    ack = -20*np.exp(-0.2*np.sqrt(np.sum(z**2)/D)) - np.exp(np.sum(np.cos(2*np.pi*z))/D) + 20 + np.e
    sch = np.sum(np.abs(z)) + np.prod(np.abs(z))
    ros = np.sum(100*(z[1:]-z[:-1]**2)**2 + (z[:-1]-1)**2) if D > 1 else 0
    ell = float(np.sum((1e6)**((i-1)/max(D-1,1)) * z**2))
    gri = np.sum(z**2)/4000 - np.prod(np.cos(z/np.sqrt(i))) + 1
    fns = [ras, ack, sch, ros, ell, gri]
    return float(sum(w*f for w, f in zip(weights, fns))) + BIAS_2022[11]

# ── CEC-2022 Registry ─────────────────────────────────────────────────────────
FUNCTIONS_2022 = [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12]
FUNC_NAMES_2022 = [f"H{i+1}" for i in range(12)]
BOUNDS_2022 = (-100.0, 100.0)
NUM_FUNCS_2022 = 12


# ── Convenience getter ────────────────────────────────────────────────────────
def get_cec2020(fid):
    """1-indexed. Returns (callable, name, lb, ub)."""
    return FUNCTIONS_2020[fid-1], FUNC_NAMES_2020[fid-1], *BOUNDS_2020

def get_cec2022(fid):
    """1-indexed. Returns (callable, name, lb, ub)."""
    return FUNCTIONS_2022[fid-1], FUNC_NAMES_2022[fid-1], *BOUNDS_2022


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    dim = 10
    x = np.zeros(dim)
    print("CEC-2020 (dim=10):")
    for f, n, lb, ub in [get_cec2020(i) for i in range(1, 11)]:
        print(f"  {n}: {f(x):.4e}")
    print("\nCEC-2022 (dim=10):")
    for f, n, lb, ub in [get_cec2022(i) for i in range(1, 13)]:
        print(f"  {n}: {f(x):.4e}")
