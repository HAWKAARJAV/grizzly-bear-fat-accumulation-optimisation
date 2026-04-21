# 30 benchmark functions (CEC-2014 style): shifted unimodal, multimodal, and hybrid landscapes.
import numpy as np

# ─── Fixed shift vectors (one per function, seeded for reproducibility) ───────
def _shift(dim, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-100, 100, dim)

# Bias offsets (CEC-2014 convention)
BIAS = [100*(i+1) for i in range(30)]

# ─── F1–F5  :  Unimodal functions ─────────────────────────────────────────────

def F1(x):
    """Shifted Sphere"""
    z = x - _shift(len(x), 1)
    return float(np.sum(z**2)) + BIAS[0]

def F2(x):
    """Shifted Schwefel 1.2"""
    z = x - _shift(len(x), 2)
    return float(sum(np.sum(z[:i+1])**2 for i in range(len(z)))) + BIAS[1]

def F3(x):
    """Shifted High-Conditioned Elliptic"""
    z = x - _shift(len(x), 3)
    D = len(z); i = np.arange(1, D+1)
    return float(np.sum((1e6)**((i-1)/(D-1)) * z**2)) + BIAS[2]

def F4(x):
    """Shifted Schwefel 1.2 with Noise"""
    z = x - _shift(len(x), 4)
    val = sum(np.sum(z[:i+1])**2 for i in range(len(z)))
    return float(val * (1 + 0.4*abs(np.random.randn()))) + BIAS[3]

def F5(x):
    """Shifted Rosenbrock"""
    z = x - _shift(len(x), 5) + 1
    return float(np.sum(100*(z[1:]-z[:-1]**2)**2 + (z[:-1]-1)**2)) + BIAS[4]

# ─── F6–F10 :  Multimodal functions ───────────────────────────────────────────

def F6(x):
    """Shifted Ackley"""
    z = x - _shift(len(x), 6); D = len(z)
    a = -20*np.exp(-0.2*np.sqrt(np.sum(z**2)/D))
    b = -np.exp(np.sum(np.cos(2*np.pi*z))/D)
    return float(a + b + 20 + np.e) + BIAS[5]

def F7(x):
    """Shifted Rastrigin"""
    z = x - _shift(len(x), 7)
    return float(np.sum(z**2 - 10*np.cos(2*np.pi*z) + 10)) + BIAS[6]

def F8(x):
    """Shifted Griewank"""
    z = x - _shift(len(x), 8); D = len(z)
    i = np.arange(1, D+1)
    return float(np.sum(z**2)/4000 - np.prod(np.cos(z/np.sqrt(i))) + 1) + BIAS[7]

def F9(x):
    """Shifted Weierstrass"""
    z = x - _shift(len(x), 9)
    a, b, kmax = 0.5, 3.0, 20; k = np.arange(kmax+1)
    s = sum(np.sum(a**k * np.cos(2*np.pi*b**k*(zi+0.5))) for zi in z)
    s -= len(z)*np.sum(a**k * np.cos(np.pi*b**k))
    return float(s) + BIAS[8]

def F10(x):
    """Shifted Schwefel 2.22"""
    z = x - _shift(len(x), 10)
    return float(np.sum(np.abs(z)) + np.prod(np.abs(z))) + BIAS[9]

# ─── F11–F15 : Expanded / Hybrid unimodal ─────────────────────────────────────

def F11(x):
    """Shifted Schwefel 2.21"""
    z = x - _shift(len(x), 11)
    return float(np.max(np.abs(z))) + BIAS[10]

def F12(x):
    """Shifted Expanded Scaffer F6"""
    z = x - _shift(len(x), 12)
    def scaffer_pair(a, b):
        t = a**2 + b**2
        return 0.5 + (np.sin(np.sqrt(t))**2 - 0.5) / (1 + 0.001*t)**2
    val = sum(scaffer_pair(z[i], z[(i+1)%len(z)]) for i in range(len(z)))
    return float(val) + BIAS[11]

def F13(x):
    """Shifted Expanded Griewank + Rosenbrock"""
    z = x - _shift(len(x), 13) + 1; D = len(z)
    ros = np.sum(100*(z[1:]-z[:-1]**2)**2 + (z[:-1]-1)**2)
    gri = np.sum(z**2)/4000 - np.prod(np.cos(z/np.sqrt(np.arange(1,D+1)))) + 1
    return float(ros + gri) + BIAS[12]

def F14(x):
    """Shifted Styblinski-Tang"""
    z = x - _shift(len(x), 14)
    return float(np.sum(z**4 - 16*z**2 + 5*z)/2) + BIAS[13]

def F15(x):
    """Shifted Zakharov"""
    z = x - _shift(len(x), 15); i = np.arange(1, len(z)+1)
    t = np.sum(0.5*i*z)
    return float(np.sum(z**2) + t**2 + t**4) + BIAS[14]

# ─── F16–F20 : Multimodal with structure ──────────────────────────────────────

def F16(x):
    """Shifted Levy"""
    z = x - _shift(len(x), 16)
    w = 1 + (z-1)/4
    t1 = np.sin(np.pi*w[0])**2
    t2 = np.sum((w[:-1]-1)**2 * (1+10*np.sin(np.pi*w[:-1]+1)**2))
    t3 = (w[-1]-1)**2 * (1+np.sin(2*np.pi*w[-1])**2)
    return float(t1 + t2 + t3) + BIAS[15]

def F17(x):
    """Shifted Schwefel 2.26 (penalty)"""
    z = x - _shift(len(x), 17)
    penalty = np.sum(np.where(np.abs(z)>500, (np.abs(z)-500)**2, 0))
    return float(-np.sum(z*np.sin(np.sqrt(np.abs(z)))) + penalty) + BIAS[16]

def F18(x):
    """Shifted Alpine"""
    z = x - _shift(len(x), 18)
    return float(np.sum(np.abs(z*np.sin(z) + 0.1*z))) + BIAS[17]

def F19(x):
    """Shifted Quartic with noise"""
    z = x - _shift(len(x), 19)
    i = np.arange(1, len(z)+1)
    return float(np.sum(i*z**4) + np.random.rand()) + BIAS[18]

def F20(x):
    """Shifted Salomon"""
    z = x - _shift(len(x), 20)
    r = np.sqrt(np.sum(z**2))
    return float(1 - np.cos(2*np.pi*r) + 0.1*r) + BIAS[19]

# ─── F21–F25 : Composition functions ──────────────────────────────────────────

def F21(x):
    """Hybrid: Rastrigin + Griewank"""
    z = x - _shift(len(x), 21); D = len(z)
    i = np.arange(1, D+1)
    ras = np.sum(z**2 - 10*np.cos(2*np.pi*z) + 10)
    gri = np.sum(z**2)/4000 - np.prod(np.cos(z/np.sqrt(i))) + 1
    return float(0.5*ras + 0.5*gri) + BIAS[20]

def F22(x):
    """Hybrid: Ackley + Rosenbrock"""
    z = x - _shift(len(x), 22); D = len(z)
    ack = -20*np.exp(-0.2*np.sqrt(np.sum(z**2)/D)) - np.exp(np.sum(np.cos(2*np.pi*z))/D) + 20 + np.e
    ros = np.sum(100*(z[1:]-z[:-1]**2)**2 + (z[:-1]-1)**2)
    return float(0.4*ack + 0.6*ros) + BIAS[21]

def F23(x):
    """Hybrid: Sphere + Schwefel 2.22"""
    z = x - _shift(len(x), 23)
    sp  = np.sum(z**2)
    sch = np.sum(np.abs(z)) + np.prod(np.abs(z))
    return float(0.5*sp + 0.5*sch) + BIAS[22]

def F24(x):
    """Hybrid: Weierstrass + Rastrigin"""
    z = x - _shift(len(x), 24)
    a, b, kmax = 0.5, 3.0, 20; k = np.arange(kmax+1)
    wei = sum(np.sum(a**k * np.cos(2*np.pi*b**k*(zi+0.5))) for zi in z)
    wei -= len(z)*np.sum(a**k * np.cos(np.pi*b**k))
    ras = np.sum(z**2 - 10*np.cos(2*np.pi*z) + 10)
    return float(0.5*wei + 0.5*ras) + BIAS[23]

def F25(x):
    """Hybrid: Levy + Scaffer"""
    z = x - _shift(len(x), 25)
    w = 1 + (z-1)/4
    lev = np.sin(np.pi*w[0])**2 + np.sum((w[:-1]-1)**2*(1+10*np.sin(np.pi*w[:-1]+1)**2)) + (w[-1]-1)**2
    def sc(a,b):
        t=a**2+b**2; return 0.5+(np.sin(np.sqrt(t))**2-0.5)/(1+0.001*t)**2
    scf = sum(sc(z[i],z[(i+1)%len(z)]) for i in range(len(z)))
    return float(0.5*lev + 0.5*scf) + BIAS[24]

# ─── F26–F30 : High-dimensional difficult functions ───────────────────────────

def F26(x):
    """Shifted Penalized 1"""
    z = x - _shift(len(x), 26)
    u = np.where(z>5, 100*(z-5)**4, np.where(z<-5, 100*(-z-5)**4, 0))
    val = (np.pi/len(z))*(10*np.sin(np.pi*(1+(z[0]+1)/4))**2 +
          np.sum(((z[:-1]+1)/4)**2 * (1+10*np.sin(np.pi*(1+(z[1:]+1)/4))**2)) +
          ((z[-1]+1)/4-1)**2) + np.sum(u)
    return float(val) + BIAS[25]

def F27(x):
    """Shifted Penalized 2"""
    z = x - _shift(len(x), 27)
    u = np.where(z>5, 100*(z-5)**4, np.where(z<-5, 100*(-z-5)**4, 0))
    val = 0.1*(np.sin(3*np.pi*z[0])**2 +
               np.sum((z[:-1]-1)**2*(1+np.sin(3*np.pi*z[1:])**2)) +
               (z[-1]-1)**2*(1+np.sin(2*np.pi*z[-1])**2)) + np.sum(u)
    return float(val) + BIAS[26]

def F28(x):
    """Shifted Shubert-like"""
    z = x - _shift(len(x), 28)
    result = 0.0
    for zi in z:
        j = np.arange(1,6)
        result += np.sum(j * np.cos((j+1)*zi + j))
    return float(result) + BIAS[27]

def F29(x):
    """Shifted Dixon-Price"""
    z = x - _shift(len(x), 29)
    i = np.arange(2, len(z)+1)
    return float((z[0]-1)**2 + np.sum(i*(2*z[1:]**2 - z[:-1])**2)) + BIAS[28]

def F30(x):
    """Shifted Powell"""
    z = x - _shift(len(x), 30); D = len(z)
    z = z[:D - D%4] if D%4 != 0 else z  # ensure multiple of 4
    i = np.arange(0, len(z)//4)
    t1 = (z[4*i]   + 10*z[4*i+1])**2
    t2 = 5*(z[4*i+2] - z[4*i+3])**2
    t3 = (z[4*i+1] - 2*z[4*i+2])**4
    t4 = 10*(z[4*i] - z[4*i+3])**4
    return float(np.sum(t1+t2+t3+t4)) + BIAS[29]

# ─── Registry ─────────────────────────────────────────────────────────────────
FUNCTIONS = [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,
             F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,
             F21,F22,F23,F24,F25,F26,F27,F28,F29,F30]

FUNC_NAMES = [f"F{i+1}" for i in range(30)]
BOUNDS     = (-100.0, 100.0)   # search space for all functions
NUM_FUNCS  = 30

def get_function(fid):
    """Return (callable, name, lb, ub).  fid is 1-indexed."""
    return FUNCTIONS[fid-1], FUNC_NAMES[fid-1], BOUNDS[0], BOUNDS[1]
