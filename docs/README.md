# 🐻 Grizzly Bear Fat Accumulation Optimisation (GBFAO)

> **Assignment**: Nature-Inspired Optimisation — Modified ARO with CEC Benchmarking  
> **Base Algorithm**: Artificial Rabbit Optimisation (ARO)  
> **Proposed Algorithm**: GBFAO — a biologically inspired metaheuristic

---

## 📋 Table of Contents

1. [Biological Inspiration](#biological-inspiration)  
2. [Algorithm Description](#algorithm-description)  
3. [Four Progressive Versions](#four-progressive-versions)  
4. [Project Structure](#project-structure)  
5. [Installation & Quick Start](#installation--quick-start)  
6. [Experimental Settings](#experimental-settings)  
7. [Benchmark Suites](#benchmark-suites)  
8. [Engineering Problems](#engineering-problems)  
9. [Competitor Algorithms](#competitor-algorithms)  
10. [Results Interpretation](#results-interpretation)  
11. [Mathematical Formulation](#mathematical-formulation)  
12. [References](#references)

---

## 🦁 Biological Inspiration

Grizzly bears (*Ursus arctos horribilis*) exhibit three distinct seasonal behaviours that map directly onto the **exploration–exploitation** balance in optimisation:

```
Spring / Summer ──► FORAGING        = Global Exploration
Autumn          ──► HYPERPHAGIA     = Balanced Search  
Winter          ──► HIBERNATION     = Local Exploitation
```

| Season      | Bear Behaviour              | Optimisation Analogy        |
|-------------|-----------------------------|-----------------------------|
| Spring/Summer | Wide-area food search     | Exploration of search space |
| Autumn      | Intense fat accumulation    | Exploitation near optima    |
| Winter      | Sleep in den (fixed point)  | Fine-tuning best solution   |

The **fat accumulation ratio** `α(t) = t / T_max` is the central control variable.  
As `α` increases from 0 → 1, the algorithm naturally transitions from exploration to exploitation — mirroring a bear preparing for hibernation.

---

## 🧠 Algorithm Description

### Core GBFAO Phases

#### Phase 1 — Foraging (α < 1/3)
Bears explore widely, attracted toward the best known food source while also following random conspecifics:

```
X_new = X + r₁ · (X_best − X) + r₂ · (X_rand1 − X_rand2)
```

#### Phase 2 — Hyperphagia (1/3 ≤ α < 2/3)
Bears intensify feeding close to the richest source. The step decays exponentially with fat level:

```
X_new = X_best + r · exp(−α) · (X_best − X)
```

#### Phase 3 — Hibernation (α ≥ 2/3)
Bears enter their den. Only small local perturbations occur:

```
σ  = (1 − α) · (ub − lb)
X_new = X_best + N(0, σ²)
```

### Flowchart

```
┌──────────────────────────────────────────────────────┐
│                    GBFAO START                       │
│  Initialise population X (+ OBL in v3/v4)           │
│  Evaluate fitness f(X)                               │
│  Identify X_best                                     │
└────────────────────┬─────────────────────────────────┘
                     │
           ┌─────────▼──────────┐
           │  t = 1 to T_max    │◄──────────────────┐
           │  α ← t / T_max     │                   │
           └─────────┬──────────┘                   │
                     │                               │
          ┌──────────▼──────────────┐                │
          │ α < 1/3 ?               │                │
          │  YES → FORAGING         │                │
          │  NO  → α < 2/3 ?        │                │
          │    YES → HYPERPHAGIA    │                │
          │    NO  → HIBERNATION    │                │
          └──────────┬──────────────┘                │
                     │                               │
          ┌──────────▼──────────────┐                │
          │ Boundary handling       │                │
          │ Greedy selection        │                │
          │ Update X_best           │                │
          │ [OBJ in v3/v4]          │────────────────┘
          └─────────────────────────┘
                     │
          ┌──────────▼──────────────┐
          │  FES ≥ max_fes ?        │
          │  YES → Return X_best    │
          └─────────────────────────┘
```

---

## 🔢 Four Progressive Versions

Each version achieves **better results with fewer iterations** than the previous:

| Version  | Budget (FEs) | Key Enhancements                              | Iterations |
|----------|-------------|-----------------------------------------------|------------|
| `v1` | 60 000 | Basic 3-phase GBFAO                               | 1 200      |
| `v2` | 45 000 | + **Lévy-flight** foraging (long-range jumps)     | 900        |
| `v3` | 30 000 | + **Opposition-Based Learning** (OBL) init + OBJ | 600        |
| `v4` ★ | 20 000 | + **Adaptive α**, Elite Archive, Cauchy mutation, Dim mask | 400 |

### Why each version is better

**v2 (Lévy Flight)**  
Lévy flights produce occasional very long steps — perfect for escaping local optima. Real bears do not walk in straight lines; they follow scale-free foraging patterns (Power law distribution).

**v3 (OBL)**  
For any candidate solution `X`, its *opposite* `X̃ = lb + ub − X` may be closer to the optimum. By evaluating both and keeping the better one at initialisation, and periodically "jumping" the worst agents to their opposites, the algorithm dramatically improves diversity.

**v4 (Elite-Den)**  
Five synergistic enhancements:
- `E1` Sigmoid schedule: longer exploration, sharper exploitation
- `E2` Elite archive: foraging guided by top-k bears, not just global best
- `E3` Cauchy mutation: heavy-tailed den disturbance prevents stagnation
- `E4` Dimensional mask: only a subset of dimensions updated per step
- `E5` Vectorised evaluation: NumPy-level speed

---

## 📁 Project Structure

```
gbfao_project/
│
├── benchmark_functions.py    ← 30 CEC-2014 style functions (F1–F30)
├── competitors.py            ← PSO, GWO, WOA, DE, SMA, HHO, AO
├── engineering_problems.py   ← 5 engineering design problems
│
├── gbfao_v1.py               ← Basic GBFAO         (60 000 FEs)
├── gbfao_v2.py               ← + Lévy Flight        (45 000 FEs)
├── gbfao_v3.py               ← + OBL               (30 000 FEs)
├── gbfao_v4.py  ★            ← Full Enhanced        (20 000 FEs)
│
├── run_experiments.py        ← Main orchestrator (CLI + full pipeline)
├── statistical_test.py       ← Wilcoxon rank-sum test
├── plot_convergence.py       ← All visualisations
│
└── results/                  ← Generated outputs (auto-created)
    ├── results_table.csv
    ├── wilcoxon_results.csv
    ├── rank_heatmap.pdf
    ├── engineering_results.pdf
    ├── version_comparison.pdf
    ├── convergence/
    └── boxplots/
```

---

## 🚀 Installation & Quick Start

### Prerequisites

```bash
pip install numpy scipy pandas matplotlib
```

### Demo Run (< 60 seconds)

```bash
python run_experiments.py --demo
```

This runs 5 functions × 5 runs × 6 000 FEs — useful to verify the pipeline.

### Full Assignment Run (Assignment-compliant)

```bash
python run_experiments.py --max_fes 60000 --runs 50 --dim 30
```

### Run individual GBFAO versions

```python
from benchmark_functions import F7
from gbfao_v4 import GBFAO_v4

best_fit, best_pos, curve = GBFAO_v4(F7, lb=-100, ub=100, dim=30, max_fes=60000)
print(f"Best fitness: {best_fit:.6e}")
```

### Test a single competitor

```python
from competitors import GWO
from benchmark_functions import F6

best, pos, curve = GWO(F6, -100, 100, 30, max_fes=60000)
```

---

## ⚙️ Experimental Settings

| Parameter          | Value            |
|--------------------|------------------|
| Max FEs per run    | **60 000**       |
| Independent runs   | **50**           |
| Population size    | 50               |
| Dimension          | 30               |
| Search bounds      | [−100, 100]      |
| Statistical test   | Wilcoxon rank-sum (α = 0.05) |
| Metrics            | Mean, Std Dev, Rank |

---

## 📊 Benchmark Suites

### CEC-2014 Style Functions (F1–F30)

| Functions | Type                        |
|-----------|-----------------------------|
| F1–F5     | Unimodal (basic)            |
| F6–F10    | Multimodal (simple)         |
| F11–F15   | Multimodal (expanded)       |
| F16–F20   | Multimodal with structure   |
| F21–F25   | Hybrid compositions         |
| F26–F30   | High-difficulty             |

> **Note**: The implementations here are shifted/biased variants equivalent to CEC-2014 in difficulty. For exact CEC-2014/2017/2020/2022 compliance, load the official rotation matrices from the `.mat` files provided by the CEC organisers and pass them to each function.

---

## 🔧 Engineering Problems

| #  | Problem                   | Variables | Known Best |
|----|---------------------------|-----------|------------|
| 1  | Pressure Vessel Design    | 4         | 6059.714   |
| 2  | Tension/Compression Spring| 3         | 0.012665   |
| 3  | Welded Beam Design        | 4         | 1.7248     |
| 4  | Speed Reducer             | 7         | 2994.471   |
| 5  | Three-Bar Truss           | 2         | 263.8958   |

Constraints are handled via the **quadratic exterior penalty method**:

```
F_pen(x) = f(x) + λ · Σ max(0, gᵢ(x))²,   λ = 10⁶
```

---

## 🤖 Competitor Algorithms

| Algorithm | Year | Reference |
|-----------|------|-----------|
| PSO — Particle Swarm Optimisation  | 1995 | Kennedy & Eberhart |
| GWO — Grey Wolf Optimizer          | 2014 | Mirjalili et al.   |
| WOA — Whale Optimisation Algorithm | 2016 | Mirjalili & Lewis  |
| DE  — Differential Evolution       | 1997 | Storn & Price      |
| SMA — Slime Mould Algorithm        | 2020 | Li et al.          |
| HHO — Harris Hawks Optimisation    | 2019 | Heidari et al.     |
| AO  — Aquila Optimizer             | 2021 | Abualigah et al.   |

---

## 📈 Results Interpretation

### Reading the Mean/Std/Rank Table

- **Mean**: Average best fitness over 50 runs (lower = better for minimisation)
- **Std**: Standard deviation — lower means more consistent
- **Rank**: 1 = best performing algorithm on that function

### Wilcoxon Test Symbols

| Symbol | Meaning |
|--------|---------|
| `+`    | GBFAO-v4 significantly **better** (p < 0.05) |
| `−`    | GBFAO-v4 significantly **worse**  (p < 0.05) |
| `≈`    | No statistically significant difference       |

---

## 📐 Mathematical Formulation

### Lévy Flight (v2, v3, v4)

```
s = u / |v|^(1/β)

u ~ N(0, σᵤ²),   v ~ N(0, 1)

σᵤ = [Γ(1+β)·sin(πβ/2) / (Γ((1+β)/2)·β·2^((β-1)/2))]^(1/β)

β = 1.5  (standard Mantegna's algorithm)
```

### Opposition-Based Learning (v3, v4)

```
X̃ⱼ = lbⱼ + ubⱼ − Xⱼ,   j = 1 … dim

Select X* = argmin{f(X), f(X̃)}
```

### Sigmoid Fat-Accumulation Schedule (v4)

```
α(t) = 1 / (1 + exp(−k·(t/T − 0.5))),   k = 10
```

This keeps α small (exploration) for the first half of iterations and
rapidly switches to exploitation in the second half — unlike the linear
schedule of v1–v3.

### Cauchy Mutation (v4)

```
X_mut = X_best + Cauchy(0, γ)

γ = max((1−α)·(ub−lb)·0.3,  1e−8)
```

Cauchy distribution has undefined variance — its heavy tails generate
occasional very large perturbations, helping escape deep local optima.

---

## 🔖 References

1. Wang, L., et al. (2022). *Artificial rabbits optimisation: A new bio-inspired meta-heuristic algorithm*. Engineering Applications of Artificial Intelligence.
2. Mirjalili, S., et al. (2014). *Grey Wolf Optimizer*. Advances in Engineering Software.
3. Mirjalili, S., & Lewis, A. (2016). *The Whale Optimization Algorithm*. Advances in Engineering Software.
4. Storn, R., & Price, K. (1997). *Differential Evolution*. Journal of Global Optimisation.
5. Li, S., et al. (2020). *Slime mould algorithm: A new method for stochastic optimisation*. Future Generation Computer Systems.
6. Heidari, A. A., et al. (2019). *Harris hawks optimization: Algorithm and applications*. Future Generation Computer Systems.
7. Abualigah, L., et al. (2021). *Aquila Optimizer*. Computers & Industrial Engineering.
8. Liang, J. J., et al. (2013). *Problem Definitions and Evaluation Criteria for the CEC 2014 Special Session*. RMIT University.
9. Rahnamayan, S., et al. (2008). *Opposition-Based Differential Evolution*. IEEE Transactions on Evolutionary Computation.
10. Mantegna, R. N. (1994). *Fast, accurate algorithm for numerical simulation of Lévy stable stochastic processes*. Physical Review E.
