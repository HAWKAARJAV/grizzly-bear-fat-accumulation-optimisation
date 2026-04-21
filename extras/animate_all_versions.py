"""
animate_all_versions.py
───────────────────────
Live animation of GBFAO v1, v2, v3, v4 running simultaneously on a 2D
Rastrigin landscape.  Each version runs its own population step-by-step
so you can watch the bears move in real-time.

Layout
------
  [v1 landscape] [v2 landscape]
  [v3 landscape] [v4 landscape]
  [────── shared convergence curve ──────]

Run:
    python3 animate_all_versions.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from math import gamma, sin, pi

matplotlib.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "text.color":       "#e6edf3",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "axes.edgecolor":   "#30363d",
    "grid.color":       "#21262d",
    "font.family":      "DejaVu Sans",
})

# ── 2D target function (Rastrigin, dim=2) ─────────────────────────────────────
LB, UB = -5.12, 5.12

def rastrigin2d(x):
    return float(x[0]**2 - 10*np.cos(2*np.pi*x[0]) +
                 x[1]**2 - 10*np.cos(2*np.pi*x[1]) + 20)

# Precompute landscape grid
G = 200
gx = np.linspace(LB, UB, G)
gy = np.linspace(LB, UB, G)
XX, YY = np.meshgrid(gx, gy)
ZZ = np.vectorize(lambda a, b: rastrigin2d(np.array([a, b])))(XX, YY)

# ── Lévy helpers ──────────────────────────────────────────────────────────────
def _levy_vec(dim, beta=1.5):
    su = (gamma(1+beta)*sin(pi*beta/2) /
          (gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u  = np.random.randn(dim)*su
    v  = np.random.randn(dim)
    return u / (np.abs(v)**(1/beta))

def _levy_mat(n, dim, beta=1.5):
    su = (gamma(1+beta)*sin(pi*beta/2) /
          (gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u  = np.random.randn(n, dim)*su
    v  = np.random.randn(n, dim)
    return u / (np.abs(v)**(1/beta))

# ── Per-version step-by-step classes ─────────────────────────────────────────

class GBFAOv1State:
    label    = "GBFAO-v1  Basic"
    color    = "#8338EC"
    max_iter = 300

    def __init__(self, pop=20, dim=2):
        self.pop  = pop; self.dim = dim; self.t = 0
        self.pos  = np.random.uniform(LB, UB, (pop, dim))
        self.fit  = np.array([rastrigin2d(self.pos[i]) for i in range(pop)])
        bi        = np.argmin(self.fit)
        self.best_pos = self.pos[bi].copy()
        self.best_fit = float(self.fit[bi])
        self.curve    = [self.best_fit]
        self.phase    = "Foraging"

    def step(self):
        self.t += 1
        alpha  = self.t / self.max_iter
        if   alpha < 1/3: self.phase = "Foraging"
        elif alpha < 2/3: self.phase = "Hyperphagia"
        else:             self.phase = "Hibernation"

        for i in range(self.pop):
            x = self.pos[i].copy()
            if alpha < 1/3:
                r1  = np.random.rand(self.dim)
                r2  = np.random.rand(self.dim)
                a, b = np.random.randint(self.pop, size=2)
                x_new = x + r1*(self.best_pos - x) + r2*(self.pos[a] - self.pos[b])
            elif alpha < 2/3:
                r     = np.random.rand(self.dim)
                x_new = self.best_pos + r*np.exp(-alpha)*(self.best_pos - x)
            else:
                sigma = max(float(np.mean((1-alpha)*(UB-LB))), 1e-8)
                x_new = self.best_pos + np.random.normal(0, sigma, self.dim)

            x_new = np.clip(x_new, LB, UB)
            f_new = rastrigin2d(x_new)
            if f_new < self.fit[i]:
                self.pos[i] = x_new; self.fit[i] = f_new
                if f_new < self.best_fit:
                    self.best_fit = f_new; self.best_pos = x_new.copy()

        self.curve.append(self.best_fit)
        return self.t >= self.max_iter


class GBFAOv2State:
    label    = "GBFAO-v2  Lévy"
    color    = "#2EC4B6"
    max_iter = 250

    def __init__(self, pop=20, dim=2):
        self.pop  = pop; self.dim = dim; self.t = 0
        self.pos  = np.random.uniform(LB, UB, (pop, dim))
        self.fit  = np.array([rastrigin2d(self.pos[i]) for i in range(pop)])
        bi        = np.argmin(self.fit)
        self.best_pos = self.pos[bi].copy()
        self.best_fit = float(self.fit[bi])
        self.curve    = [self.best_fit]
        self.phase    = "Foraging"

    def step(self):
        self.t += 1
        alpha = self.t / self.max_iter
        if   alpha < 1/3: self.phase = "Foraging"
        elif alpha < 2/3: self.phase = "Hyperphagia"
        else:             self.phase = "Hibernation"

        for i in range(self.pop):
            x   = self.pos[i].copy()
            lev = _levy_vec(self.dim)
            if alpha < 1/3:
                r     = np.random.rand(self.dim)
                a, b  = np.random.randint(self.pop, size=2)
                x_new = x + lev*(self.best_pos - x) + r*(self.pos[a] - self.pos[b])
            elif alpha < 2/3:
                r     = np.random.rand(self.dim)
                x_new = self.best_pos + r*np.exp(-alpha)*np.abs(lev)*(self.best_pos - x)
            else:
                sigma = max(float(np.mean((1-alpha)*(UB-LB)*0.5)), 1e-8)
                x_new = self.best_pos + np.random.normal(0, sigma, self.dim)

            x_new = np.clip(x_new, LB, UB)
            f_new = rastrigin2d(x_new)
            if f_new < self.fit[i]:
                self.pos[i] = x_new; self.fit[i] = f_new
                if f_new < self.best_fit:
                    self.best_fit = f_new; self.best_pos = x_new.copy()

        self.curve.append(self.best_fit)
        return self.t >= self.max_iter


class GBFAOv3State:
    label    = "GBFAO-v3  Lévy+OBL"
    color    = "#FF9F1C"
    max_iter = 200

    def __init__(self, pop=20, dim=2):
        self.pop  = pop; self.dim = dim; self.t = 0
        self.obl_interval = 15; self.obl_rate = 0.30

        self.pos  = np.random.uniform(LB, UB, (pop, dim))
        self.fit  = np.array([rastrigin2d(self.pos[i]) for i in range(pop)])
        # OBL init
        opp  = np.clip(LB + UB - self.pos, LB, UB)
        fo   = np.array([rastrigin2d(opp[i]) for i in range(pop)])
        mask = fo < self.fit
        self.pos  = np.where(mask[:, None], opp, self.pos)
        self.fit  = np.where(mask, fo, self.fit)

        bi        = np.argmin(self.fit)
        self.best_pos = self.pos[bi].copy()
        self.best_fit = float(self.fit[bi])
        self.curve    = [self.best_fit]
        self.phase    = "Foraging"

    def step(self):
        self.t += 1
        alpha = self.t / self.max_iter
        if   alpha < 1/3: self.phase = "Foraging"
        elif alpha < 2/3: self.phase = "Hyperphagia"
        else:             self.phase = "Hibernation"

        L = _levy_mat(self.pop, self.dim)
        R = np.random.rand(self.pop, self.dim)

        if alpha < 1/3:
            iA = np.random.randint(0, self.pop, self.pop)
            iB = np.random.randint(0, self.pop, self.pop)
            cand = self.pos + L*(self.best_pos - self.pos) + R*(self.pos[iA] - self.pos[iB])
        elif alpha < 2/3:
            cand = self.best_pos + R*np.exp(-alpha)*np.abs(L)*(self.best_pos - self.pos)
        else:
            sigma = max(float(np.mean((1-alpha)*(UB-LB)*0.5)), 1e-8)
            cand  = self.best_pos + np.random.normal(0, sigma, (self.pop, self.dim))

        cand = np.clip(cand, LB, UB)
        for i in range(self.pop):
            f_new = rastrigin2d(cand[i])
            if f_new < self.fit[i]:
                self.pos[i] = cand[i]; self.fit[i] = f_new
                if f_new < self.best_fit:
                    self.best_fit = f_new; self.best_pos = cand[i].copy()

        # OBL jump for worst bears
        if self.t % self.obl_interval == 0:
            n_j = max(1, int(self.pop * self.obl_rate))
            for idx in np.argsort(self.fit)[::-1][:n_j]:
                opp   = np.clip(LB + UB - self.pos[idx], LB, UB)
                f_opp = rastrigin2d(opp)
                if f_opp < self.fit[idx]:
                    self.pos[idx] = opp; self.fit[idx] = f_opp
                    if f_opp < self.best_fit:
                        self.best_fit = f_opp; self.best_pos = opp.copy()

        self.curve.append(self.best_fit)
        return self.t >= self.max_iter


class GBFAOv4State:
    label    = "GBFAO-v4  Elite-Den ★"
    color    = "#E63946"
    max_iter = 150

    def __init__(self, pop=20, dim=2):
        self.pop  = pop; self.dim = dim; self.t = 0
        self.elite_size  = max(2, int(pop * 0.20))
        self.obl_interval = 12; self.obl_rate = 0.30; self.CR = 0.9

        self.pos  = np.random.uniform(LB, UB, (pop, dim))
        self.fit  = np.array([rastrigin2d(self.pos[i]) for i in range(pop)])
        opp  = np.clip(LB + UB - self.pos, LB, UB)
        fo   = np.array([rastrigin2d(opp[i]) for i in range(pop)])
        mask = fo < self.fit
        self.pos  = np.where(mask[:, None], opp, self.pos)
        self.fit  = np.where(mask, fo, self.fit)

        bi        = np.argmin(self.fit)
        self.best_pos = self.pos[bi].copy()
        self.best_fit = float(self.fit[bi])
        self.elite    = self.pos[np.argsort(self.fit)[:self.elite_size]].copy()
        self.curve    = [self.best_fit]
        self.phase    = "Foraging"

    def _sig_alpha(self):
        return 1.0 / (1.0 + np.exp(-10*(self.t/self.max_iter - 0.5)))

    def step(self):
        self.t += 1
        alpha = self._sig_alpha()
        if   alpha < 0.33: self.phase = "Foraging"
        elif alpha < 0.67: self.phase = "Hyperphagia"
        else:              self.phase = "Hibernation"

        L = _levy_mat(self.pop, self.dim)
        R = np.random.rand(self.pop, self.dim)

        if alpha < 0.33:
            e_ref = self.elite[np.random.randint(self.elite_size)]
            iA    = np.random.randint(0, self.pop, self.pop)
            iB    = np.random.randint(0, self.pop, self.pop)
            cand  = self.pos + L*(e_ref - self.pos) + R*(self.pos[iA] - self.pos[iB])
        elif alpha < 0.67:
            cand = self.best_pos + R*np.exp(-alpha)*np.abs(L)*(self.best_pos - self.pos)
        else:
            sigma = max(np.mean((1-alpha)*(UB-LB)*0.3), 1e-8)
            cand  = np.array([self.best_pos + np.random.standard_cauchy(self.dim)*sigma
                              for _ in range(self.pop)])

        # Dimensional mask
        new_pos = self.pos.copy()
        for i in range(self.pop):
            mask_d = np.random.rand(self.dim) < self.CR
            if not mask_d.any(): mask_d[np.random.randint(self.dim)] = True
            new_pos[i] = np.where(mask_d, cand[i], self.pos[i])
        new_pos = np.clip(new_pos, LB, UB)

        for i in range(self.pop):
            f_new = rastrigin2d(new_pos[i])
            if f_new < self.fit[i]:
                self.pos[i] = new_pos[i]; self.fit[i] = f_new
                if f_new < self.best_fit:
                    self.best_fit = f_new; self.best_pos = new_pos[i].copy()

        self.elite = self.pos[np.argsort(self.fit)[:self.elite_size]].copy()

        if self.t % self.obl_interval == 0:
            n_j = max(1, int(self.pop * self.obl_rate))
            for idx in np.argsort(self.fit)[::-1][:n_j]:
                opp   = np.clip(LB + UB - self.pos[idx], LB, UB)
                f_opp = rastrigin2d(opp)
                if f_opp < self.fit[idx]:
                    self.pos[idx] = opp; self.fit[idx] = f_opp
                    if f_opp < self.best_fit:
                        self.best_fit = f_opp; self.best_pos = opp.copy()

        self.curve.append(self.best_fit)
        return self.t >= self.max_iter


# ── Figure layout ─────────────────────────────────────────────────────────────
POP  = 20
DIMS = 2

states = [GBFAOv1State(POP, DIMS),
          GBFAOv2State(POP, DIMS),
          GBFAOv3State(POP, DIMS),
          GBFAOv4State(POP, DIMS)]

PHASE_COLORS = {
    "Foraging":    "#06D6A0",
    "Hyperphagia": "#FFB703",
    "Hibernation": "#E63946",
}

fig = plt.figure(figsize=(16, 11), facecolor="#0d1117")
fig.suptitle("GBFAO v1 → v4  |  Live Optimisation on 2D Rastrigin",
             fontsize=14, fontweight="bold", color="#e6edf3", y=0.98)

# 4 landscape axes (top 2×2) + 1 convergence axis (bottom strip)
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6],
                      hspace=0.35, wspace=0.25,
                      left=0.06, right=0.97, top=0.94, bottom=0.05)

land_axes  = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
conv_ax    = fig.add_subplot(gs[2, :])

# Draw landscape contours once
CMAP = plt.cm.plasma
for ax, st in zip(land_axes, states):
    ax.contourf(XX, YY, ZZ, levels=40, cmap=CMAP, alpha=0.75)
    ax.contour (XX, YY, ZZ, levels=15, colors="white", linewidths=0.3, alpha=0.15)
    ax.set_xlim(LB, UB); ax.set_ylim(LB, UB)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#0d1117")
    for spine in ax.spines.values():
        spine.set_edgecolor(st.color); spine.set_linewidth(2)

# Scatter handles for bears
scat_bears = []
scat_bests = []
phase_txts = []
iter_txts  = []

for ax, st in zip(land_axes, states):
    sb = ax.scatter([], [], s=55, c=st.color, edgecolors="white",
                    linewidths=0.5, zorder=5, alpha=0.9)
    sg = ax.scatter([], [], s=160, c="white", marker="*", zorder=10,
                    edgecolors=st.color, linewidths=1.2, label="Best")
    scat_bears.append(sb)
    scat_bests.append(sg)

    # Phase badge (top-left)
    ptxt = ax.text(0.03, 0.97, "", transform=ax.transAxes,
                   va="top", ha="left", fontsize=8, fontweight="bold",
                   color="#0d1117",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#06D6A0",
                             edgecolor="none", alpha=0.9))
    phase_txts.append(ptxt)

    # Iter + best (bottom-right)
    itxt = ax.text(0.97, 0.03, "", transform=ax.transAxes,
                   va="bottom", ha="right", fontsize=7.5,
                   color="#e6edf3",
                   bbox=dict(boxstyle="round,pad=0.25", facecolor="#161b22",
                             edgecolor="#30363d", alpha=0.85))
    iter_txts.append(itxt)

    # Title strip
    ax.set_title(f"  {st.label}", fontsize=9.5, fontweight="bold",
                 color=st.color, loc="left", pad=4)

# Convergence axis
conv_ax.set_facecolor("#0d1117")
conv_ax.set_xlabel("Iteration", fontsize=9, color="#8b949e")
conv_ax.set_ylabel("Best Fitness", fontsize=9, color="#8b949e")
conv_ax.set_title("Convergence — all versions", fontsize=9,
                  color="#8b949e", loc="left")
conv_ax.grid(True, alpha=0.15, linestyle="--")
conv_ax.tick_params(labelsize=7)

conv_lines = []
for st in states:
    ln, = conv_ax.plot([], [], color=st.color, linewidth=2.0,
                       label=st.label.split()[0] + " " + st.label.split()[1])
    conv_lines.append(ln)
conv_ax.legend(loc="upper right", fontsize=7.5, framealpha=0.2,
               labelcolor="linecolor")

# Global-best star annotation on convergence
best_conv_dot, = conv_ax.plot([], [], "*", color="white", markersize=10, zorder=10)

TOTAL_FRAMES = max(st.max_iter for st in states) + 5
done = [False]*4


def update(frame):
    artists = []

    for k, (ax, st, sb, sg, ptxt, itxt, ln) in enumerate(
            zip(land_axes, states, scat_bears, scat_bests,
                phase_txts, iter_txts, conv_lines)):

        if not done[k]:
            done[k] = st.step()

        # Bear positions
        sb.set_offsets(st.pos)
        sg.set_offsets([st.best_pos])
        artists += [sb, sg]

        # Phase badge
        pc = PHASE_COLORS.get(st.phase, "#8b949e")
        ptxt.set_text(f" {st.phase} ")
        ptxt.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=pc,
                           edgecolor="none", alpha=0.9))
        ptxt.set_color("#0d1117")
        artists.append(ptxt)

        # Iter + best info
        pct = int(100 * st.t / st.max_iter)
        itxt.set_text(f"Iter {st.t}/{st.max_iter}  [{pct}%]\nBest: {st.best_fit:.4f}")
        artists.append(itxt)

        # Convergence line
        iters = list(range(len(st.curve)))
        ln.set_data(iters, st.curve)
        artists.append(ln)

    # Auto-scale convergence axis
    all_fits = [v for st in states for v in st.curve]
    if all_fits:
        ymin = max(0, min(all_fits) - 1)
        ymax = max(all_fits) + 2
        conv_ax.set_ylim(ymin, ymax)
        conv_ax.set_xlim(0, max(st.t for st in states) + 2)

    # Star on the globally best value found so far
    best_k  = min(range(4), key=lambda k: states[k].best_fit)
    best_st = states[best_k]
    if best_st.curve:
        bi = int(np.argmin(best_st.curve))
        best_conv_dot.set_data([bi], [best_st.curve[bi]])
    artists.append(best_conv_dot)

    return artists


ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                    interval=80, blit=False, repeat=False)

plt.show()
