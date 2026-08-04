"""
visualize_mps.py — Step-by-step visualization of MPSEigensolver.solve_interval

Produces a multi-panel static overview figure (mps_overview.png) and an
animated GIF (mps_refinement.gif) illustrating each stage of the algorithm:

  Panel 0: Domain setup — boundary/interior points and basis sources
  Panel 1: Tension curve σ_min(λ) with eigenvalue dips
  Panel 2: GSVD spectrum at eigenvalue vs. non-eigenvalue
  Panel 3: Grid evaluation — coarse scan highlighting local minima
  Panel 4: Adaptive refinement — how bracket_mins densifies the grid
  Panel 5: Parabolic minimization — iterative parabola fitting on one bracket
  Panel 6: Multiplicity estimation — multiple SVs dip at a degenerate eigenvalue
  Panel 7: Full solve summary

Run from the repo root:
    .venv/bin/python scripts/visualize_mps.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from lappy.geometry import Polygon
from lappy.bases import FourierBesselBasis
from lappy.mps import MPSEigensolver, make_lamgrid
from lappy.opt import discrete_locmin_idx, bracket_mins, parabola_vertex, fill_refinement, flag_refinement_intervals, merge_refinement_intervals
from lappy.reference import rect_eigs

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── colors ───────────────────────────────────────────────────────────────────
C_TENSION   = '#2166ac'   # main tension curve
C_TENSION2  = '#92c5de'   # second tension curve
C_EIG       = '#d6604d'   # exact eigenvalue verticals
C_THRESH    = '#999999'   # ttol threshold
C_BRACKET   = '#4dac26'   # brackets / accepted minima
C_PARA      = '#e08214'   # parabola iterations
C_GRID      = '#888888'   # coarse grid markers

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Build solver
# ═══════════════════════════════════════════════════════════════════════════════
print("Building solver on unit square...")
verts = np.array([0, 1, 1+1j, 1j])
dom = Polygon(verts)
basis = FourierBesselBasis.from_domain(dom, orders=[40, 0, 0, 0])
bdry_pts = dom.bdry_pts([0, 40, 40, 0], kind='even')
int_pts = dom.int_pts(method='random', npts_rand=60)
solver = MPSEigensolver(basis, bdry_pts, int_pts)

# interval for all demos
LAM_A, LAM_B = 5.0, 75.0
TTOL = 1e-3

# ── exact eigenvalues for reference (first 20) ───────────────────────────────
# rect_eigs(k, L, H) returns shape (k,) for scalar L, H
exact_eigs = rect_eigs(20, 1, 1).flatten()
exact_eigs_in_range = exact_eigs[(exact_eigs >= LAM_A) & (exact_eigs <= LAM_B)]

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Pre-compute tension data  (dense grid for smooth plots)
# ═══════════════════════════════════════════════════════════════════════════════
print("Evaluating tensions on dense grid (this may take ~30 s)...")
N_DENSE = 400
lam_dense = np.linspace(LAM_A, LAM_B, N_DENSE)
t_list = solver.tensions(lam_dense, n_workers=4)
t0_dense = np.array([t[0] for t in t_list])
t1_dense = np.array([t[1] for t in t_list])
print("  done.")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Helper: draw tension curve background
# ═══════════════════════════════════════════════════════════════════════════════
def draw_tension_bg(ax, show_t1=False, show_exact=True, xlim=None, ylim=(0, 0.25)):
    """Plot the precomputed dense tension curve as a light background."""
    ax.plot(lam_dense, t0_dense, color=C_TENSION, lw=1.2, alpha=0.5, label=r'$\sigma_{\min}(\lambda)$')
    if show_t1:
        ax.plot(lam_dense, t1_dense, color=C_TENSION2, lw=0.8, alpha=0.4, ls='--',
                label=r'$\sigma_2(\lambda)$')
    ax.axhline(TTOL, color=C_THRESH, lw=0.8, ls=':', label=f'ttol={TTOL}')
    if show_exact:
        for lam_e in exact_eigs_in_range:
            ax.axvline(lam_e, color=C_EIG, lw=0.6, alpha=0.35, ls='--')
    ax.set_xlabel(r'$\lambda$', fontsize=10)
    ax.set_ylabel(r'$\sigma_{\min}$', fontsize=10)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Build figure
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 20))
fig.suptitle('MPSEigensolver.solve_interval — Algorithm Walkthrough\n(unit square, Dirichlet BC)',
             fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
                       left=0.07, right=0.97, top=0.94, bottom=0.04)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])
ax6 = fig.add_subplot(gs[3, 0])
ax7 = fig.add_subplot(gs[3, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Panel 0: Domain setup
# ─────────────────────────────────────────────────────────────────────────────
print("Panel 0: domain setup...")

bpts = bdry_pts.pts   # complex
ipts = int_pts.pts

ax0.set_aspect('equal')
ax0.set_xlim(-0.15, 1.15)
ax0.set_ylim(-0.15, 1.15)
ax0.set_title('Panel 0 — Domain Setup', fontsize=11, fontweight='bold')

# domain outline
square_xy = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
ax0.plot(square_xy[:,0], square_xy[:,1], 'k-', lw=2, zorder=3)

# boundary points
ax0.scatter(bpts.real, bpts.imag, s=8, color='#4575b4', zorder=4,
            label=f'boundary pts ({len(bpts)})')

# interior points
ax0.scatter(ipts.real, ipts.imag, s=10, color='#d73027', marker='x', lw=0.8, zorder=4,
            label=f'interior pts ({len(ipts)})')

# basis sources (the non-zero corner)
src = basis.sources[0].pts if hasattr(basis.sources[0], 'pts') else basis.sources[0]
src_pt = src[0] if hasattr(src, '__len__') else src
ax0.scatter([src_pt.real], [src_pt.imag], s=80, color='#fdae61', zorder=5,
            marker='*', label='basis source (corner)')

ax0.legend(fontsize=7, loc='upper right')
ax0.set_xlabel('x'); ax0.set_ylabel('y')

# annotation
ax0.text(0.5, -0.12, 'Unit square domain', ha='center', fontsize=9, style='italic',
         transform=ax0.transAxes)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1: Tension curve
# ─────────────────────────────────────────────────────────────────────────────
print("Panel 1: tension curve...")

ax1.set_title('Panel 1 — Tension Curve $\\sigma_{\\min}(\\lambda)$', fontsize=11, fontweight='bold')
draw_tension_bg(ax1, show_t1=True, show_exact=True)

# label a few exact eigenvalues
label_eigs = exact_eigs_in_range[:8]
for lam_e in label_eigs:
    ax1.axvline(lam_e, color=C_EIG, lw=0.8, alpha=0.6, ls='--')

ax1.set_ylim(0, 0.22)
ax1.legend(fontsize=8, loc='upper right')

# annotate a dip
dip_lam = exact_eigs_in_range[0]  # π²
idx_near = np.argmin(np.abs(lam_dense - dip_lam))
ax1.annotate(r'eigenvalue dip $\approx \pi^2$',
             xy=(lam_dense[idx_near], t0_dense[idx_near]),
             xytext=(dip_lam + 4, 0.08),
             arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
             fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2: GSVD spectrum at eigenvalue vs. non-eigenvalue
# ─────────────────────────────────────────────────────────────────────────────
print("Panel 2: GSVD spectra...")

lam_eig   = float(exact_eigs_in_range[0])      # π² — simple eigenvalue
lam_noneig = (float(exact_eigs_in_range[0]) +
              float(exact_eigs_in_range[1])) / 2  # midpoint — not an eigenvalue

t_eig_full    = solver.tensions(lam_eig)
t_noneig_full = solver.tensions(lam_noneig)
n_show = min(15, len(t_eig_full), len(t_noneig_full))

ax2.set_title('Panel 2 — GSVD Spectrum at Eigenvalue vs. Non-eigenvalue',
              fontsize=11, fontweight='bold')

x_idx = np.arange(1, n_show+1)
width = 0.35
bars_eig    = ax2.bar(x_idx - width/2, t_eig_full[:n_show],    width,
                      label=rf'at $\lambda={lam_eig:.3f}$ (eigenvalue)',
                      color=C_EIG, alpha=0.8)
bars_noneig = ax2.bar(x_idx + width/2, t_noneig_full[:n_show], width,
                      label=rf'at $\lambda={lam_noneig:.3f}$ (non-eigenvalue)',
                      color=C_TENSION, alpha=0.6)
ax2.axhline(TTOL, color=C_THRESH, lw=1, ls=':', label=f'ttol={TTOL}')
ax2.set_yscale('log')
ax2.set_xlabel('Generalized singular value index $j$', fontsize=9)
ax2.set_ylabel(r'$\sigma_j$ (log scale)', fontsize=9)
ax2.legend(fontsize=8, loc='upper left')
ax2.set_xticks(x_idx)

# annotate collapse to zero
ax2.annotate('collapses to ≈ 0\nat eigenvalue',
             xy=(1 - width/2, t_eig_full[0]),
             xytext=(3, t_eig_full[0]*5),
             arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
             fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 3: Grid evaluation — coarse scan + local minima
# ─────────────────────────────────────────────────────────────────────────────
print("Panel 3: coarse grid scan...")

N_COARSE = 30
lam_coarse = make_lamgrid(LAM_A, LAM_B, N_COARSE)
t_coarse = np.array([solver.tensions(float(l))[0] for l in lam_coarse])

locmin_idx = discrete_locmin_idx(t_coarse)

ax3.set_title('Panel 3 — Coarse Grid Scan & Discrete Local Minima',
              fontsize=11, fontweight='bold')
draw_tension_bg(ax3, show_exact=True)

# coarse grid evaluations
ax3.scatter(lam_coarse, t_coarse, s=18, color=C_GRID, zorder=4,
            label=f'coarse grid ({N_COARSE} pts)')
ax3.plot(lam_coarse, t_coarse, color=C_GRID, lw=0.7, ls='-', alpha=0.5)

# highlight local minima
ax3.scatter(lam_coarse[locmin_idx], t_coarse[locmin_idx], s=60, color=C_BRACKET,
            zorder=5, marker='v', label='discrete local minima')

ax3.set_ylim(0, 0.22)
ax3.legend(fontsize=8)

# ghost-point annotation
ax3.annotate('ghost\npoints', xy=(lam_coarse[0], t_coarse[0]),
             xytext=(lam_coarse[0]+2, 0.18),
             arrowprops=dict(arrowstyle='->', lw=0.8),
             fontsize=7, ha='center')
ax3.annotate('', xy=(lam_coarse[-1], t_coarse[-1]),
             xytext=(lam_coarse[-1]-2, 0.18),
             arrowprops=dict(arrowstyle='->', lw=0.8))


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4: Adaptive refinement (bracket_mins progression)
# ─────────────────────────────────────────────────────────────────────────────
print("Panel 4: refinement...")

# Zoom into a narrow window around the 5th eigenvalue to show refinement clearly
LAM_ZOOM_A, LAM_ZOOM_B = 45.0, 60.0
N_ZOOM_COARSE = 12

lam_zoom = make_lamgrid(LAM_ZOOM_A, LAM_ZOOM_B, N_ZOOM_COARSE)
tg_zoom  = np.array([solver.tensions(float(l))[:2] for l in lam_zoom]).T  # (2, npts)

locmin0 = discrete_locmin_idx(tg_zoom[0])
locmin1 = discrete_locmin_idx(tg_zoom[1])

# flag intervals that need refinement
refine_flag = flag_refinement_intervals(len(lam_zoom)-1, locmin0, locmin1)
refine_runs = merge_refinement_intervals(refine_flag)

ax4.set_title('Panel 4 — Adaptive Refinement (bracket_mins zoom [45,60])',
              fontsize=11, fontweight='bold')

# background: dense tension curve in zoom window
mask = (lam_dense >= LAM_ZOOM_A) & (lam_dense <= LAM_ZOOM_B)
ax4.plot(lam_dense[mask], t0_dense[mask], color=C_TENSION, lw=1.0, alpha=0.4)
ax4.plot(lam_dense[mask], t1_dense[mask], color=C_TENSION2, lw=0.8, alpha=0.3, ls='--')
ax4.axhline(TTOL, color=C_THRESH, lw=0.8, ls=':')

# coarse grid
ax4.scatter(lam_zoom, tg_zoom[0], s=18, color=C_GRID, zorder=3, label='coarse grid')
ax4.plot(lam_zoom, tg_zoom[0], color=C_GRID, lw=0.6, alpha=0.5)

# local minima of σ_0 and σ_1
ax4.scatter(lam_zoom[locmin0], tg_zoom[0, locmin0], s=55, color=C_BRACKET,
            marker='v', zorder=5, label=r'$\sigma_{\min}$ local min')
ax4.scatter(lam_zoom[locmin1], tg_zoom[1, locmin1], s=40, color=C_TENSION2,
            marker='^', zorder=5, label=r'$\sigma_2$ local min')

# shade intervals flagged for refinement
for start, end in refine_runs:
    ax4.axvspan(lam_zoom[start], lam_zoom[end], alpha=0.12, color=C_PARA,
                label='flagged for refinement' if start == refine_runs[0][0] else None)

# refined sub-grid for one flagged run
if len(refine_runs) > 0:
    s, e = refine_runs[0]
    x_ref, y_ref, _ = fill_refinement(
        lambda l: solver.tensions(float(l))[:2], lam_zoom, tg_zoom, s, e, shrink=4)
    ax4.scatter(x_ref, y_ref[0], s=10, color=C_PARA, zorder=6,
                label='refined sub-grid (shrink=4)')

ax4.set_xlim(LAM_ZOOM_A, LAM_ZOOM_B)
ax4.set_ylim(0, 0.22)
ax4.set_xlabel(r'$\lambda$', fontsize=10)
ax4.set_ylabel(r'$\sigma_{\min}$', fontsize=10)
ax4.legend(fontsize=7, loc='upper right')


# ─────────────────────────────────────────────────────────────────────────────
# Panel 5: Parabolic minimization on one bracket
# ─────────────────────────────────────────────────────────────────────────────
print("Panel 5: parabolic minimization...")

# Pick the bracket around the first eigenvalue
TARGET_EIG = float(exact_eigs_in_range[0])
half_width = 0.8
xa, xb = TARGET_EIG - half_width, TARGET_EIG + half_width

# 3 initial points
x_init = np.array([xa, (xa + xb)/2, xb])
y_init = np.array([solver.tensions(float(xi))[0]**2 for xi in x_init])

# Collect parabolic iterations manually
MAXITER = 6
iter_x  = [x_init.copy()]
iter_y  = [y_init.copy()]

x_cur = x_init.copy()
y_cur = y_init.copy()
f_sq  = lambda xi: solver.tensions(float(xi))[0]**2

for _ in range(MAXITER):
    v = parabola_vertex(x_cur, y_cur)
    v = np.clip(v, x_cur[0], x_cur[2])
    yv = f_sq(v)
    if v < x_cur[1]:
        x_cur = np.array([x_cur[0], v, x_cur[1]])
        y_cur = np.array([y_cur[0], yv, y_cur[1]])
    else:
        x_cur = np.array([x_cur[1], v, x_cur[2]])
        y_cur = np.array([y_cur[1], yv, y_cur[2]])
    iter_x.append(x_cur.copy())
    iter_y.append(y_cur.copy())

ax5.set_title('Panel 5 — Parabolic Iterative Minimization\n(bracket around $\\pi^2$)',
              fontsize=11, fontweight='bold')

# background tension²
lam_zoom5 = np.linspace(xa - 0.1, xb + 0.1, 200)
t_zoom5 = np.array([solver.tensions(float(l))[0]**2 for l in lam_zoom5])
ax5.plot(lam_zoom5, t_zoom5, color=C_TENSION, lw=1.5, alpha=0.5, label=r'$\sigma_{\min}^2(\lambda)$')

# exact eigenvalue
ax5.axvline(TARGET_EIG, color=C_EIG, lw=1.2, ls='--', label=rf'exact $\pi^2={TARGET_EIG:.4f}$')

# parabolic iterations
colors_iter = plt.cm.autumn(np.linspace(0, 0.9, len(iter_x)))
for i, (xi, yi) in enumerate(zip(iter_x, iter_y)):
    # fit & plot parabola
    if i < len(iter_x) - 1:
        v_para = parabola_vertex(xi, yi)
        v_para = np.clip(v_para, xi[0], xi[2])
        # sample parabola
        xp = np.linspace(xi[0], xi[2], 80)
        # fit 2nd-degree polynomial through 3 points
        coeff = np.polyfit(xi, yi, 2)
        yp = np.polyval(coeff, xp)
        ax5.plot(xp, yp, '--', color=colors_iter[i], lw=0.8, alpha=0.6)
    # plot 3-point bracket
    ax5.scatter(xi, yi, s=30, color=colors_iter[i], zorder=5)

# label iterations
ax5.scatter([], [], s=30, color='#fdae61', label=f'iterations 0–{MAXITER}')
ax5.set_xlabel(r'$\lambda$', fontsize=10)
ax5.set_ylabel(r'$\sigma_{\min}^2$', fontsize=10)
ax5.legend(fontsize=8)

# mark final minimizer
final_min = iter_x[-1][1]
ax5.axvline(final_min, color=C_BRACKET, lw=1.2, ls='-.',
            label=f'final min={final_min:.6f}')
ax5.set_xlim(xa - 0.1, xb + 0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 6: Multiplicity estimation near a degenerate eigenvalue
# ─────────────────────────────────────────────────────────────────────────────
print("Panel 6: multiplicity estimation...")

# 5π² ≈ 49.348 is a double eigenvalue of the unit square (1,2) and (2,1)
DEG_EIG = float(exact_eigs_in_range[exact_eigs_in_range > 49][0])
hw6 = 2.5
lam_zoom6 = np.linspace(DEG_EIG - hw6, DEG_EIG + hw6, 200)
t_mult = [solver.tensions(float(l))[:4] for l in lam_zoom6]
t_mult = np.array(t_mult).T   # (4, 200)

ax6.set_title(r'Panel 6 — Multiplicity Estimation near $5\pi^2$ (degenerate)',
              fontsize=11, fontweight='bold')

colors6 = [C_TENSION, C_TENSION2, C_EIG, '#999999']
labels6 = [r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$', r'$\sigma_4$']
for k in range(4):
    ax6.plot(lam_zoom6, t_mult[k], color=colors6[k], lw=1.4,
             alpha=0.85, label=labels6[k])

ax6.axhline(TTOL, color=C_THRESH, lw=1, ls=':', label=f'ttol={TTOL}')
ax6.axvline(DEG_EIG, color=C_EIG, lw=1.2, ls='--',
            label=rf'$5\pi^2 \approx {DEG_EIG:.3f}$ (mult=2)')

# shade the "both below ttol" region
lam_arr = lam_zoom6
in_region = (t_mult[0] <= TTOL) & (t_mult[1] <= TTOL)
ax6.fill_between(lam_arr, 0, TTOL, where=in_region,
                 alpha=0.18, color=C_BRACKET, label=r'$\sigma_1,\sigma_2 \leq$ ttol (mult=2)')

ax6.set_ylim(0, 0.22)
ax6.set_xlabel(r'$\lambda$', fontsize=10)
ax6.set_ylabel(r'$\sigma_j$', fontsize=10)
ax6.legend(fontsize=8, loc='upper right')

ax6.annotate('Both $\\sigma_1$ and $\\sigma_2$\ndip — multiplicity = 2',
             xy=(DEG_EIG, t_mult[1][len(lam_zoom6)//2]),
             xytext=(DEG_EIG + 1.0, 0.12),
             arrowprops=dict(arrowstyle='->', lw=0.8),
             fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 7: Full solve summary
# ─────────────────────────────────────────────────────────────────────────────
print("Panel 7: full solve summary...")

print("  Running solver (this takes ~10 s)...")
eigs_found, mults_found, fevals = solver.solve_interval(LAM_A, LAM_B, 60, n_workers=4)
print(f"  Found {len(eigs_found)} distinct eigenvalues, total mult={mults_found.sum()}, fevals={fevals}")

ax7.set_title('Panel 7 — Full Solve Summary', fontsize=11, fontweight='bold')
draw_tension_bg(ax7, show_exact=True)

# mark found eigenvalues
for lam_f, mult in zip(eigs_found, mults_found):
    ax7.axvline(lam_f, color=C_BRACKET, lw=1.5, alpha=0.75)
    ax7.text(lam_f, 0.205, str(int(mult)), ha='center', fontsize=6,
             color=C_BRACKET, fontweight='bold')

# legend entries
ax7.axvline(np.nan, color=C_BRACKET, lw=1.5, label=f'found eigenvalues ({len(eigs_found)})')
ax7.axvline(np.nan, color=C_EIG, lw=0.8, ls='--', alpha=0.5, label='exact (reference)')

# text box with summary stats
n_exact_in = int(((exact_eigs_in_range[:, None] >= LAM_A) &
                  (exact_eigs_in_range[:, None] <= LAM_B)).any(axis=1).sum())
summary_txt = (
    f"Interval: [{LAM_A}, {LAM_B}]\n"
    f"Grid pts: 60 (+2 ghost)\n"
    f"Found:    {len(eigs_found)} distinct eigs  (total mult={int(mults_found.sum())})\n"
    f"Fevals:   {fevals}"
)
ax7.text(0.98, 0.97, summary_txt, transform=ax7.transAxes,
         ha='right', va='top', fontsize=8,
         bbox=dict(boxstyle='round', fc='white', alpha=0.85))

ax7.set_ylim(0, 0.22)
ax7.legend(fontsize=8, loc='upper left')


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Save static figure
# ═══════════════════════════════════════════════════════════════════════════════
out_dir  = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'mps_overview.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved static overview → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Animated GIF: refinement progression
# ═══════════════════════════════════════════════════════════════════════════════
print("Building refinement animation...")

LAM_ANIM_A, LAM_ANIM_B = 5.0, 55.0
N_ANIM_INIT = 20

# Collect snapshots of (lamgrid, t0grid) at each stage: coarse + successive refinements
lam_anim_init = make_lamgrid(LAM_ANIM_A, LAM_ANIM_B, N_ANIM_INIT)
tg_anim_init  = np.array([solver.tensions(float(l))[:2] for l in lam_anim_init]).T

def collect_refinement_snapshots(lam_init, tg_init, max_depth=3, shrink=2):
    """Collect (x, y0) pairs at each depth of bracket_mins recursion."""
    snapshots = [(lam_init.copy(), tg_init[0].copy(), 'initial coarse grid')]

    # manually replicate one level of refinement to gather frames
    x, y = lam_init.copy(), tg_init.copy()
    for depth in range(max_depth):
        y0_min = discrete_locmin_idx(y[0])
        y1_min = discrete_locmin_idx(y[1])
        if len(y0_min) == 0:
            break
        refine_flag = flag_refinement_intervals(len(x)-1, y0_min, y1_min)
        refine_runs = merge_refinement_intervals(refine_flag)
        if len(refine_runs) == 0:
            break

        x_new_parts = []
        y_new_parts = []
        prev_end_idx = 0
        for start, end in refine_runs:
            # copy segment before this run
            x_new_parts.append(x[prev_end_idx:start+1])
            y_new_parts.append(y[:, prev_end_idx:start+1])
            # refined segment
            x_ref, y_ref, _ = fill_refinement(
                lambda l: solver.tensions(float(l))[:2], x, y, start, end, shrink)
            # strip ghost points that fill_refinement prepended/appended
            if start > 0:
                x_ref, y_ref = x_ref[1:], y_ref[:, 1:]
            if end < len(x)-1:
                x_ref, y_ref = x_ref[:-1], y_ref[:, :-1]
            x_new_parts.append(x_ref)
            y_new_parts.append(y_ref)
            prev_end_idx = end + 1
        x_new_parts.append(x[prev_end_idx:])
        y_new_parts.append(y[:, prev_end_idx:])

        x = np.concatenate(x_new_parts)
        y = np.concatenate(y_new_parts, axis=1)

        snapshots.append((x.copy(), y[0].copy(), f'after refinement pass {depth+1}'))

    return snapshots

snapshots = collect_refinement_snapshots(lam_anim_init, tg_anim_init)

# Build animation
fig_anim, ax_anim = plt.subplots(figsize=(10, 4))
fig_anim.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.15)

mask_anim = (lam_dense >= LAM_ANIM_A) & (lam_dense <= LAM_ANIM_B)

line_bg,   = ax_anim.plot(lam_dense[mask_anim], t0_dense[mask_anim],
                           color=C_TENSION, lw=1.0, alpha=0.3)
ax_anim.axhline(TTOL, color=C_THRESH, lw=0.8, ls=':')
for lam_e in exact_eigs_in_range[exact_eigs_in_range < LAM_ANIM_B]:
    ax_anim.axvline(lam_e, color=C_EIG, lw=0.6, alpha=0.3, ls='--')

scat_pts  = ax_anim.scatter([], [], s=20, color=C_GRID, zorder=4)
line_conn,= ax_anim.plot([], [], color=C_GRID, lw=0.6, alpha=0.5)
scat_min  = ax_anim.scatter([], [], s=60, color=C_BRACKET, zorder=5, marker='v')
title_txt = ax_anim.set_title('', fontsize=11)

ax_anim.set_xlim(LAM_ANIM_A, LAM_ANIM_B)
ax_anim.set_ylim(0, 0.22)
ax_anim.set_xlabel(r'$\lambda$', fontsize=10)
ax_anim.set_ylabel(r'$\sigma_{\min}$', fontsize=10)

def init_anim():
    scat_pts.set_offsets(np.empty((0, 2)))
    line_conn.set_data([], [])
    scat_min.set_offsets(np.empty((0, 2)))
    return scat_pts, line_conn, scat_min, title_txt

def update_anim(frame):
    x, t0, label = snapshots[frame]
    pts = np.column_stack([x, t0])
    scat_pts.set_offsets(pts)
    line_conn.set_data(x, t0)
    # highlight local minima
    lm_idx = discrete_locmin_idx(t0)
    if len(lm_idx) > 0:
        scat_min.set_offsets(np.column_stack([x[lm_idx], t0[lm_idx]]))
    else:
        scat_min.set_offsets(np.empty((0, 2)))
    title_txt.set_text(f'Refinement Animation — {label}  ({len(x)} grid pts)')
    return scat_pts, line_conn, scat_min, title_txt

anim = animation.FuncAnimation(
    fig_anim, update_anim, frames=len(snapshots),
    init_func=init_anim, interval=1200, blit=True, repeat_delay=2000
)

gif_path = os.path.join(out_dir, 'mps_refinement.gif')
try:
    anim.save(gif_path, writer='pillow', fps=0.8, dpi=120)
    print(f"Saved animation       → {gif_path}")
except Exception as exc:
    print(f"Warning: could not save GIF ({exc}). Install pillow: pip install pillow")

plt.show()
print("\nDone.")
