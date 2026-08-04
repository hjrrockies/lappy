"""
demo_adaptive_pts.py — Adaptive vs uniform boundary sampling on curved domains.

Shows how adaptive_pts places points densely where curvature is high and sparsely
on flat regions, compared to a uniform-in-parameter distribution.

Run from the repo root:
    .venv/bin/python scripts/demo_adaptive_pts.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lappy.geometry import disk, disk_sector, mushroom

# ---------------------------------------------------------------------------
# Helper: plot one domain comparison panel
# ---------------------------------------------------------------------------

def plot_panel(ax, domain, eps, n_uniform, title):
    bdry = domain.bdry

    # adaptive sample
    adpt = bdry.adaptive_pts(eps=eps)

    # uniform sample (same method as plotting, linspace in tau)
    tau_u = np.linspace(0, 1, n_uniform + 1)
    segs = bdry.segments
    # build uniform points by stacking per-segment linspace evaluations
    uni_pts = []
    for seg in segs:
        tau = np.linspace(0, 1, n_uniform // len(segs) + 2)
        uni_pts.append(seg.p(tau))
    uni_pts = np.concatenate(uni_pts)

    # draw the true boundary (dense)
    tau_dense = np.linspace(0, 1, 2000)
    true_pts = []
    for seg in segs:
        true_pts.append(seg.p(np.linspace(0, 1, 500)))
    true_pts = np.concatenate(true_pts)

    ax.plot(true_pts.real, true_pts.imag, 'k-', lw=1.2, label='true boundary', zorder=1)
    ax.plot(uni_pts.real, uni_pts.imag, 'b--', lw=0.8, alpha=0.7, label=f'uniform ({len(uni_pts)} pts)', zorder=2)
    ax.plot(adpt.pts.real, adpt.pts.imag, 'r--', lw=0.8, alpha=0.7,
            label=f'adaptive ε={eps} ({len(adpt)} pts)', zorder=3)
    ax.scatter(adpt.pts.real, adpt.pts.imag, s=18, color='red', zorder=4)
    ax.scatter(uni_pts.real, uni_pts.imag, s=10, color='blue', alpha=0.6, zorder=3)

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(fontsize=7, loc='upper right')
    ax.axis('off')


# ---------------------------------------------------------------------------
# Figure: three domains side by side
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Adaptive vs uniform boundary sampling', fontsize=13)

plot_panel(axes[0], disk(r=1.0),        eps=1e-4, n_uniform=20,  title='Unit disk  (ε = 1e-4)')
plot_panel(axes[1], disk_sector(r=1.0, theta=np.pi*2/3),
                                         eps=1e-4, n_uniform=20,  title='Disk sector  (ε = 1e-4)')
plot_panel(axes[2], mushroom(),          eps=1e-4, n_uniform=30,  title='Mushroom  (ε = 1e-4)')

plt.tight_layout()
plt.savefig('scripts/adaptive_pts_demo.png', dpi=150, bbox_inches='tight')
print("Saved scripts/adaptive_pts_demo.png")
plt.show()
