"""
visualize_branch_cut.py — Polyline+ray branch cut on a spiral domain.

Some corners of a non-convex domain (here, the inner coils of a spiral) have no
straight-ray sightline to infinity that stays in the exterior, so the branch cut for
a corner-centered Fourier-Bessel function must bend through the exterior channel.
This script:

  - builds a spiral domain with several "surrounded" corners,
  - auto-generates a polyline+ray branch cut for one of them,
  - evaluates the resulting continuous local angle Theta on a grid (inside AND
    outside the domain), and
  - plots Theta as a colormap with the domain boundary and the branch cut overlaid.

The takeaway: Theta varies smoothly throughout the domain, and the only 2*pi
discontinuity is routed along the branch cut, which lies entirely in the exterior.

Run from the repo root:
    .venv/bin/python scripts/visualize_branch_cut.py

Outputs (next to this script): branch_cut_spiral.pdf and branch_cut_spiral.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from lappy.geometry import spiral, PointSet, corner_branch_cut_rays, corner_branch_cut_polyline
from lappy.bases import FourierBesselBasis

# ── spiral domain + a surrounded corner with an auto-generated polyline cut ──
sp = spiral(turns=1.6, pitch=1.0, width=0.45, n=14)
phi0, phi1 = sp.corner_angles
rays = corner_branch_cut_rays(sp)
nan_idx = np.where(np.isnan(rays))[0]
# pick the surrounded corner with the longest polyline for the richest picture
i = int(max(nan_idx, key=lambda k: len(corner_branch_cut_polyline(sp, int(k))[0])))
verts, beta = corner_branch_cut_polyline(sp, i)
c = sp.corners[i]

# single-source basis at that corner, using the polyline+ray branch cut
orders = np.zeros(len(sp.corners), int)
orders[i] = 1
bpl = [None] * len(sp.corners)
bpl[i] = (verts, beta)
basis = FourierBesselBasis(sp.corners, phi0, phi1, orders, np.nan_to_num(rays), 'sin',
                           branch_polylines=bpl)

# ── evaluate the continuous angle Theta on a grid (inside AND outside the domain) ──
vx = np.concatenate([sp.vertices, verts, [c]])
pad = 0.5
xmin, xmax = vx.real.min() - pad, vx.real.max() + pad
ymin, ymax = vx.imag.min() - pad, vx.imag.max() + pad
ng = 700
X, Y = np.meshgrid(np.linspace(xmin, xmax, ng), np.linspace(ymin, ymax, ng))
theta = basis._theta(PointSet((X + 1j * Y).ravel()))[:, 0].reshape(ng, ng)

# branch cut polyline + ray for drawing
R = 1.2 * max(xmax - xmin, ymax - ymin)
cut = np.concatenate([[c], verts, [verts[-1] + R * np.exp(1j * beta)]])

# ── plot ──
fig, ax = plt.subplots(figsize=(9, 9))
pcm = ax.pcolormesh(X, Y, theta, cmap='viridis', shading='auto')
cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
cb.set_label(r'local angle $\Theta$ about the corner (radians)')

# domain boundary outline (white halo + black line)
bv = np.concatenate([sp.vertices, sp.vertices[:1]])
ax.plot(bv.real, bv.imag, color='white', lw=1.6)
ax.plot(bv.real, bv.imag, color='black', lw=0.8)

# branch cut: polyline part solid, final ray dashed
ax.plot(cut[:-1].real, cut[:-1].imag, color='red', lw=2.4, label='polyline branch cut')
ax.plot(cut[-2:].real, cut[-2:].imag, color='red', lw=2.4, ls='--', label='branch-cut ray')
ax.plot(cut.real, cut.imag, 'o', color='red', ms=4)
ax.plot([c.real], [c.imag], '*', color='yellow', ms=18, mec='black', label='corner (source)')

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal')
ax.set_title(r'Spiral domain: continuous $\Theta$ inside, branch-cut discontinuity routed through the exterior')
ax.legend(loc='upper right', framealpha=0.9)

here = os.path.dirname(os.path.abspath(__file__))
for ext in ('pdf', 'png'):
    out = os.path.join(here, f'branch_cut_spiral.{ext}')
    fig.savefig(out, bbox_inches='tight', dpi=(150 if ext == 'png' else None))
    print('saved', out)

print('corner index', i, 'at', np.round(c, 3),
      '| polyline verts:', len(verts), '| beta=', round(beta, 3))
print('Theta range on grid: [%.2f, %.2f]' % (theta.min(), theta.max()))
