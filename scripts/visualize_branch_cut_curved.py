"""
visualize_branch_cut_curved.py — Polyline+ray branch cut on a CURVED-boundary domain.

Companion to ``visualize_branch_cut.py`` (which uses a polygonal spiral). This script
exercises the same polyline+ray branch-cut machinery on a domain whose boundary is
genuinely curved: a smooth spiral strip whose two long edges are cubic spline segments
and whose inner/outer ends are straight caps.

As with the polygonal spiral, the inner coil "surrounds" the corner at the inner cap,
so there is no straight ray from that corner to infinity that stays in the exterior.
The branch cut for a corner-centered Fourier-Bessel function must therefore bend
through the exterior channel between the coils. The polyline cut is built on the
boundary's polyline() discretization, so it generalizes from polygons to curved
boundaries with no special handling.

This script:
  - builds a curved spiral strip (spline edges + straight caps), oriented CCW,
  - finds the surrounded inner corner and auto-generates a polyline+ray branch cut,
  - evaluates the continuous local angle Theta on a grid (inside AND outside), and
  - plots Theta as a colormap with the curved boundary and the branch cut overlaid.

The takeaway: Theta varies smoothly throughout the domain, and the only 2*pi
discontinuity is routed along the branch cut, which lies entirely in the exterior.

Run from the repo root:
    .venv/bin/python scripts/visualize_branch_cut_curved.py

Outputs (next to this script): branch_cut_curved_spiral.pdf and .png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from lappy.geometry import (MultiSegment, LineSegment, SplineSegment, Domain,
                            PointSet, corner_branch_cut_polyline)
from lappy.bases import FourierBesselBasis


def curved_spiral(turns=1.6, pitch=1.0, width=0.45, n=200, r0=0.6, bc='dir', nsamp=120):
    """A smooth spiral strip with spline edges and straight end caps, oriented CCW.

    The centerline is an Archimedean spiral r(phi) = r0 + (pitch/2pi)*phi. The strip is
    the set of points within width/2 of the centerline (along the normal). The inner
    coils surround the inner-cap corner, leaving it with no straight-ray branch-cut
    sightline to infinity (cf. the polygonal `spiral` in lappy.geometry)."""
    phi = np.linspace(0, 2 * np.pi * turns, n)
    a = pitch / (2 * np.pi)
    center = (r0 + a * phi) * np.exp(1j * phi)
    t = np.gradient(center)
    t /= np.abs(t)
    nrm = 1j * t                       # left normal
    outer = center + width / 2 * nrm
    inner = center - width / 2 * nrm

    # Boundary loop ordered CCW: inner edge (phi increasing) -> outer cap ->
    # outer edge (phi decreasing) -> inner cap. Edges are cubic spline segments.
    segs = [
        SplineSegment.interp_from_pts(inner, bc=bc, nsamp=nsamp),
        LineSegment(inner[-1], outer[-1], bc=bc),
        SplineSegment.interp_from_pts(outer[::-1], bc=bc, nsamp=nsamp),
        LineSegment(outer[0], inner[0], bc=bc),
    ]
    bdry = MultiSegment(segs, val_simple=False, val_contiguous=False)
    bdry._is_contiguous = True
    bdry._is_closed = True
    return Domain(bdry, val_simple=False, val_closed=False)


# ── curved spiral domain + the surrounded inner corner with a polyline cut ──
sp = curved_spiral(turns=1.6, pitch=1.0, width=0.45, n=200)
phi0, phi1 = sp.corner_angles
rays = sp.branch_cut_rays()
nan_idx = np.where(np.isnan(rays))[0]
assert len(nan_idx) > 0, "expected at least one surrounded corner with no straight-ray cut"
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
bx, _, _ = sp.bdry.polyline()                       # boundary discretization for extent
vx = np.concatenate([bx, verts, [c]])
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

# curved domain boundary outline (white halo + black line), using the polyline samples
bv = np.concatenate([bx, bx[:1]])
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
ax.set_title(r'Curved spiral domain: continuous $\Theta$ inside, branch-cut discontinuity routed through the exterior')
ax.legend(loc='upper right', framealpha=0.9)

here = os.path.dirname(os.path.abspath(__file__))
for ext in ('pdf', 'png'):
    out = os.path.join(here, f'branch_cut_curved_spiral.{ext}')
    fig.savefig(out, bbox_inches='tight', dpi=(150 if ext == 'png' else None))
    print('saved', out)

print('corner index', i, 'at', np.round(c, 3),
      '| polyline verts:', len(verts), '| beta=', round(beta, 3))
print('Theta range on grid: [%.2f, %.2f]' % (theta.min(), theta.max()))
