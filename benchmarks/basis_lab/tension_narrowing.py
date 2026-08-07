"""Why a better basis can make the eigenvalue search FAIL.

`chevron_2_3` at `n_basis=480` with distributed sources missed a true eigenvalue at
226.6204 that the `n_basis=160` default basis finds every time. The reason is visible in one
picture: better conditioning deepens and NARROWS the tension minima, while the bracket search
runs on a fixed grid of `11 * n_eigs` points across the whole window. Past some point the
wells are narrower than the grid spacing and the scan steps over them.

    python -m benchmarks.basis_lab.tension_narrowing
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'benchmarks', 'reference'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmarks.suite.domains import SUITE
from lappy import bases, mps, MPSEigensolver
from basis_lab import fb_plus_fs_bdry
from common import lambda_window

KEY = 'chevron_2_3'
MISSED = 226.6204          # true eigenvalue the better basis skipped
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'suite', 'run', 'curves', 'tension_narrowing.png')


def solver_for(dom, kind, nb, seed=0):
    np.random.seed(seed)
    basis = (bases.make_default_basis(dom, nb) if kind == 'default'
             else fb_plus_fs_bdry(dom, bases.fb_corner_orders(dom, nb//2), nb//2, d=0.4))
    bp = dom.bdry_pts(mps.pts_per_seg(dom, basis, mult=2))
    ip = dom.int_pts(method='random', npts_rand=max(2*nb, 500))
    return MPSEigensolver(basis.to_normalized((bp, ip)), bp, ip, rtol=1e-14, ttol=1e-3)


def well_width(lams, sig, centre, rise=100.0):
    """Width of the minimum around `centre`, measured where sigma rises to `rise` x its floor."""
    i = int(np.argmin(np.abs(lams - centre)))
    j = int(np.argmin(sig[max(i-40, 0):i+40])) + max(i-40, 0)
    floor = sig[j]
    lo = hi = lams[j]
    k = j
    while k > 0 and sig[k] < rise*floor:
        lo = lams[k]; k -= 1
    k = j
    while k < len(lams)-1 and sig[k] < rise*floor:
        hi = lams[k]; k += 1
    return hi - lo, floor


def main():
    dom = SUITE[KEY].domain()
    n_eigs = SUITE[KEY].n_eigs
    a, b = lambda_window(dom, n_eigs)
    grid = mps.make_lamgrid(a, b, max(11*n_eigs, 50))          # what the search actually samples
    dense = mps.make_lamgrid(a, b, max(30*n_eigs, 50))         # the corrected --pts-per-eig=30

    cases = [('default, n_basis=160  (finds it)', 'default', 160, 'tab:orange'),
             ('distributed sources, n_basis=480  (misses it)', 'placed', 480, 'tab:blue')]

    zoom = np.linspace(MISSED - 1.2, MISSED + 1.2, 260)
    wide = np.linspace(a, b, 420)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.4))
    widths = {}
    for label, kind, nb, colour in cases:
        s = solver_for(dom, kind, nb)
        sw = np.array([s.sigma(x) for x in wide])
        sz = np.array([s.sigma(x) for x in zoom])
        axes[0].semilogy(wide, sw, color=colour, lw=1.1, label=label)
        axes[1].semilogy(zoom, sz, color=colour, lw=1.6, label=label)
        widths[label] = well_width(zoom, sz, MISSED)

    for ax, g, gl in ((axes[0], grid, 'search grid (11/eig)'),):
        for x in g:
            ax.axvline(x, color='0.85', lw=0.5, zorder=0)
        ax.plot([], [], color='0.85', lw=0.5, label=gl)

    axes[1].vlines(grid[(grid > zoom[0]) & (grid < zoom[-1])], 1e-15, 1e0,
                   color='0.75', lw=1.0, label='search grid (11/eig)')
    axes[1].vlines(dense[(dense > zoom[0]) & (dense < zoom[-1])], 1e-15, 1e0,
                   color='tab:green', lw=0.7, ls=':', label='corrected grid (30/eig)')
    axes[1].axvline(MISSED, color='k', lw=1.0, ls='--', label=f'true eigenvalue {MISSED}')

    axes[0].set_title(f'{KEY}: tension over the whole search window')
    axes[0].set_ylabel(r'$\sigma(\lambda)$')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[1].set_title('zoom on the eigenvalue the better basis missed')
    axes[1].set_xlabel(r'$\lambda$'); axes[1].set_ylabel(r'$\sigma(\lambda)$')
    axes[1].set_xlim(zoom[0], zoom[-1]); axes[1].legend(fontsize=8, loc='upper right')

    txt = '\n'.join(f'{k.split(",")[0]}: well width {w:.3f}, floor {f:.1e}'
                    for k, (w, f) in widths.items())
    gap = grid[1] - grid[0]
    txt += f'\nsearch grid spacing: {gap:.3f}'
    axes[1].text(0.02, 0.04, txt, transform=axes[1].transAxes, fontsize=8,
                 va='bottom', family='monospace',
                 bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f'wrote {OUT}')
    for k, (w, f) in widths.items():
        print(f'  {k}: well width (100x floor) = {w:.4f}, floor = {f:.2e}')
    print(f'  search grid spacing = {gap:.4f}  ({max(11*n_eigs,50)} points over [{a:.1f}, {b:.1f}])')


if __name__ == '__main__':
    main()
