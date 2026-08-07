"""Judge an MPS instance *before* handing it to the eigenvalue search.

The premise (user's taxonomy): basis insufficiency, basis ill-conditioning and
bad collocation are all properties of the **instance**, and any one of them
breaks the minimization routine. So the minimizer should never see an ill-posed
instance -- we check first, and abort or repair.

The check evaluates sigma(lambda) on a grid over the search window and reports:

  n_minima     discrete local minima found
  n_expected   eigenvalues the window should contain (two-term Weyl)
  ratio        n_minima / n_expected -- the calibration quantity
  sigma_min    depth of the deepest well
  sigma_med    typical value away from wells
  contrast     sigma_med / sigma_min -- how far the wells stand out

A well-posed instance has roughly as many minima as eigenvalues and high
contrast. A noisy one has many more minima than eigenvalues, because roundoff
wiggle manufactures them.

`plot_curve` renders the curve so it can be inspected by eye, which is still the
only test known to work reliably; the numbers above are the candidate objective
replacement, and both get recorded so we can find out whether they agree.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.join(HERE, 'run', 'curves')


def expected_count(domain, a, b):
    """Eigenvalues expected in [a, b], from the two-term Weyl law."""
    def count(lam):
        n = (domain.area * lam / (4 * np.pi)
             - domain.perimeter * np.sqrt(lam) / (4 * np.pi))
        try:
            gam = np.asarray(domain.int_angles, dtype=float)
            n += np.sum((np.pi ** 2 - gam ** 2) / (24 * np.pi * gam))
        except Exception:
            pass
        return n
    return max(count(b) - count(a), 1.0)


def scan(solver, a, b, n_pts=300, n_workers=1):
    """Evaluate sigma on a grid. Returns (lamgrid, sigma)."""
    lam = np.linspace(a, b, n_pts)
    tens = solver.tensions(lam, n_workers=n_workers)
    sigma = np.array([t[0] for t in tens])
    return lam, sigma


def metrics(domain, lam, sigma, a=None, b=None):
    """Curve statistics, the candidate objective noise test."""
    from lappy.opt import discrete_locmin_idx
    a = lam[0] if a is None else a
    b = lam[-1] if b is None else b
    idx = discrete_locmin_idx(sigma)
    n_exp = expected_count(domain, a, b)
    s_min = float(sigma.min())
    s_med = float(np.median(sigma))
    return dict(
        n_minima=int(len(idx)),
        n_expected=float(n_exp),
        ratio=float(len(idx) / n_exp),
        sigma_min=s_min,
        sigma_med=s_med,
        contrast=float(s_med / s_min) if s_min > 0 else float('inf'),
        n_pts=int(len(lam)),
    )


# Moler--Payne forces eps >= dist(lam, spectrum)/lam, so AWAY from eigenvalues the tension
# cannot be small: the low spectrum's relative gaps are 1-10%, which puts a healthy background
# at ~1e-2 or above. A background orders below that is not a better basis, it is a broken one,
# and the two ways to break it are both silent:
#
#   * a basis column that is not a particular solution in Omega -- a FundamentalBasis source
#     that landed INSIDE the domain, which normal-offset placement does routinely on a strongly
#     reentrant boundary. Measured on chevron_2_3 (24 of 240 sources inside): background sigma
#     3.2e-07 across the whole window, against 5e-02 once those columns are dropped.
#   * boundary collocation at ratio ~1 to the column count, where the fit can be zero AT the
#     points and huge between them. Measured on the same domain with a LEGITIMATE basis at
#     lam=240: sigma reads 9.1e-04 while the true Moler-Payne eps is 3.7e+01, a factor of 4e+04.
#     At ratio 1.5 and above sigma tracks eps to a factor of 5-40. lappy's default mult=2 sits
#     just above the cliff.
#
# Either way every downstream number is void, including the certificate, because ttol stops
# discriminating: at a 3e-07 background, ttol=1e-3 accepts every point in the window.
BACKGROUND_FLOOR = 1e-3


def background_suspect(m, floor=BACKGROUND_FLOOR):
    """True when the median tension is too small to be consistent with Moler--Payne.

    Cheap and it needs no reference values: the preflight scan already computes `sigma_med`.
    """
    return bool(m.get('sigma_med', 1.0) < floor)


def plot_curve(lam, sigma, title, outfile, minima=True):
    """Render sigma(lambda) for visual inspection. Returns the path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lappy.opt import discrete_locmin_idx

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.semilogy(lam, sigma, lw=0.9, color='#1f77b4')
    if minima:
        idx = discrete_locmin_idx(sigma)
        ax.plot(lam[idx], sigma[idx], 'o', ms=4, color='#d62728',
                label=f'{len(idx)} local minima')
        ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('lambda')
    ax.set_ylabel('sigma')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, which='both', lw=0.4)
    fig.tight_layout()
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    return outfile


def preflight(entry, n_basis=None, rtol=None, int_npts=None, bdry_mult=2,
              n_pts=300, n_eigs=None, plot=True, tag=''):
    """Full pre-flight on one suite domain. No eigenvalue search is performed."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                    'benchmarks', 'reference'))
    from common import build_solver, lambda_window

    dom = entry.domain()
    n_basis = n_basis or entry.n_basis
    n_eigs = n_eigs or entry.n_eigs
    solver = build_solver(dom, n_basis, rtol=rtol, bdry_mult=bdry_mult,
                          int_npts=int_npts or max(2 * n_basis, 500))
    a, b = lambda_window(dom, n_eigs)
    lam, sigma = scan(solver, a, b, n_pts=n_pts)
    m = metrics(dom, lam, sigma, a, b)
    m.update(key=entry.key, n_basis=n_basis, rtol=solver.rtol,
             int_npts=int_npts, window=(float(a), float(b)))
    if plot:
        name = f'{entry.key}{("__" + tag) if tag else ""}.png'
        m['plot'] = plot_curve(
            lam, sigma,
            f'{entry.key}  n_basis={n_basis}  rtol={solver.rtol:.0e}  '
            f'minima={m["n_minima"]} vs Weyl {m["n_expected"]:.1f}',
            os.path.join(CURVES, name))
    return solver, (a, b), m


# Calibrated on instances of known character (NOTEBOOK.md, session 7): every
# clean curve measured ratio <= 1.08, while GWW1 at rtol=1e-14 -- genuinely
# ill-conditioned, ~12 spurious minima riding on the tops of its humps -- came
# in at 2.02. Note the calibration set had to be *relabelled* first: reg_ngon_8,
# stadium and chevron all look noisy by their end results but have clean curves,
# and fail for other reasons (#4 multiplicity, #1 insufficiency, #1).
NOISE_RATIO = 1.5


def is_noisy(m, threshold=NOISE_RATIO):
    """Verdict from the pre-flight metrics: is this instance ill-conditioned?

    Detects taxonomy #2 only. A clean verdict does NOT promise accuracy -- #1
    (basis insufficiency) shows up as clean wells with too high a floor, and #4
    (search failure) shows up as a clean curve the search still mishandles.
    """
    return m['ratio'] > threshold


def max_minima_for(m, threshold=NOISE_RATIO):
    """Abort threshold to hand `bracket_mins` via `bracket_kwargs`.

    Conservative at depth: sub-windows during refinement expect far fewer than
    the whole window's count, so this cap will not false-fire there, while noise
    at depth generates minima far in excess of it anyway.
    """
    return int(np.ceil(threshold * m['n_expected']))


def summary(m):
    return (f'{m["key"]:20s} nb={m["n_basis"]:4d} rtol={m["rtol"]:.0e}  '
            f'minima={m["n_minima"]:4d} expected={m["n_expected"]:5.1f} '
            f'ratio={m["ratio"]:6.2f}  sigma_min={m["sigma_min"]:.2e} '
            f'sigma_med={m["sigma_med"]:.2e} contrast={m["contrast"]:.1e}'
            + ('   BACKGROUND SUSPECT (basis or collocation)' if background_suspect(m) else ''))
