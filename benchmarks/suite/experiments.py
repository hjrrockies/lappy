"""Diagnostic experiments for the reference run. Not part of the suite proper.

    python -m benchmarks.suite.experiments rank_vs_p [--n-basis 160]
    python -m benchmarks.suite.experiments rank_curve <key> [--sizes 60,120,240]

`rank_vs_p` is the decisive test of the "precision-bound near-dependence"
hypothesis (see NOTEBOOK.md). It walks the `disk_sector` family, which sweeps
the corner exponent `p = pi/gamma` while holding *everything else* fixed --
same one curved edge, same two straight edges, same symmetry, closed-form
spectrum -- and reports, at fixed basis size:

  n_reg / n   numerical rank of the collocation pencil after regularization
  sigma       tension at the (exactly known) first eigenvalue
  err         true relative error against `reference.sector_eigs`

If the hypothesis holds, `n_reg/n` should fall and `err` should rise
monotonically with `p`, with no other variable moving. If `n_reg/n` stays flat
while `err` rises, the problem is approximation power, not conditioning, and
the whole line of attack is wrong.

Results are appended to `run/experiments.jsonl` so they survive a context reset.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'benchmarks', 'reference'))

OUT = os.path.join(HERE, 'run', 'experiments.jsonl')


def _record(rec):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')


def probe(domain, n_basis, lam, rtol=1e-14):
    """Numerical rank and tension of the pencil at a given lambda."""
    from common import build_solver
    solver = build_solver(domain, n_basis)
    sigma = float(solver.sigma(lam))
    A_B, A_I = solver.A_B(lam), solver.A_I(lam)
    n_cols = A_B.shape[1]
    # numerical rank of the stacked collocation matrix, the quantity
    # regularize_pencil truncates on
    s = np.linalg.svd(np.vstack([A_B, A_I]), compute_uv=False)
    n_reg = int(np.count_nonzero(s > rtol * s[0]))
    return dict(sigma=sigma, n_cols=int(n_cols), n_reg=n_reg,
                rank_frac=n_reg / n_cols,
                cond=float(s[0] / max(s[-1], 1e-300)))


def cmd_rank_vs_p(args):
    """Sweep the corner exponent against exact truth, everything else fixed."""
    from lappy import geometry as G, reference as R

    # gamma chosen so p is never pi/integer: the corner must stay singular.
    thetas = [('slit_p0.50', 2 * np.pi - 0.05),
              ('reflex_p0.67', 3 * np.pi / 2),
              ('p1.4', np.pi / 1.4),
              ('p2.3', np.pi / 2.3),
              ('p3.7', np.pi / 3.7),
              ('p6.5', np.pi / 6.5),
              ('p9.1', np.pi / 9.1),
              ('p13.3', np.pi / 13.3)]
    print(f'{"case":14s} {"p":>6} {"n_reg/n":>10} {"cond":>9} '
          f'{"sigma":>10} {"err_lam1":>10}')
    for name, theta in thetas:
        dom = G.disk_sector(1, theta)
        lam_exact = float(R.sector_eigs(1, 1, theta)[0])
        try:
            pr = probe(dom, args.n_basis, lam_exact)
        except Exception as exc:
            print(f'{name:14s} FAILED {type(exc).__name__}: {exc}')
            continue
        p = np.pi / theta
        rec = dict(experiment='rank_vs_p', case=name, theta=theta, p=p,
                   n_basis=args.n_basis, lam_exact=lam_exact, **pr)
        _record(rec)
        print(f'{name:14s} {p:6.2f} {pr["n_reg"]:4d}/{pr["n_cols"]:<4d} '
              f'{pr["cond"]:9.2e} {pr["sigma"]:10.2e}')
    return 0


def cmd_rank_curve(args):
    """How does numerical rank grow with nominal basis size for one domain?

    If rank saturates while n_basis keeps growing, extra basis functions are
    numerically redundant and escalating n_basis cannot help.
    """
    from benchmarks.suite.domains import SUITE
    entry = SUITE[args.key]
    dom = entry.domain()
    sizes = [int(s) for s in args.sizes.split(',')]

    if entry.truth_fn is not None:
        lam = float(entry.truth_fn(1)[0])
    else:
        from common import lambda_window
        a, b = lambda_window(dom, 1)
        lam = 0.5 * (a + b)

    print(f'{args.key}  lam={lam:.6f}')
    print(f'{"n_basis":>8} {"n_reg":>7} {"n_cols":>7} {"frac":>6} '
          f'{"cond":>9} {"sigma":>10}')
    for nb in sizes:
        try:
            pr = probe(dom, nb, lam)
        except Exception as exc:
            print(f'{nb:8d} FAILED {type(exc).__name__}: {exc}')
            continue
        _record(dict(experiment='rank_curve', key=args.key, n_basis=nb,
                     lam=lam, **pr))
        print(f'{nb:8d} {pr["n_reg"]:7d} {pr["n_cols"]:7d} '
              f'{pr["rank_frac"]:6.2f} {pr["cond"]:9.2e} {pr["sigma"]:10.2e}')
    return 0


def exact_interior_factor(basis, lam, rellich_data, rtol=1e-13):
    """Factor ``L`` with ``L^T L = G``, where ``G[i,j] = <phi_i, phi_j>_{L2(Omega)}``.

    The MPS tension is a GSVD of the boundary matrix against an *interior*
    matrix whose only job is to supply ``||u||_{L2(Omega)}``. That interior
    matrix is normally a Monte-Carlo collocation, which is why the answer moves
    when the draw moves -- up to 3.3 digits on iso_right_tri, and occasionally a
    missed eigenvalue.

    The Rellich identity gives the same Gram matrix exactly, from boundary data
    alone. Substituting a factor of it for the sampled block makes the interior
    norm exact and removes the random draw from the pipeline entirely.

    Uses an eigendecomposition rather than Cholesky: ``G`` is genuinely
    near-singular (the near-null space is the whole point of this run), so
    Cholesky fails, while the symmetric factorization handles it.

    Returns the **square** symmetric square root, not a reduced-rank factor.
    Dropping the null directions changes the block's row count, and
    ``manual_solve`` guides its grid refinement using the *two* smallest
    tensions -- so a shape change quietly perturbs that heuristic. With a
    reduced factor, iso_right_tri lost lambda_1 and lambda_4 while every other
    mode came out exact.
    """
    import numpy as np
    raise NotImplementedError(
        "RETIRED with lappy.rellich. This experiment forms the BASIS-LEVEL (N x N) Rellich "
        "Gram, which no longer exists: a basis-level Gram mixes columns centred at other "
        "corners, which are plain analytic there, so no corner-adapted quadrature can serve "
        "it -- see lappy/eigfun_integrals.py's module docstring. Its own conclusion is "
        "recorded above and in NOTEBOOK: forming G was unaffordable (>30 min against 165s), "
        "which is why cmd_inexact_rellich exists. Kept for the record, not for running.")
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def cmd_exact_interior(args):
    """Replace the sampled interior block with the exact Rellich Gram factor."""
    import numpy as np
    from lappy import bases, mps, MPSEigensolver
    from lappy.eigfun_integrals import boundary_quadrature
    from benchmarks.suite.domains import SUITE
    from common import manual_solve, polish_eigs, lambda_window
    from certify import certify_solver

    np.random.seed(args.seed)
    entry = SUITE[args.key]
    dom = entry.domain()
    n_basis = args.n_basis or entry.n_basis
    n_eigs = entry.n_eigs

    basis = bases.make_default_basis(dom, n_basis)
    n_per_seg = mps.pts_per_seg(dom, basis, mult=2)
    bdry_pts = dom.bdry_pts(n_per_seg)
    int_pts = dom.int_pts(method='random', npts_rand=max(2 * n_basis, 500),
                          rng=args.seed)
    basis = basis.to_normalized((bdry_pts, int_pts))

    a, b = lambda_window(dom, n_eigs)
    rd = boundary_quadrature(dom, b)   # no basis needed; sizes itself from geometry+lam

    solver = MPSEigensolver(basis, bdry_pts, int_pts, rtol=1e-14, ttol=1e-3)
    A_B = solver.A_B

    # Monkey-patch the interior block for this experiment only.
    def A_I_exact(lam):
        return exact_interior_factor(basis, lam, rd)
    solver.A_I = A_I_exact
    for attr in list(solver.__dict__):
        if attr.startswith('_icache_'):
            del solver.__dict__[attr]

    e, mults, _ = manual_solve(solver, a, b, max(11 * n_eigs, 50), n_workers=1)
    if not len(e):
        print(f'{args.key}: no eigenvalues found'); return 1
    eigs, tens = polish_eigs(solver, e, ltol=1e-14, bracket_rel_width=1e-9)
    eigs, tens, mults = eigs[:n_eigs], tens[:n_eigs], mults[:n_eigs]

    out = dict(experiment='exact_interior', key=args.key, n_basis=n_basis,
               seed=args.seed, n_found=len(eigs),
               eigs=[float(x) for x in eigs])
    msg = f'{args.key} exact-interior nb={n_basis} seed={args.seed} found={len(eigs)}/{n_eigs}'
    if entry.truth_fn is not None:
        ref = np.asarray(entry.truth_fn(n_eigs), dtype=float)
        k = min(len(ref), len(eigs))
        rel = np.abs(eigs[:k] - ref[:k]) / np.abs(ref[:k])
        dig = float(-np.log10(max(rel.max(), 1e-300)))
        out['analytic_min_digits'] = dig
        msg += f' TRUE_digits={dig:.1f}'
    _record(out)
    print(msg)
    return 0


def rellich_interior_block(basis, domain, bdry_pts, normals, x0=None):
    """A cheap stand-in for the interior collocation block.

    The interior block exists only to supply ``||u||_{L2(Omega)}`` so the tension
    ratio cannot be driven to zero by the trivial solution. It does **not** have
    to be the exact L2 norm -- any block whose induced norm is equivalent to it,
    with a bounded and slowly-varying ratio, puts the tension minima in the same
    places.

    For a Dirichlet domain the Rellich identity reduces to a single term
    (``rellich.rellich_gram_from_cauchy_data``, dir branch):

        G = (1/2 lam) * A_N^H diag(w * rN) A_N,     rN = (x - x0) . n

    so that ``||u||^2 = c^H G c = ||B c||^2`` with

        B(lam) = (2 lam)^{-1/2} diag(sqrt(w * rN)) A_N(lam).

    The point is that **G never needs forming**. Building it explicitly costs an
    O(n_quad * n^2) assembly plus an O(n^3) factorization at every lambda, which
    is what made the exact route unaffordable (>30 min versus 165s). ``B`` is
    just one normal-derivative Vandermonde -- the same cost as ``A_B``.

    "Inexact" in two senses, both harmless here:
      * it reuses the ordinary boundary quadrature rather than the dense graded
        one ``eigfun_integrals.boundary_quadrature`` constructs, so ``G`` is only
        approximated;
      * the identity in this form is exact for functions vanishing on the
        boundary. Trial functions in the search do not, but their boundary
        residual is precisely the quantity being minimized, so the error
        vanishes exactly where accuracy matters.

    Requires ``rN >= 0``, i.e. the domain star-shaped about ``x0``. Returns
    ``(block_fn, min_rN)`` so the caller can check.
    """
    import numpy as np
    from lappy.utils import complex_dot
    from lappy.eigfun_integrals import default_x0
    from lappy import mps

    if x0 is None:
        x0 = default_x0(domain)
    pts = bdry_pts.pts
    nrm = normals.pts if hasattr(normals, 'pts') else np.asarray(normals)
    rN = complex_dot(pts - x0, nrm)
    w = bdry_pts.wts if hasattr(bdry_pts, 'wts') else np.ones(len(pts))

    scale = np.sqrt(np.clip(w * rN, 0.0, None))
    A_N = mps.make_ddiff_vander(basis, bdry_pts, nrm)

    def block(lam):
        return (scale / np.sqrt(2.0 * lam))[:, None] * A_N(lam)

    return block, float(rN.min())


def cmd_inexact_rellich(args):
    """Compare the sampled interior block against the cheap Rellich block."""
    import time
    import numpy as np
    from lappy import bases, mps, MPSEigensolver
    from benchmarks.suite.domains import SUITE
    from common import manual_solve, polish_eigs, lambda_window

    np.random.seed(args.seed)
    entry = SUITE[args.key]
    dom = entry.domain()
    n_basis = args.n_basis or entry.n_basis
    n_eigs = entry.n_eigs

    basis = bases.make_default_basis(dom, n_basis)
    n_per_seg = mps.pts_per_seg(dom, basis, mult=2)
    bdry_pts = dom.bdry_pts(n_per_seg, weights=True)
    normals = dom.bdry_normals(n_per_seg)
    int_pts = dom.int_pts(method='random',
                          npts_rand=args.int_npts or max(2 * n_basis, 500),
                          rng=args.seed)
    basis = basis.to_normalized((bdry_pts, int_pts))

    block, min_rN = rellich_interior_block(basis, dom, bdry_pts, normals)
    star = min_rN >= 0
    print(f'{args.key}: min (x-x0).n = {min_rN:+.4f} '
          f'({"star-shaped, block is PSD" if star else "NOT star-shaped -- clipped"})')

    solver = MPSEigensolver(basis, bdry_pts, int_pts, rtol=1e-14, ttol=1e-3)
    a, b = lambda_window(dom, n_eigs)

    # cost comparison at a single lambda
    lam0 = 0.5 * (a + b)
    t = time.time(); solver.A_I(lam0); t_samp = time.time() - t
    t = time.time(); block(lam0); t_rel = time.time() - t
    print(f'  per-lambda interior block: sampled {t_samp*1e3:.1f} ms, '
          f'rellich {t_rel*1e3:.1f} ms')

    ref = np.asarray(entry.truth_fn(n_eigs), dtype=float) if entry.truth_fn else None

    def run(tag, A_I):
        s = MPSEigensolver(basis, bdry_pts, int_pts, rtol=1e-14, ttol=1e-3)
        s.A_I = A_I
        for k in [k for k in s.__dict__ if k.startswith('_icache_')]:
            del s.__dict__[k]
        t0 = time.time()
        e, m, _ = manual_solve(s, a, b, max(11 * n_eigs, 50), n_workers=1)
        if not len(e):
            print(f'  {tag:8s} no eigenvalues found'); return
        eigs, _ = polish_eigs(s, e, ltol=1e-14, bracket_rel_width=1e-9)
        eigs = eigs[:n_eigs]
        secs = time.time() - t0
        msg = f'  {tag:8s} found={len(eigs)}/{n_eigs}  {secs:5.0f}s'
        rec = dict(experiment='inexact_rellich', key=args.key, variant=tag,
                   n_basis=n_basis, seed=args.seed, seconds=secs,
                   eigs=[float(x) for x in eigs])
        if ref is not None:
            k = min(len(ref), len(eigs))
            rel = np.abs(eigs[:k] - ref[:k]) / np.abs(ref[:k])
            dig = float(-np.log10(max(rel.max(), 1e-300)))
            rec['analytic_min_digits'] = dig
            msg += f'  TRUE_digits={dig:5.1f}'
            missing = [f'{x:.3f}' for x in ref if np.min(np.abs(eigs - x)) > 1e-3]
            if missing:
                msg += f'  MISSING={missing}'
        _record(rec)
        print(msg)

    run('sampled', solver.A_I)
    run('rellich', block)
    return 0


def cmd_exact_polish(args):
    """Hybrid: sampled interior to *find* eigenvalues, exact Rellich Gram to
    *refine* them.

    Making the interior norm exact everywhere is unaffordable -- the basis
    depends on lambda, so the Gram must be rebuilt and eigendecomposed at every
    one of the thousands of evaluations a search makes (measured: >30 min for a
    domain that takes 165s normally).

    But the search only needs to locate a bracket, and the polish only needs
    ~24 golden-section evaluations per eigenvalue. Doing the cheap thing for
    location and the exact thing for refinement costs a few hundred
    evaluations, and it puts the exactness exactly where the final digits are
    decided.

    Reports true error for both refinements against the same closed form, so
    the comparison isolates the interior norm and nothing else.
    """
    import numpy as np
    from lappy import bases, mps, opt, MPSEigensolver
    from lappy.eigfun_integrals import boundary_quadrature
    from benchmarks.suite.domains import SUITE
    from common import manual_solve, polish_eigs, lambda_window

    np.random.seed(args.seed)
    entry = SUITE[args.key]
    dom = entry.domain()
    n_basis = args.n_basis or entry.n_basis
    n_eigs = entry.n_eigs
    if entry.truth_fn is None:
        print(f'{args.key}: needs a closed form for this comparison'); return 1

    basis = bases.make_default_basis(dom, n_basis)
    n_per_seg = mps.pts_per_seg(dom, basis, mult=2)
    bdry_pts = dom.bdry_pts(n_per_seg)
    int_pts = dom.int_pts(method='random', npts_rand=max(2 * n_basis, 500),
                          rng=args.seed)
    basis = basis.to_normalized((bdry_pts, int_pts))
    solver = MPSEigensolver(basis, bdry_pts, int_pts, rtol=1e-14, ttol=1e-3)

    a, b = lambda_window(dom, n_eigs)
    e, mults, _ = manual_solve(solver, a, b, max(11 * n_eigs, 50), n_workers=1)
    if not len(e):
        print(f'{args.key}: no eigenvalues found'); return 1

    ref = np.asarray(entry.truth_fn(n_eigs), dtype=float)

    def digits(vals):
        k = min(len(ref), len(vals))
        rel = np.abs(np.asarray(vals)[:k] - ref[:k]) / np.abs(ref[:k])
        return float(-np.log10(max(rel.max(), 1e-300)))

    # (a) the normal sampled polish
    sampled, _ = polish_eigs(solver, e, ltol=1e-14, bracket_rel_width=1e-9)
    sampled = sampled[:n_eigs]

    # (b) the same coarse locations, refined against the exact interior norm
    rd = boundary_quadrature(dom, b)   # no basis needed; sizes itself from geometry+lam

    def sigma_exact(lam):
        A_B = solver.A_B(lam)
        L = exact_interior_factor(basis, lam, rd)
        return mps.tensions(A_B, L, rtol=1e-14)[0]

    exact = []
    for eig in e[:n_eigs]:
        lo, hi = eig * (1 - 1e-9), eig * (1 + 1e-9)
        exact.append(opt.golden_search(sigma_exact, lo, hi,
                                       tol=1e-14 * max(abs(eig), 1.0))[0])
    exact = np.array(exact)

    d_s, d_e = digits(sampled), digits(exact)
    _record(dict(experiment='exact_polish', key=args.key, n_basis=n_basis,
                 seed=args.seed, sampled_digits=d_s, exact_digits=d_e,
                 eigs_sampled=[float(x) for x in sampled],
                 eigs_exact=[float(x) for x in exact]))
    print(f'{args.key} nb={n_basis} seed={args.seed}  '
          f'sampled-polish {d_s:5.1f} digits   exact-polish {d_e:5.1f} digits   '
          f'delta {d_e - d_s:+.1f}')
    return 0


def cmd_mid_support(args):
    """Test the section-2 mechanism: does support BETWEEN sharp corners help?

    `make_default_basis` gives multi-singular-corner domains a Fourier--Bessel
    block plus a `FundamentalBasis.by_corners` block -- and `by_corners`
    clusters its sources at the corners, so the whole basis is corner-localized
    and the middle of the domain is represented only by slowly-decaying tails.
    Here we swap the corner-clustered fundamental block for `by_boundary`,
    which distributes sources along an offset boundary and so actually covers
    the region between the corners.
    """
    import numpy as np
    from lappy import bases, mps, asymp, MPSEigensolver
    from lappy.symmetry import (SymmetrizedBasis, prune_columns,
                                fundamental_bdry_pts, fundamental_int_pts)
    from benchmarks.suite.domains import SUITE
    from common import manual_solve, polish_eigs, lambda_window
    from certify import certify_solver

    np.random.seed(args.seed)
    entry = SUITE[args.key]
    dom = entry.domain()
    n_basis, n_eigs = args.n_basis or entry.n_basis, entry.n_eigs

    n_fs = int(round(args.fs_frac * n_basis))
    n_fb = n_basis - n_fs
    fb_orders = bases.fb_corner_orders(dom, n_fb)
    fb = bases.FourierBesselBasis.from_domain(dom, fb_orders)
    n_seg = len(dom.bdry.segments)
    # by_boundary wants a per-segment count array, not a scalar
    per_seg = np.full(n_seg, max(n_fs // n_seg, 1), dtype=int)
    fs = bases.FundamentalBasis.by_boundary(dom, per_seg, d=args.fs_d)
    basis = fb + fs

    n_per_seg = mps.pts_per_seg(dom, basis, mult=2)
    bdry_pts = dom.bdry_pts(n_per_seg)
    int_pts = dom.int_pts(method='random', npts_rand=max(2 * n_basis, 500))
    basis = basis.to_normalized((bdry_pts, int_pts))
    solver = MPSEigensolver(basis, bdry_pts, int_pts, rtol=1e-14, ttol=1e-3)

    a, b = lambda_window(dom, n_eigs)
    e, mults, _ = manual_solve(solver, a, b, max(11 * n_eigs, 50), n_workers=1)
    if not len(e):
        print(f'{args.key}: no eigenvalues found'); return 1
    eigs, tens = polish_eigs(solver, e, ltol=1e-14, bracket_rel_width=1e-9)
    eigs, tens, mults = eigs[:n_eigs], tens[:n_eigs], mults[:n_eigs]
    recs = certify_solver(solver, dom, eigs, mult=mults, verbose=False)
    eps = np.array([r['eps'] for r in recs])
    dig = float(-np.log10(eps.max()))
    _record(dict(experiment='mid_support', key=args.key, fs_frac=args.fs_frac,
                 fs_d=args.fs_d, n_basis=n_basis, seed=args.seed,
                 min_digits=dig, n_found=len(eigs),
                 eigs=[float(x) for x in eigs]))
    print(f'{args.key} by_boundary fs_frac={args.fs_frac} d={args.fs_d} '
          f'nb={n_basis} certified_digits={dig:.1f} found={len(eigs)}/{n_eigs}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    a = sub.add_parser('rank_vs_p')
    a.add_argument('--n-basis', type=int, default=160)
    a.set_defaults(func=cmd_rank_vs_p)

    b = sub.add_parser('rank_curve')
    b.add_argument('key')
    b.add_argument('--sizes', default='60,120,240,320')
    b.set_defaults(func=cmd_rank_curve)

    d = sub.add_parser('exact_interior')
    d.add_argument('key')
    d.add_argument('--n-basis', type=int, default=None)
    d.add_argument('--seed', type=int, default=0)
    d.set_defaults(func=cmd_exact_interior)

    ir = sub.add_parser('inexact_rellich')
    ir.add_argument('key')
    ir.add_argument('--n-basis', type=int, default=None)
    ir.add_argument('--seed', type=int, default=0)
    ir.add_argument('--int-npts', type=int, default=None,
                    help='starve the sampled block to expose its seed sensitivity')
    ir.set_defaults(func=cmd_inexact_rellich)

    ep = sub.add_parser('exact_polish')
    ep.add_argument('key')
    ep.add_argument('--n-basis', type=int, default=None)
    ep.add_argument('--seed', type=int, default=0)
    ep.set_defaults(func=cmd_exact_polish)

    c = sub.add_parser('mid_support')
    c.add_argument('key')
    c.add_argument('--fs-frac', type=float, default=0.5)
    c.add_argument('--fs-d', type=float, default=1.0)
    c.add_argument('--n-basis', type=int, default=None)
    c.add_argument('--seed', type=int, default=0)
    c.set_defaults(func=cmd_mid_support)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
