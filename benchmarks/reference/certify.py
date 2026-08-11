"""A posteriori *certified* error bars for MPS eigenvalues (Moler--Payne).

Everything else in this directory reports the GSVD tension and converts it to
a digit count with the folklore rule ``digits ~ -log10(sigma) - 1``. That rule
is a heuristic: the tension is a discrete, collocation-point-dependent ratio,
and the constant relating it to the eigenvalue error is nowhere justified.
For a table of *reference values* that other code will be checked against,
that is not good enough -- we want a bound, not a vibe.

The Moler--Payne / Fox--Henrici--Moler bound supplies one. Let ``u`` satisfy
``Delta u + lam u = 0`` exactly in ``Omega`` (every MPS basis function does,
so every linear combination does too), let ``g = u`` restricted to
``dOmega``, and let ``v`` be the harmonic extension of ``g``. Then
``w = u - v`` lies in ``H_0^1``, and

    Delta w + lam w = -lam v.

Expanding ``w = sum a_k phi_k`` in Dirichlet eigenfunctions gives
``sum a_k (lam - lam_k) phi_k = -lam v``, hence

    min_k |lam - lam_k| * ||w|| <= lam * ||P v|| <= lam * ||v||.

By the maximum principle ``||v||_inf <= ||g||_inf``, so
``||v||_{L2(Omega)} <= ||g||_inf sqrt(|Omega|)``, and
``||w|| >= ||u|| - ||v||``. Writing

    eps = sqrt(|Omega|) * ||u||_{Linf(dOmega)} / ||u||_{L2(Omega)}

we get the computable, rigorous statement

    min_k |lam - lam_k| <= lam * eps / (1 - eps)        (eps < 1).

So ``eps`` *is* the certified relative error, and ``-log10(eps)`` is an
honest digit count. Note it uses the ``Linf`` boundary norm, which is
strictly larger than the ``L2``-flavoured GSVD tension -- expect certified
digit counts to come out somewhat *below* the tension heuristic's.

Caveats, stated plainly:

* The bound certifies that *some* Dirichlet eigenvalue lies within the
  stated distance. It does not by itself certify *which* one. Index
  assignment here comes from separate evidence: no gap in the computed
  sequence, agreement with the two-term Weyl count, and (for symmetric
  domains) the per-sector labelling, which makes multiplicity an output.
* It is a bound on the exact arithmetic quantity; floating-point evaluation
  of ``u`` is not itself certified. At the 1e-13 level that matters, so
  treat the last reported digit as indicative.
* ``||u||_inf`` on the boundary is estimated by dense sampling. ``u`` is
  analytic on the open edges but has corner singularities of the form
  ``r^p`` with ``p < 1`` at reentrant corners, where the true sup can sit
  between samples. ``boundary_sup`` therefore samples on a graded mesh that
  clusters at corners, and reports the sampling-refinement drift so a
  under-resolved sup is visible rather than silent.

Both norms in ``eps`` are now boundary integrals
--------------------------------------------------

``||u||_{L2(Omega)}`` used to come from interior cubature, which is the
expensive and fragile half of this file: the mesh rule needs a triangulation
(minutes on a spiral, unavailable on some curved boundaries) and its Monte
Carlo fallback is deflated by three standard errors, which throws away digits
by construction. With ``MPSEigensolver``'s corner-adapted boundary quadrature
(``lappy.eigfun_integrals``) the same norm is a *boundary* integral via the
Rellich identity, sharing the node set the solver already built.

``eps`` is scale-invariant, so this does not change the certified number where
cubature was accurate -- it changes what it costs, and it removes the cases
where cubature was the limiting factor. The norm is still used as a
**conservative under-estimate**, in two ways:

* the identity holds for *every* reference point ``x0``, so the norm is
  computed at several and the smallest is taken;
* the spread across those ``x0`` is pure quadrature error, and is subtracted
  from the (squared) norm before it enters the denominator.

If that spread is large -- a near-slit corner where the rule cannot reach its
target precision -- ``certify_solver`` falls back to cubature and keeps
whichever of the two lower bounds is larger, so a weak boundary rule can never
make the reported bound *worse* than the old path's.

One caveat the ``x0`` spread does **not** cover: the Rellich identity is exact
for a *true* Dirichlet eigenfunction, and an MPS approximant solves Helmholtz
exactly but has a small nonzero trace ``g`` on the boundary, so the identity
drops a term involving ``g``. That term is second order in ``g`` -- measured on
``stadium``, the worst case in the suite, ``sup|u| = 4e-4 .. 1e-3`` against
``|G-1| = 9e-9 .. 4e-6`` -- and it enters ``eps = sqrt(area) sup|u| / ||u||``
only through the denominator, i.e. it perturbs the certified bound by a
*relative* amount of order ``eps`` itself. It cannot move a digit count that
the numerator has already fixed.
"""
import numpy as np

from lappy.cache import clear_instance_caches

from lappy.geometry import PointSet
from lappy import cubature
from lappy.eigfun_integrals import eigfun_cauchy_data, gram


# ---------------------------------------------------------------------------
# the two norms
# ---------------------------------------------------------------------------

def boundary_sup(domain, u, n_per_seg=400, grade=3.0, refine=True):
    """Estimate ``||u||_{Linf(dOmega)}`` by graded sampling of each segment.

    Nodes on each segment are ``t = s^grade`` and ``1 - s^grade`` mirrored
    about the midpoint, clustering at both endpoints (where corner
    singularities live). With ``refine``, the whole estimate is repeated at
    double density and the relative drift is returned, so an under-resolved
    sup shows up as a large ``drift`` rather than a silently low number.

    Returns ``(sup, drift)``.
    """
    def sample(n):
        vals = []
        for seg in domain.bdry.segments:
            s = np.linspace(0.0, 1.0, n // 2 + 1)[1:]      # drop the exact corner
            t = np.concatenate([0.5 * s ** grade, 1.0 - 0.5 * s ** grade])
            t = np.unique(np.clip(t, 1e-14, 1 - 1e-14))
            vals.append(np.abs(u(PointSet(seg.p(t)))).ravel())
        return np.max(np.concatenate(vals))

    hi = sample(n_per_seg)
    if not refine:
        return hi, np.nan
    hi2 = sample(2 * n_per_seg)
    top = max(hi, hi2)
    drift = abs(hi2 - hi) / top if top > 0 else 0.0
    return top, drift


_MESH_CACHE = {}


def interior_l2(domain, u, deg=10, mesh_kwargs=None, fallback_npts=200000,
                cache_key=None, strict=False):
    """``||u||_{L2(Omega)}`` by cubature, as a *conservative under-estimate*.

    ``eps`` has this quantity in its denominator, so for the Moler--Payne
    bound to stay a bound we must not over-estimate it. The mesh rule is
    used when available (exact for polygons; for curved boundaries the
    polygonal mesh under-resolves the domain slightly, which errs in the
    safe direction since it *omits* area). Monte Carlo is the fallback and
    is deflated by three standard errors.

    The mesh is cached per domain: it costs far more to build than to
    evaluate, and certification evaluates it once per eigenvalue.

    Returns ``(norm, method)``.
    """
    if mesh_kwargs is None:
        mesh_kwargs = {}
    key = (id(domain) if cache_key is None else cache_key, deg)
    if key not in _MESH_CACHE:
        try:
            # Polygon.int_pts overrides the base signature (mesh_size instead
            # of mesh_kwargs), so only forward extras the callee accepts.
            import inspect
            sig = inspect.signature(domain.int_pts).parameters
            extra = {k: v for k, v in mesh_kwargs.items() if k in sig}
            if 'mesh_kwargs' in sig and mesh_kwargs and not extra:
                extra = {'mesh_kwargs': mesh_kwargs}
            _MESH_CACHE[key] = domain.int_pts(method='mesh', weights=True,
                                              kind='dunavant', deg=deg, **extra)
        except Exception as exc:
            if strict:
                raise
            print(f'    [interior_l2] mesh cubature unavailable '
                  f'({type(exc).__name__}: {exc}); falling back to Monte Carlo')
            _MESH_CACHE[key] = None

    pts = _MESH_CACHE[key]
    if pts is not None:
        vals = np.abs(u(pts)).ravel() ** 2
        return float(np.sqrt(np.sum(pts.wts * vals))), f'mesh/dunavant{deg}'

    pts = domain.int_pts(method='random', weights=True, npts_rand=fallback_npts)
    vals = np.abs(u(pts)).ravel() ** 2
    n = len(vals)
    mean, sd = vals.mean(), vals.std(ddof=1)
    conservative = max(mean - 3 * sd / np.sqrt(n), 0.0)     # 3-sigma safe side
    return float(np.sqrt(domain.area * conservative)), 'monte-carlo(-3sigma)'


def x0_probes(bq, n=3):
    """Alternative Rellich reference points, from the node set's bounding box.

    The identity ``<u,v> = (1/2lam) int (r.N) dNu dNv`` holds for every ``x0``,
    so evaluating the Gram at several is a free, method-internal error estimate:
    any variation is quadrature error. Deterministic, and deliberately spread
    across the box -- probes clustered near ``bq.x0`` would understate it.
    """
    P = np.asarray(bq.pts)
    cx = 0.5 * (P.real.min() + P.real.max())
    cy = 0.5 * (P.imag.min() + P.imag.max())
    w = max(P.real.max() - P.real.min(), 1e-300)
    h = max(P.imag.max() - P.imag.min(), 1e-300)
    cand = [cx + 1j * cy,
            (cx + 0.31 * w) + 1j * (cy + 0.17 * h),
            (cx - 0.40 * w) + 1j * (cy + 0.23 * h),
            (cx + 0.44 * w) - 1j * (cy + 0.37 * h)]
    return cand[:n]


def boundary_l2(solver, lam, mult=1, n_probes=3):
    """``||u_j||_{L2(Omega)}`` for each column of the cluster at ``lam``, as a
    conservative under-estimate, from the solver's own boundary quadrature.

    Requires ``solver.bdry_quad`` (``MPSEigensolver.from_domain(orthonorm=True)``
    or an explicit ``bdry_quad=``). The coefficients are taken *as the solver
    returns them* -- normally already L2-orthonormalized, so the diagonal of the
    Gram should be 1 -- but nothing here assumes that: the norms are read off the
    Gram, so the routine is equally correct if orthonormalization was skipped or
    fell back (a deficient cluster Gram), and the deviation from 1 is itself
    reported.

    Returns ``(norms, info)`` with ``info`` carrying the ``x0`` spread (the
    quadrature's own error estimate), the off-diagonal residual (how orthogonal
    the cluster actually is), and the node set's size and achieved precision.
    """
    bq = solver.bdry_quad
    if bq is None:
        raise ValueError('solver has no bdry_quad; cannot certify from the boundary')
    coef = solver.eigenfunction_coef(lam, mult=int(mult))
    ed = eigfun_cauchy_data(solver.basis, lam, coef, bq)

    G = gram(ed, lam, bq)
    diags = [np.diag(G)] + [np.diag(gram(ed, lam, bq, x0=x0))
                            for x0 in x0_probes(bq, n_probes)]
    D = np.array(diags)
    lo, hi = D.min(axis=0), D.max(axis=0)
    spread = hi - lo

    # Conservative: smallest of the x0 estimates, minus the spread between them.
    # Both quantities are squared norms, so the subtraction happens there.
    norms = np.sqrt(np.clip(lo - spread, 0.0, None))
    off = float(np.abs(G - np.diag(np.diag(G))).max()) if mult > 1 else 0.0
    info = dict(l2_method='rellich-boundary',
                l2_spread=[float(x) for x in spread],
                gram_diag=[float(x) for x in np.diag(G)],
                gram_offdiag=off,
                bq_nodes=int(len(bq.pts)),
                bq_precision=float(bq.sizing_precision))
    return norms, info


# ---------------------------------------------------------------------------
# the bound
# ---------------------------------------------------------------------------

def moler_payne(domain, u, lam, n_per_seg=400, deg=10, verbose=False,
                l2=None, l2_method=None, l2_info=None):
    """Certified error bar for ``lam`` given approximate eigenfunction ``u``.

    Returns a dict with ``eps`` (certified *relative* error), ``abs_bound``
    (certified absolute error ``lam*eps/(1-eps)``), ``digits``
    (``-log10(eps)``), and the ingredients.

    ``l2`` supplies a precomputed ``||u||_{L2}`` (see ``boundary_l2``); without
    it the norm comes from interior cubature as before.
    """
    sup, drift = boundary_sup(domain, u, n_per_seg=n_per_seg)
    if l2 is None:
        l2, method = interior_l2(domain, u, deg=deg)
    else:
        l2, method = float(l2), (l2_method or 'supplied')
    area = domain.area

    # Inflate the sampled sup by the observed refinement drift: the true sup
    # can sit between samples (corner singularities are the usual culprit),
    # and the drift is our only measure of how much room is left.
    sup = sup * (1.0 + 2.0 * (drift if np.isfinite(drift) else 0.0))

    eps = np.sqrt(area) * sup / l2 if l2 > 0 else np.inf
    abs_bound = lam * eps / (1 - eps) if eps < 1 else np.inf
    out = dict(lam=lam, eps=eps, abs_bound=abs_bound,
               rel_bound=abs_bound / lam if np.isfinite(abs_bound) else np.inf,
               digits=-np.log10(eps) if eps > 0 else np.inf,
               bdry_sup=sup, bdry_sup_drift=drift, int_l2=l2,
               l2_method=method, area=area)
    if l2_info:
        out.update(l2_info)
        out['l2_method'] = method
    if verbose:
        print(f'  lam={lam:.15f}  eps={eps:.3e}  |dlam| <= {abs_bound:.3e}  '
              f'({out["digits"]:.1f} certified digits)')
        print(f'    sup|u|_bdry={sup:.3e} (drift {drift:.1e})  '
              f'||u||_L2={l2:.3e} [{method}]  sqrt(area)={np.sqrt(area):.4f}')
    return out


SPREAD_TOL = 1e-8
"""Relative ``x0``-spread above which the boundary norm is cross-checked against
cubature. The boundary rule reaches ~1e-13 on almost every suite domain; the
exceptions are near-slit corners (``chevron_2_3``, ``chevron_2_4``) and the
spirals, where ``boundary_quadrature`` warns that it fell short. There the
cross-check costs a mesh build and can only help: both numbers are lower bounds
on ``||u||``, so the larger one is used."""


def certify_solver(solver, domain, eigs, mult=None, n_per_seg=400, deg=10,
                   verbose=True, l2_source='auto', spread_tol=SPREAD_TOL):
    """Run ``moler_payne`` for each eigenvalue of one solver.

    ``mult[i]`` (default 1) is how many eigenfunctions to extract at
    ``eigs[i]``; each column is certified separately and the *worst* is
    reported, since the bound must hold for the whole cluster.

    ``l2_source`` selects the ``||u||_{L2}`` used in the denominator:

    ``'auto'``      boundary (Rellich) when the solver carries a ``bdry_quad``,
                    cubature otherwise -- and cubature *as well*, keeping the
                    larger lower bound, when the boundary rule's own ``x0``
                    spread exceeds ``spread_tol``;
    ``'boundary'``  boundary only (raises without a ``bdry_quad``);
    ``'cubature'``  interior cubature only -- the pre-orthonormalization path,
                    kept so old results stay reproducible.
    """
    if mult is None:
        mult = np.ones(len(eigs), dtype=int)
    have_bq = getattr(solver, 'bdry_quad', None) is not None
    if l2_source == 'boundary' and not have_bq:
        raise ValueError("l2_source='boundary' but the solver has no bdry_quad")
    use_bdry = l2_source == 'boundary' or (l2_source == 'auto' and have_bq)

    out = []
    for lam, m in zip(eigs, mult):
        m = int(m)
        # orthonorm=True is what makes the eigenfunction's scale meaningful.
        # `eps` is scale-invariant, so this does not by itself move the bound --
        # it is what lets `boundary_l2` read the norms off a Gram that should be
        # the identity, and makes any departure from it a diagnostic.
        ufun = solver.eigenfunction(lam, mult=m, orthonorm=use_bdry)
        norms, info = (boundary_l2(solver, lam, m) if use_bdry else (None, None))
        best = None
        for j in range(m):
            def uj(pts, j=j):
                return ufun(pts)[..., j]
            l2 = None if norms is None else norms[j]
            l2_info = None
            if info is not None:
                l2_info = dict(info, l2_spread=info['l2_spread'][j],
                               gram_diag=info['gram_diag'][j])
                rel_spread = info['l2_spread'][j] / max(info['gram_diag'][j], 1e-300)
                if l2_source == 'auto' and rel_spread > spread_tol:
                    # The boundary rule is not converged here; both estimates are
                    # lower bounds on ||u||, so take the larger.
                    l2_cub, meth = interior_l2(domain, uj, deg=deg)
                    l2_info['l2_cubature'] = float(l2_cub)
                    if l2_cub > l2:
                        l2, l2_info['l2_method'] = l2_cub, f'{meth} (boundary spread ' \
                                                           f'{rel_spread:.1e})'
                    else:
                        l2_info['l2_method'] = f'rellich-boundary (> {meth})'
            rec = moler_payne(domain, uj, lam, n_per_seg=n_per_seg, deg=deg,
                              l2=l2, l2_method=(l2_info or {}).get('l2_method'),
                              l2_info=l2_info)
            if best is None or rec['eps'] > best['eps']:
                best = rec
        best['mult'] = m
        out.append(best)
        if verbose:
            print(f'  lam={lam:.15f}  mult={int(m)}  eps={best["eps"]:.3e}  '
                  f'|dlam|<={best["abs_bound"]:.3e}  '
                  f'certified digits {best["digits"]:.1f}  '
                  f'(sup drift {best["bdry_sup_drift"]:.1e})')
    return out


def certify_sym(solvers, domain, eigs, sectors, **kwargs):
    """Certify a symmetry-sector solve: each eigenvalue is certified using the
    solver of the sector it was found in.

    The projected eigenfunction is a genuine Helmholtz solution on the *whole*
    domain (the projection is a sum over isometries of the domain), and
    ``boundary_sup``/``interior_l2`` integrate over the whole domain, so the
    bound applies verbatim -- no symmetry-specific correction is needed.
    """
    out = []
    for lam, sec in zip(eigs, sectors):
        solver = solvers[tuple(sec)]
        rec = certify_solver(solver, domain, [lam], **kwargs)[0]
        rec['sector'] = tuple(sec)
        out.append(rec)
    return out


def summarize_l2(records):
    """Compact, JSON-safe summary of how ``||u||_L2`` was obtained, for the
    result files: which method(s), the worst ``x0`` spread and off-diagonal
    Gram residual (both None when the boundary path was not used), and the node
    set's size and achieved precision."""
    spreads = [float(r['l2_spread']) for r in records if 'l2_spread' in r]
    offs = [float(r['gram_offdiag']) for r in records if 'gram_offdiag' in r]
    bq = next((r for r in records if 'bq_nodes' in r), None)
    return dict(
        l2_methods=sorted({str(r.get('l2_method')) for r in records}),
        l2_spread_max=max(spreads) if spreads else None,
        gram_offdiag_max=max(offs) if offs else None,
        bq_nodes=(bq['bq_nodes'] if bq else None),
        bq_precision=(bq['bq_precision'] if bq else None))


def report_certified(name, eigs, records, sectors=None, ref=None):
    print(f'\n=== {name}: certified ===')
    head = f"{'eigenvalue':>24}  {'eps (rel err bound)':>20}  {'digits':>7}"
    if sectors is not None:
        head += '  sector'
    if ref is not None:
        head += f"  {'|diff| vs old ref':>18}"
    print(head)
    for i, (lam, rec) in enumerate(zip(eigs, records)):
        line = f'{lam:24.15f}  {rec["eps"]:20.3e}  {rec["digits"]:7.1f}'
        if sectors is not None:
            line += f'  {sectors[i]}'
        if ref is not None and i < len(ref):
            line += f'  {abs(lam - ref[i]):18.3e}'
        print(line)
    worst = max(r['eps'] for r in records)
    print(f'worst certified relative error: {worst:.3e}  '
          f'({-np.log10(worst):.1f} digits)')
