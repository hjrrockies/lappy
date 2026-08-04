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
"""
import numpy as np

from lappy.cache import clear_instance_caches

from lappy.geometry import PointSet
from lappy import cubature


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


# ---------------------------------------------------------------------------
# the bound
# ---------------------------------------------------------------------------

def moler_payne(domain, u, lam, n_per_seg=400, deg=10, verbose=False):
    """Certified error bar for ``lam`` given approximate eigenfunction ``u``.

    Returns a dict with ``eps`` (certified *relative* error), ``abs_bound``
    (certified absolute error ``lam*eps/(1-eps)``), ``digits``
    (``-log10(eps)``), and the ingredients.
    """
    sup, drift = boundary_sup(domain, u, n_per_seg=n_per_seg)
    l2, method = interior_l2(domain, u, deg=deg)
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
    if verbose:
        print(f'  lam={lam:.15f}  eps={eps:.3e}  |dlam| <= {abs_bound:.3e}  '
              f'({out["digits"]:.1f} certified digits)')
        print(f'    sup|u|_bdry={sup:.3e} (drift {drift:.1e})  '
              f'||u||_L2={l2:.3e} [{method}]  sqrt(area)={np.sqrt(area):.4f}')
    return out


def certify_solver(solver, domain, eigs, mult=None, n_per_seg=400, deg=10,
                   verbose=True):
    """Run ``moler_payne`` for each eigenvalue of one solver.

    ``mult[i]`` (default 1) is how many eigenfunctions to extract at
    ``eigs[i]``; each column is certified separately and the *worst* is
    reported, since the bound must hold for the whole cluster.
    """
    if mult is None:
        mult = np.ones(len(eigs), dtype=int)
    out = []
    for lam, m in zip(eigs, mult):
        ufun = solver.eigenfunction(lam, mult=int(m), orthonorm=False)
        best = None
        for j in range(int(m)):
            def uj(pts, j=j):
                return ufun(pts)[..., j]
            rec = moler_payne(domain, uj, lam, n_per_seg=n_per_seg, deg=deg)
            if best is None or rec['eps'] > best['eps']:
                best = rec
        best['mult'] = int(m)
        out.append(best)
        # Each eigenfunction is evaluated over the full degree-`deg` cubature
        # mesh and the refined boundary mesh, and those Vandermondes land in
        # per-instance LRU caches sized in entries (128 for
        # NormalizedBasis.norms, 256 for _tensions_scalar). Certifying ten
        # eigenvalues across several sectors that way exhausts memory on a 16GB
        # machine. Nothing here is reused at the next lambda, so drop it.
        clear_instance_caches(solver)
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
