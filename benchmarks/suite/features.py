"""Geometric feature derivation for the benchmark domain suite.

Everything here is computed *from a built domain*, never hand-recorded, so the
suite's taxonomy table cannot silently drift away from the geometry it claims to
describe.  ``benchmarks/suite/domains.py`` declares a set of tags per entry by
hand and then cross-checks them against :func:`derive_tags`; a mismatch is an
error.

The central quantity is the **corner exponent**

    p = pi / gamma

for a corner of interior angle ``gamma``.  Local solutions of the Helmholtz
equation near such a corner behave like ``r**(m*p)``, and ``p`` indexes the two
opposite ways a corner makes life hard for the method of particular solutions:

* ``p`` a positive integer -- the corner is *regular*.  The Fourier--Bessel
  expansion is analytic there and nothing special is needed.
* ``p < 1`` (reentrant, ``gamma > pi``) -- the leading exponent is small, so any
  smooth approximant converges only algebraically.  The classic MPS accuracy
  problem; worst as ``gamma -> 2 pi`` (slit, ``p = 1/2``).
* ``p >> 1`` (sharp, ``gamma -> 0``) -- Bessel orders ``m*p`` blow up, and
  evaluation gets expensive long before accuracy becomes the binding
  constraint.  A cost problem, not an accuracy problem.
"""
import numpy as np

from lappy.geometry import LineSegment, corner_branch_cut_rays

# A corner is "sharp" once its exponent reaches this *and* it is singular.
# The regularity qualifier matters: a 45-degree corner has p = 4 but the
# Fourier--Bessel expansion there is analytic, so it costs nothing.  What hurts
# is a large *non-integer* exponent -- chevron's ~11 degree corners at
# p ~ 15.9, iso_tri(16)'s apex at p ~ 25.2.
SHARP_P = 4.0
# relative tolerance for calling an exponent an integer (i.e. the corner regular)
INT_TOL = 1e-9
# curvature values closer than this (absolute) count as continuous
CURV_TOL = 1e-6


def corner_data(domain):
    """Interior angles and exponents at the genuine corners of ``domain``.

    Junctions between boundary segments that meet tangentially are *not*
    corners and are excluded (see :func:`smooth_junctions`).

    Returns
    -------
    angles : ndarray
        Interior angle ``gamma`` at each corner, radians.
    p : ndarray
        Corner exponent ``pi / gamma``.
    """
    bdry = domain.bdry
    idx = np.asarray(bdry.corner_idx, dtype=int)
    angles = np.asarray(bdry.int_angles, dtype=float)[idx]
    return angles, np.pi / angles


def is_regular(p):
    """Elementwise: is this corner exponent a positive integer?"""
    p = np.asarray(p, dtype=float)
    return np.abs(p - np.round(p)) <= INT_TOL * np.maximum(1.0, np.abs(p))


def _curvature(seg, tau, h=1e-6):
    """Signed curvature of ``seg`` at parameter ``tau``, by differencing the
    unit tangent with respect to arc length.

    Segments are arc-length reparameterized onto ``[0, 1]``, so ``ds = len *
    dtau``.  ``LineSegment`` is special-cased to avoid differencing a constant.
    """
    if isinstance(seg, LineSegment):
        return 0.0
    lo = min(max(tau, 0.0), 1.0 - h)
    t0 = np.atleast_1d(seg.T(lo))[0]
    t1 = np.atleast_1d(seg.T(lo + h))[0]
    return float(np.angle(t1 / t0) / (seg.len * h))


def smooth_junctions(domain):
    """Tangentially-continuous junctions between distinct boundary segments.

    These are where a curvature discontinuity can hide: the boundary is C^1 but
    not C^2, which is neither a corner the Fourier--Bessel basis can be aimed at
    nor the smooth boundary the fundamental-solution basis wants.  The stadium's
    line/semicircle joins are the canonical case.

    Returns a list of ``(junction_index, kappa_before, kappa_after)``.
    """
    segs = domain.bdry.segments
    n = len(segs)
    if n < 2:
        return []
    corners = set(int(i) for i in domain.bdry.corner_idx)
    out = []
    for j in range(n):
        if j in corners:
            continue
        prev, cur = segs[(j - 1) % n], segs[j]
        out.append((j, _curvature(prev, 1.0), _curvature(cur, 0.0)))
    return out


def symmetry_of(family, params):
    """The symmetry group for a suite entry, or ``None``.

    ``lappy.symmetry.domain_symmetry`` is keyed by family name, so this takes
    the registry's ``family``/``params`` rather than a built domain.  Families
    it does not know about get ``None``.
    """
    from lappy.symmetry import domain_symmetry
    try:
        return domain_symmetry(family, **params)
    except (KeyError, ValueError, TypeError):
        return None


def describe(domain, family=None, params=None):
    """Full derived feature record for one domain."""
    angles, p = corner_data(domain)
    reg = is_regular(p)
    sharp = (~reg) & (p >= SHARP_P)
    segs = domain.bdry.segments
    n_curved = sum(0 if isinstance(s, LineSegment) else 1 for s in segs)

    junc = smooth_junctions(domain)
    n_curv_disc = sum(1 for _, k0, k1 in junc if abs(k0 - k1) > CURV_TOL)

    rays = corner_branch_cut_rays(domain)
    n_blocked = int(np.count_nonzero(np.isnan(rays)))

    grp = symmetry_of(family, params or {}) if family else None

    area, per = float(domain.area), float(domain.perimeter)
    return dict(
        n_corners=len(p),
        angles_deg=np.degrees(angles),
        p=p,
        n_regular=int(np.count_nonzero(reg)),
        n_singular=int(np.count_nonzero(~reg)),
        n_reentrant=int(np.count_nonzero(p < 1.0)),
        n_sharp=int(np.count_nonzero(sharp)),
        max_sharp_p=float(p[sharp].max()) if np.any(sharp) else 0.0,
        min_p=float(p.min()) if len(p) else float('nan'),
        max_p=float(p.max()) if len(p) else float('nan'),
        n_segments=len(segs),
        n_curved=n_curved,
        n_smooth_junctions=len(junc),
        n_curvature_discont=n_curv_disc,
        n_blocked_branch_cuts=n_blocked,
        area=area,
        perimeter=per,
        diameter=float(domain.diameter),
        inradius=float(domain.inradius),
        # dimensionless shape measures, both 1 for a disk and growing with
        # thinness. ``slenderness`` is the readable one: it is the aspect ratio
        # of the equivalent rectangle-ish shape (2 for a 2:1 ellipse, 8 for a
        # 1x8 rectangle), so it is what the ``elongated`` tag keys off.
        slenderness=float(domain.diameter / (2.0 * domain.inradius)),
        aspect=float(domain.diameter ** 2 / area),
        isoperimetric=float(per ** 2 / (4 * np.pi * area)),
        sym_name=grp.name if grp is not None else None,
        sym_order=grp.order if grp is not None else 1,
    )


# --- tags -------------------------------------------------------------------
# Closed vocabulary.  Tags split into those derivable from the geometry alone
# (checked automatically) and those that are statements about the *spectrum* or
# about known solver behaviour, which no amount of staring at the boundary will
# reveal -- those are declared in the registry and not cross-checked.

GEOMETRIC_TAGS = frozenset({
    'regular_corners', 'singular_corners', 'reentrant', 'sharp', 'many_singular',
    'many_reentrant', 'smooth', 'curvature_discont', 'mixed_smooth_corner',
    'elongated', 'no_symmetry', 'branch_cut_polyline',
})

# NOTE on ``no_symmetry``: this tag means *lappy.symmetry.domain_symmetry has no
# entry for this family*, which is not the same as the domain being
# asymmetric.  ``rect``, ``eq_tri``, ``disk``, ``disk_sector`` and
# ``parallelogram`` all have obvious symmetry groups that are simply not
# registered.  The registry declares the mathematical truth in
# ``SuiteDomain.symmetric`` and ``report`` lists the disagreements as a
# concrete to-do list for ``lappy/symmetry.py``.

SPECTRAL_TAGS = frozenset({
    'exact_multiplicity', 'near_degenerate', 'clustered', 'isospectral',
    'thin_neck',
})

TAGS = GEOMETRIC_TAGS | SPECTRAL_TAGS

# A domain is "elongated" past this slenderness. Calibration: disk 1.0,
# unit square 1.41, ellipse(2,1) 2.0, ellipse(3,1) 3.0, rect(1,8) 8.0.
ELONGATED = 3.0


def derive_tags(domain, family=None, params=None, feats=None):
    """Geometric tags implied by the domain itself.

    Only :data:`GEOMETRIC_TAGS` are produced; spectral tags are the registry's
    business.
    """
    f = feats if feats is not None else describe(domain, family, params)
    tags = set()

    if f['n_regular']:
        tags.add('regular_corners')
    if f['n_singular']:
        tags.add('singular_corners')
    if f['n_singular'] >= 3:
        tags.add('many_singular')
    if f['n_reentrant']:
        tags.add('reentrant')
    if f['n_reentrant'] >= 3:
        tags.add('many_reentrant')
    if f['n_sharp']:
        tags.add('sharp')

    if f['n_corners'] == 0:
        tags.add('smooth')
    elif f['n_curved']:
        tags.add('mixed_smooth_corner')
    if f['n_curvature_discont']:
        tags.add('curvature_discont')

    if f['slenderness'] >= ELONGATED:
        tags.add('elongated')
    if f['sym_order'] == 1:
        tags.add('no_symmetry')
    if f['n_blocked_branch_cuts']:
        tags.add('branch_cut_polyline')

    return tags
