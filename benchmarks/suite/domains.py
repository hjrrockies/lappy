"""The curated benchmark domain suite.

A designed sample rather than an accumulated list.  Each entry names the
difficulty mechanism it is there to exercise; the taxonomy in ``README.md`` is
the primary artifact and this module is its executable index.

Four tiers:

A. **analytic** -- closed-form spectra, so method error is separable from
   reference error.  The correctness floor, and the only place the corner
   exponent can be swept continuously against exact truth (via ``disk_sector``).
B. **corner** -- polygons whose difficulty is corner singularity structure.
C. **curved** -- curved and mixed boundaries, where the difficulty is
   smoothness rather than corners.
D. everything above carries a ``status``; ``'hard'`` and ``'open'`` entries are
   the improvement targets and are *expected* to underperform.

Declared tags are cross-checked against ``features.derive_tags`` by
:func:`check_tags`, so a geometric claim here cannot drift from the geometry.
Spectral tags (``exact_multiplicity``, ``clustered``, ...) are not derivable and
are taken on trust.
"""
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional
import numpy as np

from lappy import geometry as G, reference as R

from .features import (SPECTRAL_TAGS, TAGS, derive_tags, describe,
                       symmetry_of)

TIERS = ('analytic', 'corner', 'curved')
TRUTHS = ('analytic', 'certified', 'best_known', 'none')
STATUSES = ('ok', 'hard', 'open')


@dataclass(frozen=True)
class SuiteDomain:
    """One benchmark domain.

    Attributes
    ----------
    key : str
        Unique identifier, also the ``produce.py`` domain key where applicable.
    family : str
        Parametric family this belongs to.  Doubles as the lookup key for
        ``lappy.symmetry.domain_symmetry``.
    build : callable
        Zero-argument builder returning a ``Domain``.
    params : dict
        Position within the family's sweep; also the kwargs handed to
        ``domain_symmetry``.
    tags : frozenset
        Taxonomy labels.  Geometric ones are cross-checked; spectral ones are
        declared.
    truth : str
        Provenance of the reference eigenvalues.  ``'analytic'`` (closed form),
        ``'certified'`` (Moler--Payne bound in ``results_certified.json``),
        ``'best_known'`` (MPS table with a documented digit ceiling), ``'none'``.
    truth_fn : callable or None
        ``truth_fn(k)`` -> first ``k`` eigenvalues, for ``truth='analytic'``.
    symmetric : bool
        Whether the domain *mathematically* has a symmetry usable by
        ``lappy.symmetry``.  Compared against what ``domain_symmetry`` actually
        returns; a ``True`` here with no registered group is a known gap.
    status : str
        ``'ok'``, ``'hard'`` (converges but short of target accuracy), ``'open'``
        (does not converge usefully at any tried configuration).
    digit_ceiling : float or None
        Best accuracy observed to date, from ``TUNING_LOG.md`` /
        ``results_certified.json``.  ``None`` where never measured.
    n_basis, n_eigs : int
        Production hints for ``benchmarks/reference/produce.py``.
    why : str
        The one thing this entry is in the sample for.
    """
    key: str
    tier: str
    family: str
    build: Callable
    params: dict = field(default_factory=dict)
    tags: frozenset = frozenset()
    truth: str = 'none'
    truth_fn: Optional[Callable] = None
    symmetric: bool = False
    status: str = 'ok'
    digit_ceiling: Optional[float] = None
    n_basis: int = 240
    n_eigs: int = 10
    why: str = ''

    def domain(self):
        return self.build()

    def features(self):
        return describe(self.build(), self.family, self.params)

    def group(self):
        return symmetry_of(self.family, self.params)


def _d(key, tier, family, build, params_or_tags, tags_or_why, why=None, **kw):
    """Terse constructor; ``params`` may be omitted when empty."""
    if why is None:
        params, tags, why = {}, params_or_tags, tags_or_why
    else:
        params, tags = params_or_tags, tags_or_why
    return SuiteDomain(key=key, tier=tier, family=family, build=build,
                       params=params, tags=frozenset(tags), why=why, **kw)


# ===========================================================================
# Tier A -- analytic ground truth
# ===========================================================================
# These separate method error from reference error.  The disk_sector family is
# the important one: reference.sector_eigs accepts an arbitrary opening angle,
# so it sweeps the entire corner-exponent axis -- from the near-slit p ~ 1/2 to
# sharp p ~ 13 -- against exact truth.  Nothing else in the suite can do that.
# Note the angles chosen are deliberately *not* pi/integer, so the corners are
# genuinely singular; disk_sector(1, pi/6) would have p = 6 exactly and the
# Fourier--Bessel expansion there is analytic, i.e. no difficulty at all.

_ANALYTIC = [
    _d('square', 'analytic', 'rect', lambda: G.rect(1, 1),
       ['regular_corners', 'exact_multiplicity'],
       'the easy floor: all corners regular (p=2), and high multiplicity from '
       'both the D4 symmetry and the m^2+n^2 number theory',
       truth='analytic', truth_fn=partial(R.rect_eigs, L=1, H=1),
       symmetric=True, n_basis=120),

    _d('rect_near_deg_1e3', 'analytic', 'rect', lambda: G.rect(1, 1.001),
       ['regular_corners', 'near_degenerate'],
       'near-degeneracy knob, coarse: splits the exact square double at '
       '49.348 by ~1.2e-3 relative -- comfortably resolvable, the control for '
       'its 1e-5 sibling',
       truth='analytic', truth_fn=partial(R.rect_eigs, L=1, H=1.001),
       symmetric=True, n_basis=120),

    _d('rect_near_deg_1e5', 'analytic', 'rect', lambda: G.rect(1, 1.00001),
       ['regular_corners', 'near_degenerate'],
       'near-degeneracy knob, fine: split ~1.2e-5 relative -- below this the '
       'pair is indistinguishable from a true double and estimate_multiplicity '
       'should say so',
       truth='analytic', truth_fn=partial(R.rect_eigs, L=1, H=1.00001),
       symmetric=True, n_basis=120),

    _d('rect_thin', 'analytic', 'rect', lambda: G.rect(1, 8),
       ['regular_corners', 'elongated'],
       'elongation with exact truth: aspect 8, quasi-1D modes, no corner '
       'difficulty at all to confound it',
       truth='analytic', truth_fn=partial(R.rect_eigs, L=1, H=8),
       symmetric=True, n_basis=240),

    _d('eq_tri', 'analytic', 'eq_tri', lambda: G.eq_tri(1),
       ['regular_corners', 'exact_multiplicity'],
       'exact spectrum with genuine multiplicity; regular p=3 corners',
       truth='analytic', truth_fn=partial(R.eq_tri_eigs, l=1),
       symmetric=True, n_basis=120),

    _d('iso_right_tri', 'analytic', 'iso_right_tri', lambda: G.iso_right_tri(1),
       ['regular_corners'],
       'exact; mixed regular angles p=2,4. Note this is the same shape as '
       'iso_tri(1) up to similarity, which gives the iso_tri sweep an exactly '
       'known member',
       truth='analytic', truth_fn=partial(R.iso_right_tri_eigs, l=1),
       symmetric=True, n_basis=120),

    _d('disk', 'analytic', 'disk', lambda: G.disk(1),
       ['smooth', 'exact_multiplicity'],
       'smooth boundary with exact truth; every m>=1 mode is exactly double, so '
       'this is the multiplicity test for the fundamental-solution basis path',
       truth='analytic', truth_fn=partial(R.disk_eigs, r=1),
       symmetric=True, n_basis=120),

    _d('sector_reflex', 'analytic', 'disk_sector',
       lambda: G.disk_sector(1, 3 * np.pi / 2),
       ['mixed_smooth_corner', 'regular_corners', 'singular_corners', 'reentrant'],
       'THE calibration point the suite was missing: a reentrant singular corner '
       '(p=2/3, the same exponent as L_shape) with exact truth, so reentrant '
       'convergence can be measured rather than inferred',
       truth='analytic', truth_fn=partial(R.sector_eigs, R=1, alpha=3 * np.pi / 2),
       symmetric=True, n_basis=240),

    _d('sector_sharp_p65', 'analytic', 'disk_sector',
       lambda: G.disk_sector(1, np.pi / 6.5),
       ['mixed_smooth_corner', 'regular_corners', 'singular_corners', 'sharp'],
       'sharp singular corner (p=6.5) with exact truth -- isolates the chevron '
       'cost mechanism with zero reference uncertainty. The angle is pi/6.5, not '
       'pi/6, precisely so the corner is singular',
       truth='analytic', truth_fn=partial(R.sector_eigs, R=1, alpha=np.pi / 6.5),
       symmetric=True, n_basis=240),

    _d('sector_sharp_p133', 'analytic', 'disk_sector',
       lambda: G.disk_sector(1, np.pi / 13.3),
       ['mixed_smooth_corner', 'regular_corners', 'singular_corners', 'sharp',
        'elongated'],
       'the sharp end of the exact sweep (p=13.3), comparable to chevron(1,1.5)s '
       'p~15.9 but with exact truth and only one such corner',
       truth='analytic', truth_fn=partial(R.sector_eigs, R=1, alpha=np.pi / 13.3),
       symmetric=True, status='hard', n_basis=320),

    _d('sector_slit', 'analytic', 'disk_sector',
       lambda: G.disk_sector(1, 2 * np.pi - 0.05),
       ['mixed_smooth_corner', 'regular_corners', 'singular_corners', 'reentrant'],
       'the extreme reentrant case, p=0.504, essentially a slit -- worst possible '
       'corner exponent, and still exactly solvable',
       truth='analytic', truth_fn=partial(R.sector_eigs, R=1, alpha=2 * np.pi - 0.05),
       symmetric=True, status='hard', n_basis=320),
]

# ===========================================================================
# Tier B -- corner singularity structure (polygonal, MPS truth)
# ===========================================================================

_CORNER = [
    _d('L_shape', 'corner', 'L_shape', lambda: G.L_shape(),
       ['regular_corners', 'singular_corners', 'reentrant'],
       'the clean-singular control: exactly one reentrant corner (p=2/3), '
       'certified to 13 digits. Anything that regresses here is broken',
       truth='certified', symmetric=True, digit_ceiling=13.3, n_basis=240),

    _d('H_shape', 'corner', 'H_shape', lambda: G.H_shape(),
       ['regular_corners', 'singular_corners', 'many_singular', 'many_reentrant',
        'reentrant', 'elongated'],
       'four reentrant corners against L_shapes one, with only a half-turn to '
       'exploit -- the cleanest test of how accuracy degrades with the *count* '
       'of singular corners',
       truth='best_known', symmetric=True, status='hard', digit_ceiling=8.2,
       n_basis=480),

    _d('GWW1', 'corner', 'GWW1', lambda: G.GWW1(),
       ['regular_corners', 'singular_corners', 'many_singular', 'reentrant',
        'no_symmetry', 'isospectral'],
       'isospectral partner of GWW2, and one of only three domains in the suite '
       'with genuinely no symmetry to exploit',
       truth='best_known', symmetric=False, status='hard', digit_ceiling=9.9,
       n_basis=320),

    _d('GWW2', 'corner', 'GWW2', lambda: G.GWW2(),
       ['regular_corners', 'singular_corners', 'many_singular', 'reentrant',
        'no_symmetry', 'isospectral'],
       'the other half of the isospectral pair: agreement between GWW1 and GWW2 '
       'is a reference check that needs no external table',
       truth='best_known', symmetric=False, status='hard', digit_ceiling=8.7,
       n_basis=320),

    _d('reg_ngon_5', 'corner', 'reg_ngon', lambda: G.reg_ngon(5), {'n': 5},
       ['singular_corners', 'many_singular', 'exact_multiplicity'],
       'mildly singular corners (p=5/3) with rich symmetry -- the benign end of '
       'the singular scale', truth='certified', symmetric=True,
       digit_ceiling=12.3, n_basis=240),

    _d('reg_ngon_6', 'corner', 'reg_ngon', lambda: G.reg_ngon(6), {'n': 6},
       ['singular_corners', 'many_singular', 'exact_multiplicity'],
       'p=3/2 corners, D2 symmetry', truth='certified', symmetric=True,
       digit_ceiling=13.0, n_basis=320),

    _d('reg_ngon_7', 'corner', 'reg_ngon', lambda: G.reg_ngon(7), {'n': 7},
       ['singular_corners', 'many_singular', 'exact_multiplicity'],
       'p=7/5, odd n so only a single mirror', truth='certified',
       symmetric=True, digit_ceiling=11.7, n_basis=240),

    _d('reg_ngon_8', 'corner', 'reg_ngon', lambda: G.reg_ngon(8), {'n': 8},
       ['singular_corners', 'many_singular', 'exact_multiplicity', 'clustered'],
       'the near-degenerate cluster failure: five eigenvalues crowd together and '
       'a shallow local tension minimum was accepted into the reference table as '
       'a spurious eigenvalue. The multiplicity/spurious-detection target',
       truth='best_known', symmetric=True, status='hard', digit_ceiling=10.3,
       n_basis=320),

    _d('chevron_1_15', 'corner', 'chevron', lambda: G.chevron(1, 1.5),
       {'h1': 1, 'h2': 1.5},
       ['singular_corners', 'many_singular', 'reentrant', 'sharp', 'elongated'],
       'two 11.3-degree corners (p~15.9) plus a reentrant one. Corner reweighting '
       'and denser collocation both made it worse; genuinely needs n_basis 400+, '
       'which hits the Bessel-evaluation wall',
       truth='best_known', symmetric=True, status='hard', digit_ceiling=6.3,
       n_basis=480),

    _d('chevron_1_2', 'corner', 'chevron', lambda: G.chevron(1, 2),
       {'h1': 1, 'h2': 2},
       ['singular_corners', 'many_singular', 'reentrant', 'sharp', 'elongated'],
       'sharp-corner sweep, milder', truth='best_known', symmetric=True,
       status='hard', digit_ceiling=7.1, n_basis=480),

    _d('chevron_2_3', 'corner', 'chevron', lambda: G.chevron(2, 3),
       {'h1': 2, 'h2': 3},
       ['singular_corners', 'many_singular', 'reentrant', 'sharp', 'elongated'],
       'sharp-corner sweep, worst converging of the four',
       truth='best_known', symmetric=True, status='hard', digit_ceiling=4.6,
       n_basis=480),

    _d('chevron_2_4', 'corner', 'chevron', lambda: G.chevron(2, 4),
       {'h1': 2, 'h2': 4},
       ['singular_corners', 'many_singular', 'reentrant', 'sharp', 'elongated'],
       'sharp-corner sweep; three of its four corners are sharp singular',
       truth='best_known', symmetric=True, status='hard', digit_ceiling=5.0,
       n_basis=480),

    _d('chevron_1_125', 'corner', 'chevron', lambda: G.chevron(1, 1.25),
       {'h1': 1, 'h2': 1.25},
       ['singular_corners', 'many_singular', 'reentrant', 'sharp', 'elongated'],
       'OPEN: 6.3-degree corners, p~28.4. Does not converge usefully at any '
       'configuration tried. Excluded from reference production on purpose -- '
       'kept here because it is the sharpest target in the suite',
       truth='none', symmetric=True, status='open', n_basis=480),

    _d('iso_tri_h05', 'corner', 'iso_tri', lambda: G.iso_tri(0.5), {},
       ['singular_corners', 'many_singular', 'sharp', 'elongated'],
       'apex-sharpness sweep, obtuse end: all three corners singular (p=6.78, '
       '1.42, 6.78)', truth='certified', symmetric=True, digit_ceiling=12.3,
       n_basis=240),

    _d('iso_tri_h1', 'corner', 'iso_tri', lambda: G.iso_tri(1.0), {},
       ['regular_corners'],
       'the sweep member with all-regular corners (45/90/45) -- and it is '
       'similar to iso_right_tri, so it has exact truth. The control that says '
       'what the iso_tri family costs with the singularity removed',
       truth='analytic',
       truth_fn=partial(R.iso_right_tri_eigs, l=np.sqrt(2)),
       symmetric=True, digit_ceiling=13.5, n_basis=240),

    _d('iso_tri_h4', 'corner', 'iso_tri', lambda: G.iso_tri(4.0), {},
       ['singular_corners', 'many_singular', 'sharp'],
       'apex-sharpness sweep, p=6.41 apex', truth='certified', symmetric=True,
       digit_ceiling=12.9, n_basis=240),

    _d('iso_tri_h16', 'corner', 'iso_tri', lambda: G.iso_tri(16.0), {},
       ['singular_corners', 'many_singular', 'sharp', 'elongated'],
       'apex-sharpness sweep, extreme end: p=25.2 apex, aspect 16. Slow enough '
       'at n_basis=240 to have been killed once -- the cost target',
       truth='best_known', symmetric=True, status='hard', digit_ceiling=10.1,
       n_basis=240),

    _d('parallelogram_60', 'corner', 'parallelogram',
       lambda: G.parallelogram(1, 1, np.pi / 3), {},
       ['regular_corners', 'singular_corners'],
       'sharp corners with only a half-turn, contrasting with iso_tris mirror: '
       'half the symmetry reduction for the same corner structure',
       truth='none', symmetric=True, n_basis=240),

    _d('parallelogram_p65', 'corner', 'parallelogram',
       lambda: G.parallelogram(1, 1, np.pi / 6.5), {},
       ['singular_corners', 'many_singular', 'sharp', 'elongated'],
       'shear sweep, sharper (p=6.5). The shear is pi/6.5 rather than pi/6 on '
       'purpose: at pi/6 the corner exponent would be the integer 6 and the '
       'corner would not be singular at all',
       truth='none', symmetric=True, status='hard', n_basis=320),

    _d('parallelogram_p127', 'corner', 'parallelogram',
       lambda: G.parallelogram(1, 1, np.pi / 12.7), {},
       ['singular_corners', 'many_singular', 'sharp', 'elongated'],
       'shear sweep, extreme: p=12.7 corners at slenderness ~18. Combines the '
       'sharp-corner and elongation mechanisms, which no other entry does',
       truth='none', symmetric=True, status='hard', n_basis=320),

    _d('right_trapezoid', 'corner', 'right_trapezoid',
       lambda: G.right_trapezoid(1, 2), {},
       ['regular_corners', 'singular_corners', 'no_symmetry'],
       'generic asymmetric convex quadrilateral with no symmetry group at all -- '
       'the baseline against which symmetry speedups should be quoted',
       truth='none', symmetric=False, n_basis=240),

    _d('spiral', 'corner', 'spiral', lambda: G.spiral(), {},
       ['singular_corners', 'many_singular', 'reentrant', 'many_reentrant',
        'elongated', 'no_symmetry', 'branch_cut_polyline'],
       'OPEN: the only domain where corners have no straight-ray sightline to '
       'infinity (12 of 24 blocked), forcing corner_branch_cut_polyline. '
       'Benchmarked nowhere until now',
       truth='none', symmetric=False, status='open', n_basis=320),
]

# ===========================================================================
# Tier C -- curved and mixed boundaries
# ===========================================================================

_CURVED = [
    _d('ellipse_a2', 'curved', 'ellipse', lambda: G.ellipse(2, 1), {},
       ['smooth'],
       'smooth boundary, pure fundamental-solution basis path; certified to 14 '
       'digits and the fastest-converging domain in the suite',
       truth='certified', symmetric=True, digit_ceiling=14.4, n_basis=240),

    _d('ellipse_a3', 'curved', 'ellipse', lambda: G.ellipse(3, 1), {},
       ['smooth'],
       'smooth elongation sweep, midpoint (slenderness 3.0, right on the '
       'elongated threshold)', truth='certified', symmetric=True,
       digit_ceiling=13.5, n_basis=320),

    _d('ellipse_a4', 'curved', 'ellipse', lambda: G.ellipse(4, 1), {},
       ['smooth', 'elongated'],
       'smooth elongation sweep, extreme end -- tests source placement for the '
       'fundamental-solution basis, not corners',
       truth='certified', symmetric=True, digit_ceiling=13.0, n_basis=320),

    _d('stadium', 'curved', 'stadium', lambda: G.stadium(1, 1), {'L': 1.0},
       ['smooth', 'curvature_discont'],
       'HARD, and the reason curvature_discont is its own tag: zero corners, yet '
       'stuck at 2-3 digits. Four C^1-but-not-C^2 junctions where the boundary '
       'is neither a corner the FB basis can aim at nor the smooth boundary the '
       'FS basis wants. Denser collocation provably does not help',
       truth='none', symmetric=True, status='hard', digit_ceiling=3.0,
       n_basis=320),

    _d('stadium_L2', 'curved', 'stadium', lambda: G.stadium(2, 1), {'L': 2.0},
       ['smooth', 'curvature_discont'],
       'longer stadium: the curvature-discontinuity mechanism needs a sweep, not '
       'a single point, to tell "intrinsic to the C^1 junction" apart from '
       '"specific to L=H=1"',
       truth='none', symmetric=True, status='hard', n_basis=320),

    _d('mushroom', 'curved', 'mushroom', lambda: G.mushroom(), {},
       ['mixed_smooth_corner', 'regular_corners', 'singular_corners', 'reentrant'],
       'arc joined to a polygon at genuine corners, with two reentrant corners -- '
       'the mixed case where both basis families are needed at once',
       truth='certified', symmetric=True, digit_ceiling=12.5, n_basis=320),

    _d('mushroom_thin', 'curved', 'mushroom',
       lambda: G.mushroom(1, 0.25, 1.5), {},
       ['mixed_smooth_corner', 'regular_corners', 'singular_corners', 'reentrant',
        'thin_neck', 'clustered'],
       'narrow-stem variant: modes localize in the cap or the stem, so the '
       'spectrum has near-degenerate pairs of very differently-shaped '
       'eigenfunctions. No other entry produces localized modes',
       truth='none', symmetric=True, n_basis=320),

    _d('mushroom_neck01', 'curved', 'mushroom',
       lambda: G.mushroom(1, 0.1, 1.5), {},
       ['mixed_smooth_corner', 'regular_corners', 'singular_corners', 'reentrant',
        'thin_neck', 'clustered'],
       'neck-width sweep, extreme: at b=0.1 the stem modes are nearly decoupled '
       'from the cap, so pairs of eigenvalues collapse together and the two '
       'eigenfunctions live in disjoint parts of the domain',
       truth='none', symmetric=True, status='hard', n_basis=320),

    _d('cut_square_r025', 'curved', 'cut_square', lambda: G.cut_square(0.25), {},
       ['mixed_smooth_corner', 'regular_corners'],
       'HARD and instructive: every corner is regular (p=2) and there is no '
       'singularity anywhere, yet it reaches only ~6-9 digits. Whatever limits '
       'it is a property of mixing an arc into a polygon, not of corners -- the '
       'cleanest isolation of that mechanism in the suite',
       truth='best_known', symmetric=True, status='hard', digit_ceiling=9.0,
       n_basis=640),

    _d('cut_square_r05', 'curved', 'cut_square', lambda: G.cut_square(0.5), {},
       ['mixed_smooth_corner', 'regular_corners'],
       'the same shape with a larger arc, which reaches 13 digits -- the pair '
       'brackets whatever r-dependence drives the r=0.25 failure',
       truth='certified', symmetric=True, digit_ceiling=13.1, n_basis=640),
]

_CORNER.append(
    _d('spiral_t25', 'corner', 'spiral',
       lambda: G.spiral(turns=2.5), {},
       ['singular_corners', 'many_singular', 'reentrant', 'many_reentrant',
        'elongated', 'no_symmetry', 'branch_cut_polyline'],
       'OPEN: more turns, so more corners buried inside the coil with no '
       'sightline out. The second point that makes blocked branch cuts a sweep '
       'rather than an anecdote',
       truth='none', symmetric=False, status='open', n_basis=320))

SUITE = {d.key: d for d in (_ANALYTIC + _CORNER + _CURVED)}

assert len(SUITE) == len(_ANALYTIC) + len(_CORNER) + len(_CURVED), 'duplicate key'

# A minimal spanning subset: one representative per difficulty mechanism, for
# when running the full suite is too expensive. Every tag in the vocabulary is
# covered at least once. The full suite exists so that each mechanism can be
# swept rather than sampled; CORE exists so it can be *spot-checked*.
CORE = (
    'square',            # regular corners, exact multiplicity -- the floor
    'disk',              # smooth + exact multiplicity
    'rect_near_deg_1e5', # near-degeneracy
    'rect_thin',         # elongation, no corner difficulty
    'sector_reflex',     # reentrant singular corner, exact truth
    'sector_sharp_p65',  # sharp singular corner, exact truth
    'L_shape',           # one reentrant corner, certified 13 digits
    'H_shape',           # many reentrant corners
    'GWW1',              # no symmetry (and isospectral, with GWW2)
    'reg_ngon_8',        # clustered spectrum
    'chevron_1_15',      # sharp corners, hard
    'iso_tri_h16',       # sharp + elongated, cost-limited
    'spiral',            # blocked branch cuts
    'ellipse_a2',        # smooth, fundamental-solution basis path
    'stadium',           # curvature discontinuity
    'mushroom_thin',     # thin neck, localized modes
    'cut_square_r025',   # mixed arc/polygon, all-regular corners, still hard
)

assert set(CORE) <= set(SUITE), 'CORE names a domain not in the suite'


# --- selection --------------------------------------------------------------

def select(tier=None, tag=None, status=None, truth=None, family=None):
    """Filter the suite.  Every argument accepts a value or an iterable."""
    def match(field_val, want):
        if want is None:
            return True
        want = {want} if isinstance(want, str) else set(want)
        return field_val in want

    out = []
    for d in SUITE.values():
        if not (match(d.tier, tier) and match(d.status, status)
                and match(d.truth, truth) and match(d.family, family)):
            continue
        if tag is not None:
            want = {tag} if isinstance(tag, str) else set(tag)
            if not (want & d.tags):
                continue
        out.append(d)
    return out


def for_reference_production(include_open=False):
    """The ``produce.py`` view: ``key -> (builder, symmetry, n_basis, n_eigs)``.

    Excludes ``tier='analytic'`` entries, which have closed forms and need no
    production run, and (by default) ``status='open'`` entries, which do not
    converge.  Everything else is included, ``truth='none'`` entries very much
    included -- producing their first reference values is exactly what
    ``produce.py`` is for.

    ``symmetry`` is ``(family, params)`` when ``lappy.symmetry`` can supply a
    group and ``None`` otherwise, so domains whose symmetry is merely
    unregistered fall back to a full-domain solve rather than failing.
    """
    out = {}
    for d in SUITE.values():
        if d.tier == 'analytic':
            continue
        if d.status == 'open' and not include_open:
            continue
        sym = (d.family, d.params) if d.group() is not None else None
        out[d.key] = (d.build, sym, d.n_basis, d.n_eigs)
    return out


# --- validation -------------------------------------------------------------

def check_tags(entry):
    """Compare declared geometric tags against those derived from the geometry.

    Returns ``(missing, extra)``: tags the geometry implies but the registry
    does not declare, and vice versa.  Spectral tags are ignored.
    """
    dom = entry.build()
    feats = describe(dom, entry.family, entry.params)
    derived = derive_tags(dom, entry.family, entry.params, feats)
    declared = set(entry.tags) - SPECTRAL_TAGS
    # 'no_symmetry' is excluded from the comparison: derived, it means only
    # "lappy.symmetry has no entry for this family", whereas declared it is a
    # claim about the shape. validate() checks the two against each other via
    # `symmetric` and reports the disagreements as a registry gap, which is
    # information rather than an error. See the note in features.py.
    derived.discard('no_symmetry')
    declared.discard('no_symmetry')
    return derived - declared, declared - derived


def validate():
    """Check the whole registry.  Returns a list of problem strings."""
    problems = []
    for key, d in SUITE.items():
        if d.tier not in TIERS:
            problems.append(f'{key}: bad tier {d.tier!r}')
        if d.truth not in TRUTHS:
            problems.append(f'{key}: bad truth {d.truth!r}')
        if d.status not in STATUSES:
            problems.append(f'{key}: bad status {d.status!r}')
        unknown = set(d.tags) - TAGS
        if unknown:
            problems.append(f'{key}: unknown tags {sorted(unknown)}')
        if (d.truth == 'analytic') != (d.truth_fn is not None):
            problems.append(f'{key}: truth/truth_fn disagree')
        if not d.why:
            problems.append(f'{key}: no rationale')
        if ('no_symmetry' in d.tags) != (not d.symmetric):
            problems.append(f'{key}: no_symmetry tag disagrees with symmetric')

        try:
            missing, extra = check_tags(d)
        except Exception as exc:
            problems.append(f'{key}: failed to build/derive: '
                            f'{type(exc).__name__}: {exc}')
            continue
        if missing:
            problems.append(f'{key}: geometry implies undeclared {sorted(missing)}')
        if extra:
            problems.append(f'{key}: declared but not implied {sorted(extra)}')

        if d.truth_fn is not None:
            try:
                vals = np.asarray(d.truth_fn(d.n_eigs), dtype=float)
            except Exception as exc:
                problems.append(f'{key}: truth_fn failed: {exc}')
            else:
                if vals.shape != (d.n_eigs,) or not np.all(np.isfinite(vals)):
                    problems.append(f'{key}: truth_fn gave {vals.shape}, want '
                                    f'{(d.n_eigs,)} finite values')
                elif np.any(np.diff(vals) < 0) or vals[0] <= 0:
                    problems.append(f'{key}: truth_fn values not sorted positive')
    return problems
