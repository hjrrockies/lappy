"""Basis construction for polygons, designed for the shape-optimization inner loop.

    plan  = plan_basis(domain, lam_max, target)   # discrete: WHAT blocks exist. Rare, expensive.
    basis = realize(plan, domain)                 # continuous: WHERE they sit. Per iterate, cheap.

Replaces `lappy/heuristics.py`, whose closed-form recipe was measured over 1154 runs
(`benchmarks/basis_lab/HEURISTICS.md`) and found to fail in ways constants could not fix. Each
design decision below is answerable to one of those measurements.

WHY THE PLAN AND THE REALIZATION ARE SEPARATE. A shape optimizer perturbs the domain slightly and
re-solves, thousands of times. If the basis is re-derived from scratch each time, then any
*threshold* in the derivation is a discontinuity in the objective: the old recipe classified a
corner as weakly-singular when `alpha > Lambda/(2 ln 15)`, and crossing that boundary deleted a
`handover_frac*R_c` hole in the source layer, so `n` was not even monotone in the requested
precision (`iso_tri_h4`: 343 columns at 1e-2, 261 at 1e-4). A basis that jumps makes
finite-difference gradients meaningless and line searches unfalsifiable. So the discrete content --
which corners get Fourier-Bessel blocks, how many terms, which arc each block owns, how many
sources per arc -- is decided once, frozen in a `BasisPlan`, and reused; `realize` recomputes only
smooth geometric quantities. `n_basis` is then constant along a shape family by construction, not
by luck.

WHAT `target` MEANS, AND WHY IT IS NOT CALIBRATED. It is a requested relative eigenvalue precision,
entering as `Lambda = ln(1/target)`. The old recipe passed it through a Moler-Payne calibration
constant `C_Omega` and was measured *inert*: achieved accuracy was flat to within 1.5 digits from
1e-4 down on 12 of 18 domains while `n` doubled. No closed form was going to predict achieved
accuracy, so this module does not try. `refine_plan` MEASURES the boundary residual per arc and
raises the budget only where it falls short; the plan is then frozen having been demonstrated to
meet its target rather than predicted to. That also sidesteps the `C_Omega` calibration entirely.

OWNERSHIP IS A PARTITION, NOT A SUBTRACTION. Every point of the boundary belongs to exactly one
block. A corner owns the arc within `R_c` of itself along both adjacent edges and its
Fourier-Bessel series resolves it; the rest of each edge is covered by fundamental-solution
sources. The old recipe generated a global source curve and then *deleted* the sources near an
FB-equipped corner, which left near-duplicate columns that destabilized the GSVD's rank
truncation. More importantly its spacing rule was global -- `h = (pi*eta/Lambda)*dist(x, S*)`, the
distance to a set of reflected corner images -- so one sharp corner refined the *whole* boundary:
1561 of 1658 columns on `chevron_2_4` were ambient sources. Here the offset comes from **local**
thickness and the spacing from a **per-arc** requirement, so a sharp corner cannot pay for
refinement on the far side of the domain.

THERE IS A CEILING ON USEFUL COLUMNS AND IT IS ENFORCED. `mps.regularize_pencil` truncates the
pencil at `rtol`, and the surviving rank saturates: measured at 235-236 on both `square` and
`L_shape`, reached near n=360 on the latter (`PLAN_LAB.md`, S0a). Past saturation a column costs a
full transcendental evaluation and a wider QR and is then discarded before it can affect the
answer. `cfg.n_cap` refuses to build past it, and reports the shortfall instead of silently
spending. The old recipe asked for 316 columns on L_shape and 1929 on chevron_2_4.

EVERY CORNER GETS FOURIER-BESSEL TERMS, with no singular/regular branch. Two measurements force
this. `pure_fb` at matched size beats every other construction on the all-regular-corner domains
(15.1 digits against the old recipe's 9.8 on `square`), so regular corners want FB terms too; and
turning the optional regular-corner block off, which a single-factor screen scored at -0.1 digits,
cost **-8.7 digits** on `square` once combined with a sparser source layer. The term count falls
continuously out of the same formula for every corner, so there is no threshold to cross.

TUNED TO CERTIFIED EIGENVALUE DIGITS, and that is now a measured choice rather than an assumption:
`PLAN_LAB.md` S0b/S0c compared both objectives on the same bases and found 95% pairwise agreement,
with the certified bound the conservative one by a median 2.6 digits, and no case of an accurate
eigenvalue with an inaccurate derivative. `dlambda` accuracy follows; it is regression-tested, not
optimized against.

Out of scope: curved boundaries (polygon-only, as `heuristics.py` was), and multipole blocks.
"""
from dataclasses import dataclass, field, replace
from math import ceil, log, pi, sqrt

import numpy as np

from .bases import FourierBesselBasis, FundamentalBasis
from .core import BaseDomain
from .eigfun_integrals import corner_clearance, corner_specs

__all__ = ['PlanConfig', 'CornerBlock', 'ArcBlock', 'BasisPlan',
           'plan_basis', 'realize', 'refine_plan', 'polygon_default_basis',
           'residual_by_arc', 'plan_of', 'local_thickness', 'exterior_clearance']

_EPS = float(np.finfo(float).eps)


def _indep_digits(cfg):
    """How much dynamic range two basis columns must differ by to count as independent.

    Not `ln(1/eps_machine)`. Columns stop being distinguishable *to the solver* at its own
    rank-truncation threshold: `mps.regularize_pencil` discards singular values below `rtol`, so
    `ln(1/rtol)` is where redundancy begins, and that is about 27.6 against machine epsilon's 36.7.

    Using the machine constant instead was measured wrong. It let the source budget grow past the
    point of usefulness and made achieved accuracy NON-MONOTONE in the requested target -- on
    `iso_tri_h16`, 4.2 digits at 1e-7, 5.2 at 1e-10, back to 4.2 at 1e-13, with `n_reg == n`
    throughout, so the pencil was not even truncating: the extra columns were actively harmful.
    Swept over six domains:

        ceiling constant   36.7 (eps)   27.6 (rtol)   23.0     18.0
        iso_tri_h16        4.2/5.2/4.2  4.2/5.2/5.7   ...5.5   ...5.3
        chevron_1_2        6.1/6.4/6.3  6.1/6.4/8.0   ...8.8   ...7.5
        L_shape            9.4/12.9/14.0  9.4/12.9/13.9  ...12.1  ...12.5
        square             14.3/14.4/14.4  ...14.2    ...14.5  13.6 (non-mono)

    (targets 1e-7 / 1e-10 / 1e-13.) `ln(1/rtol)` is the only setting that is monotone on all six
    without giving up accuracy anywhere; tighter buys the chevrons a little and costs L_shape ~2
    digits.
    """
    return log(1.0/cfg.rtol)


@dataclass(frozen=True)
class PlanConfig:
    """Tuned constants. Every one of these is a *cost* dial: S0a/h4 found no constant in the old
    recipe that moved accuracy by more than 0.4 digits at these wavenumbers, so the honest framing
    is that these buy or spend columns and the refinement pass is what buys accuracy."""
    gamma: float = 0.40        # corner arc radius as a fraction of its clearance
    reach_frac: float = 0.35   # ...and never more than this fraction of an adjacent edge
    sharp_ref: float = 2.0     # alpha above which a corner stops owning a full arc; _corner_blocks
    airy: float = 2.0          # coefficient of (kappa R)^(1/3) in the oscillatory term
    order_margin: float = 3.0  # additive margin on the corner term count
    delta_frac_D: float = 0.15     # source offset ceiling, as a fraction of the diameter
    delta_frac_thick: float = 0.5  # ...and of the local interior thickness
    delta_frac_ext: float = 0.4    # ...and of the exterior clearance along the normal
    delta_hard_D: float = 0.40     # hard offset limits refinement may grow into (see _arc_blocks)
    delta_hard_frac: float = 0.90
    nyquist_ppw: float = 3.0   # minimum source points per wavelength
    n_cap: int = 400           # refuse to plan past the measured rank-saturation ceiling
    min_src_per_arc: int = 2
    cone: float = 0.5          # local_thickness inward cone, cos of the half-angle
    n_probe: int = 1500        # boundary samples for local_thickness
    fb_inner_frac: float = 0.77  # fraction of the owned radius the collocation resolves; _fb_ceiling
    rtol: float = 1e-12       # the solver's rank-truncation threshold; sets both ceilings
    refine_growth: float = 1.6  # per-arc budget multiplier in refine_plan
    max_refine: int = 3


@dataclass(frozen=True)
class CornerBlock:
    """A Fourier-Bessel block at one corner, and the arc it owns.

    `reach_prev`/`reach_next` are stored as FRACTIONS of the adjacent edge lengths, not as
    absolute radii, so that `realize` on a perturbed domain moves the arc smoothly with the
    geometry instead of re-deciding where it starts.
    """
    corner: int
    M: int
    alpha: float
    reach_prev: float
    reach_next: float
    R: float                 # absolute owned radius at plan time, diagnostic only
    clearance: float
    nu_osc: float
    nu_cont: float


@dataclass(frozen=True)
class ArcBlock:
    """Fundamental-solution sources covering `[tau0, tau1]` of segment `seg`.

    `delta_rel` is the offset as a fraction of the domain diameter, again so that realization is
    smooth in the shape: freezing an absolute offset would not scale with the domain, and
    recomputing the offset from local thickness would inherit that quantity's kinks.
    """
    seg: int
    tau0: float
    tau1: float
    n_src: int
    delta_rel: float
    delta_max_rel: float = 0.0   # geometric ceiling on the offset, for refinement to grow into
    arclen: float = 0.0          # at plan time; sets the conditioning ceiling on n_src


@dataclass(frozen=True)
class BasisPlan:
    corners: tuple
    arcs: tuple
    target: float
    lam_max: float
    diameter: float
    cfg: PlanConfig = field(default_factory=PlanConfig)
    capped: bool = False
    shortfall: str = ''
    refinements: int = 0

    @property
    def n_fb(self):
        return sum(c.M for c in self.corners)

    @property
    def n_fs(self):
        return sum(a.n_src for a in self.arcs)

    @property
    def n_total(self):
        return self.n_fb + self.n_fs

    @property
    def Lambda(self):
        return log(1.0/self.target)

    def summary(self):
        out = [f'BasisPlan  target={self.target:.1e}  Lambda={self.Lambda:.1f}  '
               f'lam_max={self.lam_max:.4g}  kappa={sqrt(self.lam_max):.3f}  '
               f'refinements={self.refinements}',
               f'{"corner":>7} {"omega/pi":>9} {"alpha":>7} {"clear":>8} {"R":>8} '
               f'{"nu_osc":>8} {"nu_cont":>8} {"M":>5}']
        for c in self.corners:
            out.append(f'{c.corner:>7} {pi/c.alpha/pi:>9.3f} {c.alpha:>7.3f} '
                       f'{c.clearance:>8.3f} {c.R:>8.3f} {c.nu_osc:>8.1f} {c.nu_cont:>8.1f} '
                       f'{c.M:>5}')
        out.append(f'{"seg":>7} {"tau0":>7} {"tau1":>7} {"n_src":>6} {"delta/D":>8}')
        for a in self.arcs:
            out.append(f'{a.seg:>7} {a.tau0:>7.3f} {a.tau1:>7.3f} {a.n_src:>6} '
                       f'{a.delta_rel:>8.4f}')
        out += [f'FB terms         : {self.n_fb}',
                f'FS sources       : {self.n_fs}  over {len(self.arcs)} arcs',
                f'TOTAL columns    : {self.n_total}  (cap {self.cfg.n_cap})']
        if self.capped:
            out.append(f'CAPPED: {self.shortfall}')
        elif self.shortfall:
            out.append(f'OVER CAP (not thinned): {self.shortfall}')
        return '\n'.join(out)


# ── geometry primitives ──────────────────────────────────────────────────────────────────────

def _dense_boundary(domain, n_probe):
    segs = domain.bdry.segments
    lens = np.array([s.len for s in segs])
    counts = np.maximum(8, np.round(n_probe*lens/lens.sum()).astype(int))
    return np.concatenate([s.p((np.arange(c) + 0.5)/c) for s, c in zip(segs, counts)])


def local_thickness(domain, pts, normals, cone=0.5, n_probe=1500, samples=None):
    """Distance from each boundary point to the nearest boundary piece *facing it across the
    interior*: the smallest `|q - p|` over dense samples `q` whose direction from `p` lies within
    the inward cone `cos angle(q - p, -n) > cone`.

    Promoted from `benchmarks/basis_lab/placement.py`, where the cone was shown to be load-bearing:
    an earlier version excluded boundary within a fixed arclength window instead, and then every
    domain reported a thickness equal to the exclusion radius -- the parameter, not the domain.
    `inf` where the cone is empty (a convex domain sees nothing across itself from some points).
    """
    q = _dense_boundary(domain, n_probe) if samples is None else samples
    out = np.empty(len(pts))
    for i, (p, n) in enumerate(zip(pts, normals)):
        d = q - p
        r = np.abs(d)
        good = r > 1e-12
        cosang = np.where(good, (d.real*(-n.real) + d.imag*(-n.imag))/np.where(good, r, 1.0), -1.0)
        sel = good & (cosang > cone)
        out[i] = r[sel].min() if sel.any() else np.inf
    return out


def exterior_clearance(domain, pts, normals):
    """How far the outward normal ray from each point travels before it meets the boundary again.

    This is what makes a source *exterior*, and it is a different quantity from
    `local_thickness`, which looks inward. A source placed further out than this has left the
    exterior and re-entered the domain -- which voids both the tension and any certified bound,
    and is exactly the failure the old recipe hit on sharp thin triangles (`iso_tri_h4` dropped 6
    of 621 sources at `precision=1e-10`). Bounding the offset by this beforehand is better than
    detecting it afterwards.

    Exact for polygons: analytic ray/segment intersection over every edge, ignoring hits at
    parameter ~0 (the ray's own origin).
    """
    segs = domain.bdry.segments
    a = np.array([s.p0 for s in segs])
    b = np.array([s.pf for s in segs])
    out = np.empty(len(pts))
    for i, (p, n) in enumerate(zip(pts, normals)):
        # solve p + t n = a + u (b - a), t > 0, u in [0, 1]
        e = b - a
        denom = n.real*e.imag - n.imag*e.real
        ok = np.abs(denom) > 1e-14
        w = a - p
        t = np.where(ok, (w.real*e.imag - w.imag*e.real)/np.where(ok, denom, 1.0), -1.0)
        u = np.where(ok, (w.real*n.imag - w.imag*n.real)/np.where(ok, denom, 1.0), -1.0)
        hit = ok & (t > 1e-9) & (u >= -1e-12) & (u <= 1 + 1e-12)
        out[i] = t[hit].min() if hit.any() else np.inf
    return out


def _corner_table(domain):
    """`(alpha, seg_in, seg_out, clearance)` per corner, from `eigfun_integrals.corner_specs`
    (which already resolves the corner-index vs segment-index trap) plus `corner_clearance`
    (distance to the nearest NON-adjacent boundary piece)."""
    specs = corner_specs(domain)
    rows = []
    for s in specs:
        c = corner_clearance(domain, s.point, s.seg_out, s.seg_in)
        rows.append((float(s.nu), int(s.seg_in), int(s.seg_out), float(c)))
    return rows


# ── planning ─────────────────────────────────────────────────────────────────────────────────

def _validate(domain):
    if not isinstance(domain, BaseDomain):
        raise TypeError("'domain' must be a Domain object")
    if not domain.bdry.is_polyline:
        raise TypeError('basis_plan is restricted to polygons (domain.bdry.is_polyline)')
    if len(domain.corners) == 0:
        raise ValueError('domain must have at least one corner')


def _fb_ceiling(alpha, cfg):
    """Most Fourier-Bessel terms a corner can carry before the top ones stop being independent.

    Term `j` behaves like `r^(j*alpha)`; normalized by its maximum over the owned arc it is
    `(r/R)^(j*alpha)`, which has fallen below the solver's resolution by
    `r = R*exp(-_indep_digits/(j alpha))`.
    Once that radius is inside the region the collocation actually samples, the column is
    numerically zero at every point but the outermost few, and near-duplicate columns destabilize
    the rank truncation in the same way over-packed sources do.

    `cfg.fb_inner_frac` is the fraction of the owned radius the collocation is assumed to resolve
    down to, and it is **calibrated, not derived** -- the one constant here that was fitted. It was
    swept against certified digits over nine domains (with the ceiling constant at machine
    epsilon, which `_indep_digits` has since replaced):

        fb_inner_frac   0.2    0.35   0.5    0.7
        iso_tri_h4      3.9    5.8    7.1    7.1
        iso_tri_h16     3.8    5.4    5.8    7.2
        (all others unchanged)

    0.2 is far too tight: it caps `M` below what a reentrant corner needs and then blocks the
    refinement loop from growing anything. The default 0.77 preserves that measured cap under the
    smaller `_indep_digits`, since only the ratio `_indep_digits/ln(1/fb_inner_frac)` matters.

    The ceiling still matters at SHARP
    corners, where an uncapped count lets `alpha = 9.76` reach `r^39` -- growing those turned
    `chevron_1_2` from 6.1 certified digits at n=120 into 3.2 at n=172 before this existed.
    """
    return max(1, int(_indep_digits(cfg)/(log(1.0/cfg.fb_inner_frac)*alpha)))


def _corner_blocks(domain, kappa, Lambda, cfg):
    """One Fourier-Bessel block per corner, sized by a single continuous rule.

    `nu_osc` resolves oscillation across the owned arc; `nu_cont` is the number of terms the
    series needs for `(R/clearance)^nu <= exp(-Lambda)`, i.e. to converge on its own arc given
    that the nearest thing it cannot represent sits at `clearance`. Because `R = gamma*clearance`
    unless an edge-length cap binds, `nu_cont` is usually `Lambda/ln(1/gamma)` -- and where the cap
    does bind, `R/clearance` is smaller and `nu_cont` falls out correspondingly. No branch, no
    threshold, and `M -> 1` continuously as the corner gets sharp (large `alpha`).
    """
    lens = np.array([s.len for s in domain.bdry.segments])
    blocks = []
    for i, (alpha, seg_in, seg_out, clear) in enumerate(_corner_table(domain)):
        # A SHARP corner cannot own a full arc. Its exponents are j*alpha, so the lowest term is
        # `r^alpha`, and for alpha = 9.76 that is below a tenth of its edge value over ~80% of the
        # arc: the block is only useful within a whisker of the tip, while the ownership partition
        # forbids sources from covering the rest. Measured on `chevron_1_2` at target 1e-10 --
        # shrinking only the over-claiming corners and giving the freed boundary to sources:
        #
        #     shrink   1.00    0.60    0.40    0.25
        #     digits   6.40    8.25    9.20    9.76      (contrast 4e7 -> 3e10)
        #
        # `sharp_ref/alpha` is a FITTED functional form, not a derived one: it reproduces the
        # measured optimum and is monotone in the right direction, which is as much as the evidence
        # supports. Corners with alpha <= sharp_ref (a right angle or blunter) are unaffected, which
        # is why 17 of 25 suite polygons do not change at all.
        #
        # `sharp_ref = 2.0` was swept on the dev set and CONFIRMED ON THE HOLDOUT before adoption
        # (`plan_lab sharp`): 6 domains better by >0.3 digits, 1 worse, at +0% median columns --
        # chevron_2_4 0.95 -> 6.64, chevron_2_3 1.90 -> 6.06, chevron_1_2 6.40 -> 10.29,
        # iso_tri_h16 5.19 -> 7.05, against right_trapezoid 13.83 -> 13.33 as the only regression.
        # 3.4 was the other candidate and lost: it cost iso_tri_h05 1.5 digits.
        sharp = min(1.0, cfg.sharp_ref/alpha)
        R = sharp*min(cfg.gamma*clear, cfg.reach_frac*lens[seg_in], cfg.reach_frac*lens[seg_out])
        kR = kappa*R
        nu_osc = kR + cfg.airy*kR**(1.0/3.0) + cfg.order_margin
        ratio = clear/R if R > 0 else np.inf
        nu_cont = Lambda/log(ratio) if ratio > 1.0 + 1e-12 else np.inf
        M = max(1, min(int(ceil((nu_osc + nu_cont)/alpha)), _fb_ceiling(alpha, cfg)))
        blocks.append(CornerBlock(corner=i, M=M, alpha=alpha,
                                  reach_prev=R/lens[seg_in], reach_next=R/lens[seg_out],
                                  R=R, clearance=clear, nu_osc=nu_osc, nu_cont=nu_cont))
    return blocks


def _free_arcs(domain, blocks):
    """The parts of each edge no corner owns, as `(seg, tau0, tau1)`.

    Corner `i` owns `reach_next` of the start of its outgoing edge and `reach_prev` of the end of
    its incoming edge. Where the two reaches meet or overlap, the edge is fully owned and no arc
    is emitted; the corner blocks are then responsible for all of it, which is the point of making
    ownership a partition.
    """
    table = _corner_table(domain)
    n_seg = len(domain.bdry.segments)
    head = np.zeros(n_seg)      # fraction of each segment owned from its start
    tail = np.zeros(n_seg)      # ...and from its end
    for b, (alpha, seg_in, seg_out, clear) in zip(blocks, table):
        head[seg_out] = max(head[seg_out], b.reach_next)
        tail[seg_in] = max(tail[seg_in], b.reach_prev)
    arcs = []
    for s in range(n_seg):
        t0, t1 = head[s], 1.0 - tail[s]
        if t1 - t0 > 1e-6:
            arcs.append((s, float(t0), float(t1)))
    return arcs


def _arc_blocks(domain, arcs, kappa, Lambda, cfg):
    """Source count and offset for each free arc, from LOCAL geometry only.

    `delta` is bounded three ways: by the diameter (so a big smooth domain does not put sources
    absurdly far out), by the interior thickness (so the two sides of a thin neck do not fight),
    and by the exterior clearance along the normal (so the source is actually outside). `h` is the
    coarser of the Nyquist limit and the MFS decay requirement `2 pi delta / Lambda`. Nothing here
    refers to any other arc, which is the whole design change: the old global rule refined every
    edge because *some* corner image was near.
    """
    if not arcs:
        return []
    segs = domain.bdry.segments
    D = domain.diameter
    mids = np.array([segs[s].p(np.array([0.5*(t0 + t1)]))[0] for s, t0, t1 in arcs])
    nrms = np.array([segs[s].N(np.array([0.5*(t0 + t1)]))[0] for s, t0, t1 in arcs])
    samples = _dense_boundary(domain, cfg.n_probe)
    thick = local_thickness(domain, mids, nrms, cfg.cone, samples=samples)
    ext = exterior_clearance(domain, mids, nrms)

    h_nyq = 2*pi/(cfg.nyquist_ppw*kappa)
    out = []
    for (s, t0, t1), th, ex in zip(arcs, thick, ext):
        delta = min(cfg.delta_frac_D*D, cfg.delta_frac_thick*th, cfg.delta_frac_ext*ex)
        # The HARD geometric limit, as opposed to the conservative choice above: refinement may
        # trade offset for source count up to here, but never past it -- beyond `ex` the source
        # would re-enter the domain, and beyond `th` the two sides of a thin region interfere.
        delta_max = min(cfg.delta_hard_D*D, cfg.delta_hard_frac*th, cfg.delta_hard_frac*ex)
        h = min(h_nyq, 2*pi*delta/Lambda)
        arclen = (t1 - t0)*segs[s].len
        n_src = max(cfg.min_src_per_arc, int(ceil(arclen/h)))
        out.append(ArcBlock(seg=s, tau0=t0, tau1=t1,
                            n_src=min(n_src, _src_ceiling(arclen, delta/D, D, cfg)),
                            delta_rel=delta/D, delta_max_rel=max(delta, delta_max)/D,
                            arclen=arclen))
    return out


def _src_ceiling(arclen, delta_rel, D, cfg):
    """Most sources an arc can carry before they stop being independent.

    A source layer standing off by `delta` with spacing `h` represents the field to about
    `exp(-2 pi delta / h)`, so `2 pi delta / h` is the accuracy the layer can deliver -- and it is
    useless past `ln(1/eps_machine) ~ 36`, because beyond that adjacent columns are equal to
    working precision. Packing more in does not merely waste them: near-duplicate columns
    destabilize the rank truncation in `mps.regularize_pencil`, so the basis gets *worse*.

    Measured, on `chevron_1_2` at target 1e-7, by a refinement loop that grew `n_src` at fixed
    `delta` without this ceiling:

        n = 120  worst arc residual 7.3e-07     (just short of target)
        n = 172                     6.2e-04
        n = 206                     8.3e-01     certified digits 6.1 -> 1.2

    So the refinement direction is not free: an arc that is already at its ceiling must be given a
    larger offset (which raises the ceiling) or more Fourier-Bessel help, never more sources.
    """
    delta = delta_rel*D
    if delta <= 0:
        return cfg.min_src_per_arc
    h_min = 2*pi*delta/_indep_digits(cfg)
    return max(cfg.min_src_per_arc, int(arclen/h_min))


def _apply_cap(corners, arcs, cfg):
    """Scale the source budget back to `cfg.n_cap` rather than build past the rank ceiling.

    Sources are thinned before Fourier-Bessel terms because the corner blocks are what the
    measurements say carry the accuracy: `pure_fb` at matched size beat every mixed construction
    on the easy domains, and the source layer is where the old recipe's waste lived. If the FB
    budget alone exceeds the cap the plan reports a shortfall -- a domain this module cannot serve
    at this target is worth saying out loud, not worth silently under-serving.

    `capped` MEANS "THE SOURCE BUDGET WAS ACTUALLY REDUCED", and nothing weaker. It used to mean
    "the cap was consulted and the total was over it", which made it true of plans that were then
    served the full budget anyway -- GWW1 and H_shape reported `capped=True` at 39% and 88% OVER
    the nominal ceiling, because the refusal branch returns the arcs untouched. The regular N-gon
    at N >= 8 hit the same thing from the other side: its sources are already at
    `min_src_per_arc`, so there is nothing left to thin and the thinning branch returned a
    byte-identical plan flagged `capped=True`. Sweeping `n_cap` over {150 ... 600} at N >= 8
    produces the same plan and the same digits to every figure while the flag flips at 240, which
    is how the flag came to be read as a cause of lost accuracy that it never was.

    A plan that exceeds the cap without being thinned still says so in `shortfall`; that is the
    honest signal, and it is reported independently of `capped`.
    """
    n_fb = sum(c.M for c in corners)
    n_fs = sum(a.n_src for a in arcs)
    if n_fb + n_fs <= cfg.n_cap:
        return arcs, False, ''
    room = cfg.n_cap - n_fb
    if room < cfg.min_src_per_arc*len(arcs):
        return (arcs, False,
                f'Fourier-Bessel budget alone is {n_fb} columns against a cap of {cfg.n_cap}; '
                f'{n_fs} sources requested and served in full ({n_fb + n_fs} columns, '
                f'{100*(n_fb + n_fs)/cfg.n_cap - 100:.0f}% over). Raise cfg.n_cap, loosen target, '
                f'or accept that this geometry is not servable at this target.')
    scale = room/n_fs
    thinned = tuple(replace(a, n_src=max(cfg.min_src_per_arc, int(a.n_src*scale)))
                    for a in arcs)
    n_thinned = sum(a.n_src for a in thinned)
    if n_thinned == n_fs:
        return (arcs, False,
                f'{n_fb + n_fs} columns against a cap of {cfg.n_cap}, but every arc is already at '
                f'min_src_per_arc={cfg.min_src_per_arc}: nothing to thin, budget served in full.')
    return (thinned, True,
            f'source budget thinned {n_fs} -> {n_thinned} to respect '
            f'cfg.n_cap={cfg.n_cap} (rank saturation ceiling)')


def plan_basis(domain, lam_max, target=1e-7, cfg=None):
    """Decide the discrete content of a basis for `domain`. See the module docstring."""
    _validate(domain)
    if not 0 < target < 1:
        raise ValueError("'target' must be in (0, 1)")
    cfg = cfg or PlanConfig()
    kappa = sqrt(lam_max)
    Lambda = log(1.0/target)

    corners = _corner_blocks(domain, kappa, Lambda, cfg)
    arcs = _arc_blocks(domain, _free_arcs(domain, corners), kappa, Lambda, cfg)
    arcs, capped, shortfall = _apply_cap(corners, arcs, cfg)
    return BasisPlan(corners=tuple(corners), arcs=tuple(arcs), target=float(target),
                     lam_max=float(lam_max), diameter=float(domain.diameter), cfg=cfg,
                     capped=capped, shortfall=shortfall)


# ── realization ──────────────────────────────────────────────────────────────────────────────

def realize(plan, domain, check_exterior=True):
    """Build the basis `plan` describes on `domain`.

    Cheap and smooth in the geometry: term counts and arc endpoints come from the frozen plan, and
    only positions are recomputed. `n_basis` is therefore identical for every domain a given plan
    is realized on, which is what lets a shape optimizer trust that a change in `lambda` came from
    the shape rather than from the basis.
    """
    _validate(domain)
    segs = domain.bdry.segments
    D = domain.diameter

    sources = []
    for a in plan.arcs:
        seg = segs[a.seg]
        u = a.tau0 + (a.tau1 - a.tau0)*(np.arange(a.n_src) + 0.5)/a.n_src
        sources.append(seg.p(u) + (a.delta_rel*D)*seg.N(u))
    src = np.concatenate(sources) if sources else np.empty(0, dtype=complex)

    if check_exterior and len(src):
        inside = np.asarray(domain.contains(src), dtype=bool)
        if inside.any():
            raise ValueError(
                f'{int(inside.sum())} of {len(src)} sources landed inside the domain. The offset '
                f'is bounded by exterior_clearance at plan time, so this means the domain moved '
                f'far enough from the planned one to invalidate the plan -- re-plan rather than '
                f'drop sources, which would void the tension and any certified bound.')

    fb_orders = np.zeros(len(domain.corners), dtype=int)
    for c in plan.corners:
        fb_orders[c.corner] = c.M

    fb = FourierBesselBasis.from_domain(domain, fb_orders) if fb_orders.sum() else None
    fs = FundamentalBasis(src, 1) if len(src) else None
    if fb is None and fs is None:
        raise ValueError('plan is empty')
    basis = fs if fb is None else (fb if fs is None else fb + fs)
    # Tag, so that code holding only a solver can ask what plan produced it (`plan_of`). This is
    # introspection, not part of the basis contract -- nothing in `bases.py` knows or cares.
    basis._basis_plan = plan
    return basis


def plan_of(obj):
    """The `BasisPlan` that produced `obj`, or None.

    Walks `.basis` because the solver wraps what `realize` returned: `from_domain` calls
    `to_normalized`, so `solver.basis` is a `NormalizedBasis` around the tagged object.
    """
    for _ in range(8):
        plan = getattr(obj, '_basis_plan', None)
        if plan is not None:
            return plan
        obj = getattr(obj, 'basis', None)
        if obj is None:
            return None
    return None


# ── measured refinement ──────────────────────────────────────────────────────────────────────

def _graded(t0, t1, n, toward):
    """`n` parameters in `[t0, t1]`, clustered cubically toward one end (`toward` in {0, 1, None}).

    Grading matters more than the point count. A boundary residual on a domain with corners peaks
    within a few percent of the corner, and `benchmarks/reference/certify.boundary_sup` grades its
    400-per-segment sampling by `t = 0.5 s^3` for exactly that reason. An earlier version of this
    function sampled 48 points uniformly and reported 4e-8 on `chevron_1_2` where the certified
    bound was 6.1 digits -- it was missing the peak, which would have let `refine_plan` stop while
    believing it had converged.
    """
    s = (np.arange(n) + 0.5)/n
    if toward == 0:
        f = s**3
    elif toward == 1:
        f = 1.0 - (1.0 - s)**3
    else:
        f = np.where(s < 0.5, 0.5*(2*s)**3, 1.0 - 0.5*(2 - 2*s)**3)
    return t0 + (t1 - t0)*f


def residual_by_arc(plan, domain, solver, lams, n_probe=160):
    """`(arc_residuals, corner_residuals)`, each the WORST over the eigenvalues in `lams`.

    `lams` may be a scalar or a sequence. Which to use depends on what the basis is for, and the
    two cases pull in opposite directions:

    * a shape-optimization loop tracks ONE eigenvalue as the domain moves, so refining at that one
      is exactly right and refining at others spends columns on modes nobody asked for;
    * `Eigenproblem.solve(k)` wants a whole window, and higher modes oscillate faster, so a plan
      refined only at `lam_1` under-serves `lam_k`.

    Passing the eigenvalues you actually care about resolves it without a policy baked in here.
    """
    lams = np.atleast_1d(np.asarray(lams, dtype=float))
    arc, cor = None, None
    for lam in lams:
        a, c = _residual_by_arc(plan, domain, solver, float(lam), n_probe=n_probe)
        arc = a if arc is None else np.maximum(arc, a)
        cor = c if cor is None else np.maximum(cor, c)
    return arc, cor


def _residual_by_arc(plan, domain, solver, lam, n_probe=160):
    """Each block's own contribution to the Moler--Payne bound, so that "residual <= target" means
    "this block would certify the target".

    `eps = sqrt(|Omega|) * ||u||_Linf(dOmega) / ||u||_L2(Omega)`, and with `orthonorm=True` the
    denominator is 1 by construction, so the per-block quantity is just `sqrt(area) * sup|u|` over
    that block's arc. Measuring the refinement loop in the objective's own currency is what stops
    it from converging on a proxy: the interior-collocation normalization it used before ran about
    two digits optimistic on elongated domains.

    Costs one basis evaluation per arc and no new factorization -- `eigenfunction_coef` is cached,
    and `eigenfunction` evaluates the already-minimized coefficient vector anywhere.
    """
    u = solver.eigenfunction(lam, mult=1, orthonorm=True)
    scale = sqrt(domain.area)
    segs = domain.bdry.segments

    def sup_on(seg_idx, t0, t1, toward=None, n=n_probe):
        t = _graded(t0, t1, n, toward)
        return float(scale*np.abs(u(segs[seg_idx].p(t))).max())

    arc_res = [sup_on(a.seg, a.tau0, a.tau1) for a in plan.arcs]
    corner_res = []
    for c, (alpha, seg_in, seg_out, clear) in zip(plan.corners, _corner_table(domain)):
        # the corner sits at tau=0 of its outgoing edge and tau=1 of its incoming one
        r = max(sup_on(seg_out, 0.0, max(c.reach_next, 1e-6), toward=0),
                sup_on(seg_in, 1.0 - max(c.reach_prev, 1e-6), 1.0, toward=1))
        corner_res.append(r)
    return np.array(arc_res), np.array(corner_res)


def _grow_arc(arc, domain, cfg):
    """Give a short arc more resolution without pushing it past `_src_ceiling`.

    Below the ceiling, add sources. At the ceiling, raise the offset instead -- which raises the
    ceiling, since it scales with `delta` -- up to the geometric bound `delta_max_rel` that plan
    time established from local thickness and exterior clearance. At both ceilings the arc is
    returned unchanged, and `refine_plan` then sees no growth and stops rather than degrading the
    basis.
    """
    D = domain.diameter
    ceiling = _src_ceiling(arc.arclen, arc.delta_rel, D, cfg)
    want = max(arc.n_src + 1, int(ceil(cfg.refine_growth*arc.n_src)))
    if want <= ceiling:
        return replace(arc, n_src=want)
    grown_delta = min(cfg.refine_growth*arc.delta_rel, arc.delta_max_rel)
    if grown_delta > arc.delta_rel*(1 + 1e-9):
        new_ceiling = _src_ceiling(arc.arclen, grown_delta, D, cfg)
        return replace(arc, delta_rel=grown_delta, n_src=min(want, new_ceiling))
    return replace(arc, n_src=ceiling)


def refine_plan(plan, domain, solver_factory, lams, verbose=0):
    """Grow the plan where the boundary residual actually misses `target`, then stop.

    This is what makes `target` mean something. The old recipe predicted its accuracy from a
    closed form and was measured inert -- flat to 1.5 digits across four decades of requested
    precision on 12 of 18 domains. Here the plan is *measured* against its target before being
    frozen, which costs a handful of solves once per optimization run and is amortized over every
    iterate after that.

    `solver_factory(basis)` must return a solver for `domain` with that basis, and `lams` is the
    eigenvalue (or eigenvalues) the plan has to serve -- see `residual_by_arc` on why that choice
    belongs to the caller. Returns a new plan.

    **It returns the best plan it measured, never the last one.** Growth is not monotone in
    accuracy: on `chevron_2_4` a loop that trusted its last step went from 0.8 certified digits at
    n=191 having started at a better place, because adding columns to a block already at its
    conditioning limit degrades the pencil. The per-block ceilings (`_src_ceiling`, `_fb_ceiling`)
    are the first line of defence and this is the second -- with both, refinement can only improve
    on where it started, which is the property that makes it safe to run unattended.
    """
    cfg = plan.cfg
    best, best_res, best_it = plan, np.inf, 0
    for it in range(cfg.max_refine + 1):
        basis = realize(plan, domain)
        solver = solver_factory(basis)
        arc_res, corner_res = residual_by_arc(plan, domain, solver, lams)
        worst = max(arc_res.max(initial=0.0), corner_res.max(initial=0.0))
        if worst < best_res:
            best, best_res, best_it = plan, worst, it
        if verbose:
            print(f'  refine {it}: n={plan.n_total} worst residual {worst:.2e} '
                  f'(target {plan.target:.1e})' + ('  <- best so far' if plan is best else ''))
        if worst <= plan.target:
            return replace(plan, refinements=it)
        if it == cfg.max_refine:
            break

        arcs = tuple(_grow_arc(a, domain, cfg) if r > plan.target else a
                     for a, r in zip(plan.arcs, arc_res))
        corners = tuple(replace(c, M=min(max(c.M + 1, int(ceil(cfg.refine_growth*c.M))),
                                         _fb_ceiling(c.alpha, cfg)))
                        if r > plan.target else c
                        for c, r in zip(plan.corners, corner_res))
        arcs, capped, shortfall = _apply_cap(corners, arcs, cfg)
        grown = replace(plan, arcs=arcs, corners=corners, capped=capped, shortfall=shortfall)
        if grown.n_total <= plan.n_total:
            if verbose:
                print('  refine: no block can grow (every one is at a ceiling); stopping')
            break
        plan = grown

    return replace(best, refinements=best_it, capped=True,
                   shortfall=(f'target {plan.target:.1e} not reached; best measured residual '
                              f'{best_res:.2e} at n={best.n_total}'))


def polygon_default_basis(domain, lam_max, target=1e-7, cfg=None):
    """One-shot facade: plan and realize in a single call, no refinement.

    For the inner loop, call `plan_basis` (once, optionally through `refine_plan`) and then
    `realize` per iterate -- that is the whole point of the split, and this convenience wrapper
    throws it away.
    """
    return realize(plan_basis(domain, lam_max, target, cfg), domain)
