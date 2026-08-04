# The benchmark domain suite

A curated, feature-indexed sample of planar domains: 44 entries across 20
parametric families, chosen so that every way a domain can be hard for the
method of particular solutions is represented, and so that each mechanism can be
*swept* rather than merely sampled.

This directory contains no solver harness and runs no eigenvalue computations.
It answers one question — **which domains should we be testing on, and why** —
and nothing else.

- `domains.py` — the registry. One `SuiteDomain` per entry.
- `features.py` — geometric metadata derived from a built domain.
- `report.py` — renders `SUITE.md`. Run `python -m benchmarks.suite.report --write`.
- `SUITE.md` — **generated**; the full table, rationale, coverage matrix, and
  the symmetry to-do list. Do not edit it by hand.

## Why this exists

`benchmarks/reference/produce.py` had 26 domains that accumulated as a
reference-value production list. It was redundant along some axes (seven
`iso_tri` heights, four `chevron` parameters) and empty along others, and the
record of *why* each domain was hard lived scattered across `TUNING_LOG.md` and
per-domain script docstrings rather than organized by mechanism.

The suite reorganizes that around the difficulty taxonomy below, adds the
mechanisms that were missing, and — crucially — adds domains where the exact
answer is known for corner geometries that previously had only MPS values to
check against.

## The taxonomy

### 1. Corner exponent

Near a corner of interior angle `gamma`, solutions of the Helmholtz equation
behave like `r**(m*p)` where

    p = pi / gamma

`p` is the single most useful number to know about a domain. When `p` is a
positive integer the corner is **regular**: the expansion is analytic and the
corner costs nothing. Otherwise it is **singular**, in one of two opposite ways.

**Reentrant (`p < 1`, `gamma > pi`).** The leading exponent is small, so any
approximation by smooth functions converges only algebraically. This is the
classic MPS accuracy problem. `L_shape` and `H_shape` both sit at `p = 2/3`;
the worst possible case is the slit, `p -> 1/2`.

**Sharp (`p >> 1`, `gamma -> 0`).** Bessel orders `m*p` grow with the order
index, so evaluation becomes expensive and eventually overflows. This is a
**cost** problem, not an accuracy problem, and the distinction matters: the
tuning log records that reweighting the basis toward a sharp corner made
`chevron` *worse*, because the natural harmonics at an 11-degree corner already
start at exponent ~16, and asking for more just reaches further up a sparse
ladder. `chevron(1,1.25)`'s 6.3-degree corners reach `p ~ 28` and do not
converge at any configuration tried.

Note the regularity qualifier on "sharp". A 45-degree corner has `p = 4` but is
exactly representable, so it is not sharp in any sense that costs anything. This
is why the suite uses `disk_sector(1, pi/6.5)` rather than `pi/6`, and
`parallelogram(1, 1, pi/12.7)` rather than `pi/12` — at the round values the
corners would be regular and the entry would test nothing.

**Count matters as much as severity.** One reentrant corner (`L_shape`) is
handled to 13 digits. Four (`H_shape`) gets ~8. GWW's four, with no symmetry to
exploit, gets 7-9. This is the strongest predictor of achieved accuracy in the
tuning log and is why the suite carries `many_singular` and `many_reentrant`
separately from `reentrant`.

### 2. Smoothness

Fully smooth boundaries (`disk`, `ellipse`) route to a pure `FundamentalBasis`
and behave completely differently — `ellipse_a2` is the most accurate domain in
the suite at 14 digits.

Between smooth and cornered sit **curvature discontinuities**: junctions where
the boundary is C^1 but not C^2. The stadium has four of them, no corners at
all, and is stuck at 2-3 digits with denser collocation demonstrably not
helping. Such a junction is neither a corner the Fourier–Bessel basis can be
aimed at nor the smooth boundary the fundamental-solution basis wants.

`cut_square` is worth singling out. Building the suite established, contrary to
expectation, that **it has no singular corners at all** — all five corners are
regular `p = 2`, and the arc meets the edges at right angles rather than
tangentially. Yet `r=0.25` reaches only ~9 digits while `r=0.5` reaches 13. So
whatever limits it is a property of mixing an arc into a polygon, isolated from
corner singularity entirely. That makes it one of the more informative entries
in the suite, for a reason nobody wrote down before.

### 3. Elongation

Measured as `slenderness = diameter / (2 * inradius)`, which is 1.0 for a disk,
2.0 for a 2:1 ellipse, 8.0 for a 1x8 rectangle. Affects conditioning, branch-cut
ray geometry, and eigenvalue spacing. Swept by `iso_tri` (h = 0.5 to 16),
`ellipse` (a = 2 to 4), and `parallelogram`.

### 4. Spectral structure

Exact multiplicity from symmetry (`disk`, `square`, `eq_tri`, the regular
n-gons); genuinely near-degenerate but distinct pairs (`rect(1, 1+delta)`, a
knob with exact truth at both settings); crowded clusters (`reg_ngon_8`, where a
shallow local tension minimum was once accepted into the reference table as a
spurious eigenvalue); localized modes in weakly-coupled subregions
(`mushroom_neck01`); and isospectrality (the GWW pair, whose agreement is a
reference check needing no external table).

### 5. Symmetry availability

`lappy/symmetry.py` gives a large speedup where a real character group exists,
so the suite deliberately includes domains with none — `GWW1`, `GWW2`,
`right_trapezoid`, `spiral` — as the baseline against which symmetry speedups
should be quoted.

Building the suite surfaced a gap here: `rect`, `eq_tri`, `iso_right_tri`,
`disk`, `disk_sector` and `parallelogram` all have obvious symmetry groups that
`domain_symmetry` simply has no entry for. `SUITE.md` lists them under
"Symmetry gaps" as a to-do list. The registry distinguishes the two cases: the
`no_symmetry` tag means genuinely asymmetric, while `symmetric=True` with no
registered group means unregistered.

### 6. Branch-cut geometry

`spiral` is the only family whose corners can lack a straight-ray sightline to
infinity — 12 of 24 corners are blocked at the default parameters — forcing
`corner_branch_cut_polyline`. It was benchmarked nowhere before this.

## Tiers and status

Entries carry a `tier` (what kind of domain it is) and a `status` (how well the
method currently does on it).

**`tier='analytic'`** — closed-form spectra from `lappy/reference.py`, so method
error is separable from reference error. The most valuable addition here is the
`disk_sector` family: `reference.sector_eigs` accepts an arbitrary opening
angle, so it sweeps the *entire* corner-exponent axis — from the near-slit
`p = 0.504` through the reentrant `p = 2/3` (the same exponent as `L_shape`) up
to sharp `p = 13.3` — against exact truth. Before this, reentrant and sharp
convergence could only be measured against other MPS runs.

**`tier='corner'`** — polygons whose difficulty is corner structure.

**`tier='curved'`** — curved and mixed boundaries, where the difficulty is
smoothness.

**`status`** is the improvement-target axis, orthogonal to tier:

- `ok` — converges to target accuracy.
- `hard` — converges, but short of target. `H_shape`, `GWW1`, `GWW2`,
  `reg_ngon_8`, all four `chevron` parameters, `iso_tri_h16`, `stadium`,
  `cut_square_r025`, the sharper `parallelogram` and `sector` entries.
- `open` — does not converge usefully at any configuration tried:
  `chevron_1_125` (6.3-degree corners) and the two `spiral` entries (blocked
  branch cuts).

`hard` and `open` entries are *expected* to underperform. They are in the suite
precisely because they are the targets.

## Keeping it honest

Every geometric claim in `domains.py` is cross-checked against the geometry.
`features.derive_tags` computes the geometric tags from the built domain, and
`domains.validate()` reports any disagreement with what the registry declares,
in either direction. Spectral tags (`exact_multiplicity`, `clustered`,
`near_degenerate`, `isospectral`, `thin_neck`) are not derivable from a boundary
and are declared on trust.

This is not ceremony — it caught three real errors while the suite was being
written: the belief that `cut_square` has tangential fillets, the belief that a
30-degree corner is hard, and `iso_tri(1)` turning out to be the right isoceles
triangle and therefore to have an exact spectrum, which is now recorded.

Validation runs automatically as the first thing `report.py` does:

    python -m benchmarks.suite.report          # print
    python -m benchmarks.suite.report --write  # regenerate SUITE.md

## Using the registry

```python
from benchmarks.suite.domains import SUITE, CORE, select

select(tag='reentrant')                  # every reentrant-corner domain
select(tier='analytic')                  # everything with exact truth
select(status=('hard', 'open'))          # the improvement targets
select(family='chevron')                 # one parametric sweep
[SUITE[k] for k in CORE]                 # 17-domain spanning subset
```

`CORE` is a minimal subset covering every tag at least once, for when running
all 44 is too expensive.

`for_reference_production()` returns the `produce.py` view — the entries whose
eigenvalues come from MPS rather than a closed form — in the
`(builder, symmetry, n_basis, n_eigs)` shape that script already expects.

## Scope

Dirichlet only. Every entry is simply connected, because `lappy` has no
multiply-connected domain support. Non-Dirichlet boundary conditions
(`Domain.to_bc`, per-segment `bc`) and multiply-connected geometry are the two
obvious future expansions, and both need work in `lappy/geometry.py` first.
