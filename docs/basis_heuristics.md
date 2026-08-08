# Toward `make_default_basis(domain, lam_max, precision)`

**Status: a proposal with partial evidence, 2026-08-08.** Written after the bucket-2 basis
study (`benchmarks/suite/run/FINDINGS.md` sections 9 onward). The measurements are real; the
synthesis is a hypothesis that has been checked on a handful of domains and not validated
across the suite.

## Why that signature is the right target

`boundary_quadrature(domain, lam_max, precision)` already works this way and it earned it: the
node set is a pure function of geometry, the top of the spectral window, and a requested
accuracy, with no basis and no hand-tuned constants. A basis constructor can plausibly take the
same three arguments for the same reason — everything a basis needs to know is either geometric
(corners, thickness, exterior clearance), spectral (`lam_max`), or a target (`precision`).

Today's `make_default_basis(domain, n_basis, ...)` instead asks the caller for `n_basis`, which
is the one quantity they have no principled way to choose, plus seven tuning constants
(`fs_frac`, `fs_d`, `fs_C`, `fs_sigma`, `fs_bdry_order`, `fs_corner_order`). The study found
that two of those constants were badly wrong by default and that `n_basis` was the binding
constraint on two domains — i.e. the caller is being asked for exactly the things that
determine success.

## The one solid result: the offset scales with the wavelength

The dominant lever for a domain with several singular corners is *where the
fundamental-solution sources sit*, and the right offset is not a constant. Write
`h = 2*pi/sqrt(lam_max)` for the wavelength at the top of the window. Every optimum measured
in the study, in units of `h`:

    domain            d_opt   lam_max   h       d/h
    parallelogram_p65   0.30    235     0.41    0.73
    parallelogram_p127  0.40    320     0.35    1.14
    chevron_1_15        0.20    589     0.26    0.77
    chevron_1_125       0.15   1481     0.16    0.92
    chevron_2_4         0.40    146     0.52    0.77
    ellipse_a2          1.00     30.5   1.14    0.88
    stadium             0.10    107     0.61    0.16   <- outlier, see below

Seven of eight land in `0.73 <= d/h <= 1.14`, across domains whose absolute optimal offsets
differ by a factor of ten. Swept prospectively in wavelength units, the optimum is a broad
plateau rather than a point:

    parallelogram_p65   d/h:  0.25     0.50     0.75     1.00     1.50     2.50
                    sigma: 4.1e-11  1.9e-12  1.9e-12  1.9e-12  1.9e-12  2.7e-10
    chevron_1_15    sigma: 2.9e-10  2.4e-11  2.2e-11  1.1e-11  1.7e-12  2.4e-09

`d = h` sits inside the plateau on both, and on `chevron_1_15` the rule picks `d/h = 1.5`,
which is an order of magnitude better than the `d = 0.2` I had chosen by hand. **A heuristic
here would outperform hand-tuning, not merely match it.**

`stadium` is the informative exception. Its width is 1.0 against a wavelength of 0.61, so
`d = h` would place sources further out than the domain is thick; the geometric cap binds
first, and the measured optimum (0.1) is about a fifth of the thickness. That is not a
counterexample to the rule, it is the second term of it.

## The proposal

    d = min( c_wave * h ,  c_thick * local_thickness ,  c_ext * local_exterior_clearance )

with `c_wave ~ 1` (plateau 0.5-1.5), `c_thick ~ 0.2` (from `stadium`), and the third term
tapering to zero at a reentrant corner so that sources can never land inside -- the failure
that invalidated two results in the study and is now blocked in `FundamentalBasis`. Note the
third quantity is the **exterior** clearance; scaling by *interior* thickness was tried and
made the mushrooms worse, so the two must not be conflated.

The remaining pieces, in decreasing order of confidence:

**Source count.** Boundary resolution is a wavelength question: sources spaced about `h/2`
along the boundary gives `n_fs ~ 2 * perimeter / h`, scaled by a `precision` factor. Untested
as a formula, but it is the same Nyquist argument the quadrature's smooth panels already use.

**Fourier--Bessel orders per corner.** Already geometry-driven (`fb_corner_orders` weights by
angle). The study found that re-allocating them changes almost nothing on the domains where
placement matters -- default `[8, 178, 8, 45]` against `[80, 120, 80, 45]` and `[8, 300, 8, 45]`
all within a factor of 1.6 -- so this is *not* where the leverage is, and it can probably keep
its current rule. Making it precision-driven (how many terms to reach `precision` at radius
`h`) would be the natural upgrade.

**Total size.** The weakest link. `mushroom_thin` and `mushroom_neck01` needed nothing but a
larger basis -- 320 to 480 bought three orders with the default basis untouched -- and nobody
tried, because `entry.n_basis` looked authoritative. A `precision` argument only helps if it
actually drives the size, so this is the part that most needs a real model rather than a
default.

**Collocation ratio.** Not part of the basis, but it belongs in the same contract: the tension
stops tracking Moler--Payne below a boundary-to-column ratio of about 1.5, catastrophically
(at ratio 1.0, `sigma` reads 9.1e-04 where the true `eps` is 3.7e+01). `lappy`'s default
`mult=2` clears it with a 2x margin.

## Do not trust the formula: certify the outcome

The most transferable lesson from the quadrature work is that a sizing rule should be checked
against a computable consequence rather than believed. Two checks are already available and
cost nothing:

* **The tension background.** Moler--Payne forces `eps >= dist(lam, spectrum)/lam`, so away
  from eigenvalues `sigma` cannot be small; a median below ~1e-3 across the scan means the
  basis or the collocation is broken (`preflight.background_suspect`). This catches both
  interior sources and under-collocation.
* **`n_reg/n`.** How much of the basis survives regularization. Useful as a *diagnostic* and
  emphatically not as an objective -- thickness-scaled placement raised it from 181 to 273 on
  `mushroom_thin` while making the accuracy worse.

A constructor that sized itself, then verified the background and escalated on failure, would
be self-certifying in the same sense `boundary_quadrature` is.

## What would have to be true, and how to find out

The honest status: the wavelength rule is a post-hoc fit to eight domains plus two prospective
sweeps. Before it becomes a default it needs

1. a prospective test across the whole suite -- pick `d = h` capped as above, sight unseen, and
   compare against each domain's recorded best;
2. a size model, since that is the untested half of the signature;
3. a decision on the objective. Everything above optimises the eigenvalue's certified digits.
   The stated goal is a Hadamard-ready solver, and whether a basis tuned for `lambda` is also
   good for `dlambda` is unmeasured. That test (`docs/scope_and_downstream.md` section 4) should
   come first, because it may change what "better basis" means.
