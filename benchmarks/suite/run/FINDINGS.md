# How to get accuracy out of MPS — findings from the reference run

Companion to `NOTEBOOK.md` (chronological, with dead ends) and `RESULTS.md`
(the numbers). This is the distilled version: what governs precision, why the
hard domains are hard, and what to do about it.

## 1. The headline: it is one failure mode, not five

Poor accuracy, runaway memory, missed eigenvalues, and run-to-run irreproducibility
are **the same phenomenon at different severities**. The chain is:

    ill-conditioned collocation system
      -> sigma(lambda) curve is numerical noise in places
        -> spurious local minima
          -> bracket_mins refines them without bound
            -> lost digits / missed modes / exhausted memory

Evidence: the domains that exhausted memory (`rect_thin`, `iso_tri_h05`,
`reg_ngon_6/8`, `chevron`, `H_shape`) are largely the same domains with the
lowest accuracy and the highest seed-to-seed variance. Fixing the conditioning
should improve all four symptoms together; treating them separately does not
work, and most of this run was spent discovering that.

## 2. Sharpness is not the problem. Corner *coexistence* is.

The single most useful measurement of the run, from the `disk_sector` family
(exact spectra, one corner, everything else held fixed) against the
`parallelogram` family (two sharp corners, no reentrant corner):

                    p ~ 6.5     p ~ 13
    one corner       14.5        14.5      true digits, disk_sector
    two corners       7.1         4.0      certified,  parallelogram

**A single sharp corner costs nothing at any sharpness.** A 13.5-degree corner
reaches 14.5 correct digits in 17 seconds. Two sharp corners cost ~6 digits with
no reentrant corner present at all, and the cost then grows with sharpness.
This is an interaction, not a main effect — which is why "sharp corners are
hard" never explained chevron.

**Mechanism.** Corner-centred Fourier--Bessel functions decay like `r^(m p)`.
At `p ~ 6.5` they are numerically zero over most of the domain. With one such
corner the rest of the basis carries the solution everywhere else and the system
stays conditioned. With two, each block is negligible in the other's
neighbourhood *and* both are negligible in between, leaving a large near-null
space — exactly the `n_reg/n ~ 60-70%` truncation `TUNING_LOG.md` reported.

This also explains the elongation correlation without a separate cause: two
sharp corners are necessarily far apart in a slender domain, maximising the
region where both expansions vanish numerically.

**Ranking of corner difficulty**, from exact-truth measurements:
reentrant (`p<1`) costs ~1.5 digits versus sharp; the near-slit `p=0.504` still
reaches 13.1. Every *isolated* corner, of any type, is fine.

## 3. Bigger `n_basis` can make things worse — including fatal

Counter to instinct. `iso_tri_h16` fails on memory at `n_basis=240` and gives
**10.8 digits in 31 seconds at 120**. `reg_ngon_8` fails at 320 and gives 8.0 at
120. `chevron_1_125`, excluded from `produce.py` as non-convergent, returns 4.7
digits at 160.

More columns means worse conditioning, which by §1 means noisier sigma, deeper
refinement, more memory. Escalating `n_basis` to chase digits can lose the
domain entirely. **Always try smaller before larger.**

## 4. The pipeline was not reproducible, and that matters more than it sounds

Interior collocation points come from `domain.int_pts(method='random')` using
numpy's **global** RNG, unseeded. `iso_right_tri` returned 4.9, 4.0, 2.5 and 5.8
certified digits on runs of identical code. Observed spreads: `iso_right_tri`
3.3 digits, `GWW1` 2.2, `GWW2` 2.0 — all on the least accurate domains.

Now seeded (`--seed`, recorded in every result). Two consequences:

- **Seed spread is a free conditioning diagnostic.** It needs no reference
  values: a domain whose answer depends on which interior points were drawn has
  an under-determined system. Cheaper and more direct than any basis analysis.
- **A bad draw can make the solver miss a mode.** `iso_tri_h1` silently dropped
  `lambda=98.696`, producing a table wrong in every later entry, with a *valid*
  certificate on each surviving value. Re-run seeded it gives 14.4 true digits
  with all ten modes.

## 5. Certification is necessary but not sufficient

Moler--Payne certifies that *some* eigenvalue lies within the stated distance of
each value returned. It says nothing about **which** eigenvalue, and nothing
about modes you never found. Both failure modes occurred in this run and neither
was detectable from the certificate.

Calibration against every independently known case (analytic domains plus the
isospectral GWW pair): **the bound runs ~1 digit pessimistic**, never optimistic
by more than 0.6. So a domain certifying 7 is probably at 8.

**Completeness is the weak link**, and the recommendation that follows is the
most practical result of this run:

> Solve every domain at two or more seeds and require them to agree on the
> *set* of eigenvalues found. Not for accuracy — for completeness. Two agreeing
> seeds are stronger evidence than any single certificate, and cost only time.

The Weyl two-term count is a poor substitute: it missed `iso_tri_h1`'s dropped
mode (gap ~1), and it false-positives on small domains where the asymptotics
have not kicked in (`eq_tri`, gap 1.6, while agreeing with its closed form to
14.5 digits).

## 6. Some recorded "hard" verdicts were resource artifacts

`H_shape` was recorded at 8.2 digits in `TUNING_LOG.md`; with the memory bugs
fixed it reaches **9.9**. `reg_ngon_6` went from dead-at-25-seconds to 12.5.
`chevron_1_125` was excluded as non-convergent and converges at a smaller basis.

Earlier sessions were working with a pipeline that could exhaust 16GB on a
40-column problem, and several ceilings record what those runs could afford to
finish rather than what the method can do. Existing ceilings should be
re-measured, not trusted.

## 7. Genuinely hard, after all of the above

- **`stadium` / `stadium_L2` (2.9 / 3.2 digits).** Reproduces the tuning log
  exactly and independently. Zero corners, yet stuck: four C^1-but-not-C^2
  curvature junctions that are neither a corner the FB basis can target nor the
  smooth boundary the FS basis wants. The one clean, confirmed ceiling.
- **`chevron` family (3.3-5.6).** Two sharp corners plus a reentrant one; see §2.
- **`spiral` (1.6, and only 3 of 10 modes).** The only domain whose corners lack
  a straight sightline to infinity. `spiral_t25` cannot even build its branch
  cuts (`corner_branch_cut_polyline: no valid polyline cut found`).
- **`rect_thin`.** Unexplained memory runaway surviving every fix. Open.

## 8. CONFIRMED IMPROVEMENT: distribute the fundamental sources

The §2 mechanism makes a concrete prediction, and it holds.

`make_default_basis`'s multi-singular-corner branch builds its fundamental block
with `FundamentalBasis.by_corners`, which places sources on outward rays
**exponentially clustered at the corners**. So the entire basis — both blocks —
is corner-localized, and the region between two distant sharp corners is
represented only by expansion tails that are numerically zero there.

Replacing that block with `FundamentalBasis.by_boundary`, which distributes
sources along an offset boundary, on `parallelogram_p65` at n_basis=320:

    seed        by_corners (default)    by_boundary
      0               7.1                   9.0
      1               7.9                   9.0
      2               7.4                   8.7
    mean              7.5                   8.9
    spread            0.8                   0.3

**+1.4 digits on average, over the 8-digit bar on every seed, and the seed
spread drops from 0.8 to 0.3.** Accuracy and conditioning improve together,
which is precisely what §1 predicts should happen if the near-null space is the
cause. Note that simply raising `fs_frac` does nothing (7.6/7.1/7.4/7.0 across
0.3-0.85) — because it trades corner-localized functions for *other*
corner-localized functions. The placement is the lever, not the fraction.

**Scope: it is NOT a safe default.** Applied across the near-miss domains the
result is strongly domain-dependent, and the isospectral pair settles it:

    domain              default   distributed   delta
    GWW2                  7.7        9.4        +1.7
    parallelogram_p65     7.1        8.3-9.0    +1.2 to +1.9
    cut_square_r025       7.2        7.1         0.0
    mushroom_thin         7.4        6.5        -0.9
    GWW1                  6.3        2.1        -4.2

`GWW1` and `GWW2` are **isospectral** and geometrically near-identical — same
area, same perimeter, same corner angles, same absence of symmetry. They
responded in opposite directions by 1.7 and -4.2 digits. Whatever the offset
boundary is doing, it is sensitive to something incidental about where the
sources land relative to that particular boundary, not to a robust property of
the domain class.

So §8 is **a knob worth trying per domain, not a better default.** The
mechanism in §2 is still supported (it predicts *when* placement should matter,
and it does), but "distribute the sources" is not the general fix I claimed one
paragraph ago. A principled version would choose source positions from the
geometry — see the corner-separation experiment in "what to try next" — rather
than from a fixed offset.

Two things it did buy, which are real:

- **`GWW2` at 9.4 certified, agreeing with Driscoll's published table to 11.2
  digits.** The best value for that domain in this run by a wide margin.
- **`parallelogram_p65` over the 8-digit bar** on every seed tried.

**The earlier scoping claim was wrong.** I first wrote that the gain fails when
a reentrant corner is present, based on chevron. `GWW1`/`GWW2` are
reentrant-dominated and gave the largest gain *and* the largest loss, so that
explanation does not survive. The honest statement is that the effect is large
and unpredictable on domains with several spread-out singular corners.

**The original chevron observation still stands:**

    domain              default   by_boundary (0.5 / 0.7)
    parallelogram_p65     7.1        9.0
    chevron_1_15          5.6        5.4 / 5.9
    chevron_2_3           4.1        3.3 / 3.9

The distinguishing feature is the reentrant corner, which chevron has and the
parallelogram does not. When a reentrant corner is present it, not the
middle-of-domain gap, is the binding constraint — consistent with §2's finding
that reentrant corners are the more expensive kind. So this is a real fix for
the **two-sharp-corner, no-reentrant** case and not a general one.

**Proposed change** (not made — it alters existing behaviour): in
`make_default_basis`, the multi-singular-corner branch should distribute its
fundamental sources rather than cluster them at corners, or blend the two. The
current default is close to worst-case for any domain whose singular corners
are far apart relative to the wavelength.

## What to try next, in priority order

1. **Land §8 properly.** Decide between `by_boundary`, a blend, or a
   separation-aware rule, and test across the suite rather than one domain.
   Then re-measure the domains that sit just under the bar
   (`mushroom_thin` 7.4, `cut_square_r025` 7.2, `GWW2` 7.7).
2. **A seed-spread sweep across the suite**, as a conditioning map. Cheap, needs
   no reference values, and directly tests §1.
3. **Rellich orthonormalization and Cauchy singularity subtraction**
   (`lappy/rellich.py`, `lappy/cauchy.py`) — both exist, both are bypassed by
   `common.build_solver`, neither was reached this run.
4. **Size the LRU caches in bytes, not entries** (`lappy/cache.py`). Three
   separate memory failures traced to caches holding 128-256 matrices that scale
   as `n_basis x n_points`.
5. **Corner separation vs wavelength.** §2 predicts two sharp corners *close
   together* behave much better than two far apart at the same `p`. Untested.

---

## 9. Bucket-2 basis study, domain by domain

Method: screen with `basis_lab.probe` (tension at a *known* eigenvalue, no search --
seconds per configuration instead of minutes), three seeds, then confirm the winner
end-to-end through `bucket.py`. `bucket.py` grew `--fs-placement/--fs-d/--fs-frac` and
`common.build_solver` grew `basis=`, so a trial lands in `buckets.jsonl` like any other run.

### `parallelogram_p65`: 7.50 -> 11.6 certified. Bucket 2 -> 1.

Section 8 had this domain at 8.9 with `by_boundary` sources at `d=1.0`. **The offset is the
lever, and 1.0 is far from optimal.** Tension at the first, fifth and tenth eigenvalues,
median over three seeds, `n_basis=320`:

    configuration              cols  n_reg      lam1      lam5     lam10
    make_default_basis          322    134   1.4e-09   8.3e-10   2.1e-09
    fs_by_boundary d=1.0        320    128   8.9e-11   6.9e-11   3.5e-10
    fs_by_boundary d=0.3        320    217   1.9e-12   1.3e-12   1.8e-12
    fs_by_boundary d=2.0        320    117   3.9e-09   1.8e-09   4.4e-09

Three orders better than the default, and `n_reg/n` rises 42% -> 68%: accuracy and
conditioning improve together, which is what section 1 predicts if the near-null space is the
cause. The optimum is a plateau over `d = 0.2-0.4`; `d = 0.05-0.1` is worse again (sources too
close), so this is a genuine interior optimum and not "smaller is better".

`fs_frac` 0.5 and 0.75 are indistinguishable and 0.25 is worse, confirming section 8's finding
that the fraction is not the lever. `n_basis=240` already saturates, so the win costs *fewer*
basis functions than the default it beats.

End to end, `n_basis=320`, `d=0.3`, three seeds: **11.6, 11.6, 11.7** certified digits, seed
spread 0.1 against the default's 0.8.

### `parallelogram_p127`: 4.12 -> 7.5 certified. Still bucket 2, 0.5 digits short.

Same lever, same direction, and it stops short. Median tension over three seeds:

    configuration                  lam1      lam7     lam10
    make_default_basis          1.7e-06   7.9e-06   7.6e-06
    fs_by_boundary d=0.4        1.6e-09   3.1e-07   6.5e-09

`lam1` improves by three orders and `lam7` by only 1.4, so **one mode binds**, and it is
immovable: 3.0-3.3e-07 across `fs_frac` 0.5-0.9, `bdry_mult` 2-6, `d` 0.1-0.8, and
`n_basis` 320-640. The spectrum has no near-degeneracy there (relative gaps 1.5-6%), so it is
not a splitting problem. End to end at `d=0.4`: **7.5 certified digits**, up 3.4 from the
default but held exactly where the screen said it would be -- the binding mode sets the
domain's number, and it is half a digit under the bar.

Where the basis fails is localised and stark. Share of the boundary residual's L2 by segment:

    parallelogram_p127  lam7    2.8%  46.9%   1.7%  48.6%     <- the two LONG edges
    parallelogram_p65   lam7   27.1%  24.5%  20.5%  28.0%     <- spread evenly

96% of the residual sits on the two long edges, peaking mid-edge (0.27-0.37 of the edge length
from the nearest corner), not at the corners. `p65`, which now solves to 11.6 digits, spreads
its residual evenly. The distinguishing geometry is thinness: `p127` has area 1 across a
4.08-long edge, so a width of ~0.245.

**A hypothesis that failed, recorded so it is not retried.** If the width is what matters, the
source offset should scale with it, and `d/width ~ 0.5` should be the sweet spot. Measured, it
is not: `d = 0.03-0.15` (`d/width` 0.12-0.61) is uniformly *worse* at every mode, and `lam7`
is flat from `d = 0.1` upward. Whatever binds that mode is not the source stand-off distance.

### Reading so far

The two parallelograms differ only in sharpness (nu 6.5 vs 12.7) and aspect, and they respond
completely differently: one is fixed outright by source placement, the other has a single
mode that ignores every knob in the family. That is evidence the remaining bucket-2 failures
are not one phenomenon, and it argues for continuing domain by domain rather than hunting a
universal default -- which is also what section 8 concluded from the GWW1/GWW2 split.

### The chevrons: all four respond, and section 8 tested the wrong offsets

Section 8 concluded that distributing the sources "is a real fix for the two-sharp-corner,
no-reentrant case and not a general one", because chevron did not respond at `d = 0.5/0.7`.
That conclusion was an artefact of the offsets tried. Median tension over three seeds:

    chevron_1_15  nb=480      lam1      lam5     lam10     n_reg
    make_default_basis     2.6e-08   1.5e-08   5.1e-09    283/476
    fs_by_boundary d=0.2   6.1e-10   1.8e-11   7.7e-11    366/479
    fs_by_boundary d=0.8   1.6e-08   1.4e-08   4.0e-09    234/479   <- section 8's range

The useful range ends right about where section 8 started looking. Re-allocating the FB block
does nothing by comparison (`[8,178,8,45]` default vs `[80,120,80,45]` vs `[8,300,8,45]`: all
within a factor of 1.6), so the Fourier--Bessel side is not the binding constraint on these
domains -- which is worth knowing, because "sharp corners need more corner functions" is the
intuitive guess and it is wrong.

End to end, all four chevrons move from bucket 2 to bucket 1:

    domain          default   ->  distributed sources        nb     d
    chevron_1_15       6.31   ->  11.3                      480   0.2
    chevron_1_125      5.03   ->  10.7                      480   0.15
    chevron_2_3        3.65   ->  10.3                      480   0.4
    chevron_2_4        3.30   ->  (queued)                  480   0.4

`chevron_2_3` carries an extra lesson. Its recorded best was `n_basis=160`, because at 480 the
DEFAULT basis is degenerate -- preflight contrast 1.6, guard-aborted, recorded in BUCKETS.md as
"n_basis has a domain-specific optimum, and it is not monotone". With distributed sources at
480 the contrast is 2.9e+03 and the tension is six orders better than the default's
(1.8e-12 against 4.0e-06). The non-monotonicity was a property of the *basis*, not of the
domain: fix the conditioning and bigger is better again.

### The thin-neck group does not respond, and plane waves are inert

`mushroom_thin` and `mushroom_neck01` are unmoved by source placement at every offset tried
(`neck01`: 3.0e-07 at d=0.4 against the default's 3.2e-07). Their `n_reg/n` is 56%, so they are
conditioning-limited, but not by the mechanism the parallelograms and chevrons share.

Plane waves were tried as a genuinely different family (`benchmarks/basis_lab/planewave.py`:
`cos(k d.x)`, `sin(k d.x)`, verified to solve Helmholtz and to have correct gradients). Adding
80, 160 or 320 plane-wave columns to the default basis on `mushroom_neck01` or `stadium`
changes the tension by under 5% **and leaves `n_reg` exactly where it was** -- the regularizer
discards every one of them. Plane waves alone are far worse than either localized family
(`stadium` 9.2e-04 against 1.6e-04). They are not the missing ingredient for these domains.

### `stadium`: the default offset is catastrophically wrong, and it is still hard

`stadium` is corner-free, so `make_default_basis` gives it pure fundamental solutions at the
hard-coded `fs_d = 1.0` -- on a domain of half-width 0.5. The result is rank collapse:

    stadium nb=320                 cols   n_reg      lam1      lam5     lam10
    make_default_basis (d=1.0)      324      78   1.6e-04   4.9e-04   4.3e-04
    pure FS by_boundary d=0.1       320     320   6.8e-06   2.3e-05   1.8e-05

**76% of the default basis is discarded by the regularizer; at d=0.1 every column survives**
and the tension improves 20x. But it then plateaus: nb 320 -> 640 buys only 6.8e-06 -> 2.6e-06,
so the domain is not simply resolution-limited either. Stadium stays hard.

The contrast with `ellipse_a2` settles what the offset should scale with. That domain is
corner-free too, and its default `d = 1.0` is right: 1.1e-15, where `d = 0.05` gives 9.4e-05,
ten orders worse. The stadium is thin (width 1) and the ellipse is fat (width ~2.6), so the
offset wants to scale with the domain's thickness, not be a constant. **`fs_d = 1.0` as a
hard-coded default is the single most damaging line in basis construction found so far.**

### Scaling the offset by LOCAL thickness: right idea, wrong domains

The stadium/ellipse contrast says the offset should track thickness, and a mushroom's
thickness varies enormously -- measured with an inward-cone ray cast
(`benchmarks/basis_lab/placement.py`), `mushroom_neck01` runs from 0.003 at the neck to 1.497
in the cap, a factor of 500. No single `d` can serve that, which is a tidy explanation for why
no uniform offset moves the mushrooms.

The tidy explanation is wrong. Placing each source at `frac * local_thickness`:

    mushroom_thin nb=320                cols  n_reg      lam1      lam5     lam10
    make_default_basis                   322    181   1.2e-09   1.9e-11   5.8e-09
    FB + FS local-thickness frac=0.5     320    273   5.8e-09   7.1e-11   1.1e-08

**`n_reg` rises from 181 to 273 and the accuracy gets worse.** Same on `mushroom_neck01`
(3.2e-07 -> 1.7e-06). That is the opposite of the pattern on the parallelograms and chevrons,
where conditioning and accuracy moved together, and it is worth stating plainly:

**`n_reg/n` is a diagnostic, not an objective.** A basis can be better conditioned and less
accurate at the same time -- more columns surviving the regularizer does not mean more of the
solution is being represented. Optimising `n_reg` directly would have picked the worse basis
here.

A measurement error on the way, since it would be easy to repeat: the first version of
`local_thickness` excluded boundary within a fixed arclength of the query point instead of
using an inward cone. The nearest admissible point is then just outside that window on the
same wall, so every domain returns a thickness equal to the exclusion radius -- stadium (true
width 1) and ellipse_a2 (true width ~2.6) both came back at ~0.1-0.2. Check that a geometric
measurement varies with the geometry before drawing on it.

### Where bucket 2 stands after this pass

    solved (-> bucket 1)   parallelogram_p65, chevron_1_15, chevron_1_125,
                           chevron_2_3, chevron_2_4                      5 domains
    improved, still short  parallelogram_p127 (4.12 -> 7.5)
    unsolved               mushroom_thin, mushroom_neck01

The two mushrooms have now resisted a uniform offset at every distance tried, a
thickness-scaled offset, plane waves, and larger bases. They are the remaining open case, and
nothing measured so far explains what their basis is missing.

### The mushrooms were under-resolved, not mis-composed

I spent a long time on the mushrooms' basis *composition* -- uniform offsets, thickness-scaled
offsets, plane waves -- and the answer was basis *size*. With the DEFAULT basis, unchanged:

    mushroom_neck01     nb=320    nb=480    nb=640      mushroom_thin    nb=320    nb=480
    lam1               3.2e-07   5.1e-10   1.5e-10                      1.2e-09   6.3e-12
    lam10              1.3e-06   1.1e-09   9.2e-10                      5.8e-09   1.5e-11

Three orders for `neck01` and two to three for `thin`, saturating by 480-640. The recorded
bucket runs used `n_basis=320` because that is `entry.n_basis`, and escalation was never tried
on these two -- partly because BUCKETS.md's note that "n_basis has a domain-specific optimum,
and it is not monotone" (drawn from chevron_2_3, where the DEFAULT basis degenerates at 480)
made escalation look unpromising. It was exactly right here.

Lesson for the study, not just for these domains: **vary the cheap knob before redesigning the
basis.** Composition experiments at a fixed, too-small `n_basis` measure the wrong thing.

### Why the mushrooms look worse than they are: an evanescent stem

The mushroom is a half-disk cap of radius 1.5 with a stem 0.1 wide and 1 deep. For `lam1=6.5`
the stem's transverse cutoff is `lam ~ 987`, so the true eigenfunction there is evanescent, and
measured it is: `|u|` runs 6.3e-02 in the cap and 6.4e-06 -> 2.9e-09 -> 6.4e-10 down the
stem's centre line. Meanwhile the residual ON the stem walls is ~8e-09 -- the same size as the
true field beside it.

The residual's share by segment is 22% / 28% / 0.3% / 28% / 22% / **0.2%** -- almost none of it
on the cap arc, essentially all of it on the stem walls. So `sigma` is set by a region where
the eigenfunction contributes nothing, and 8e-09 against the cap's 6.3e-02 is 1.3e-07, which is
the measured tension.

That predicts the eigenvalue is better than the tension claims, and it is: located
independently at `n_basis` 240, 320 and 480, `lam1` agrees to **1.3e-08 relative** while the
tensions across those runs span 1.1e-06 to 5.0e-10. About 25x better than the worst tension
would suggest.

This is a general hazard for MPS on domains with thin appendages, and it also affects the
certified bound, which takes `sup|u|` over the whole boundary and so inherits the stem's
junk. A bound that weighted the boundary by the eigenfunction's own scale would report this
domain far more favourably -- but that is a different bound, not a tighter evaluation of this
one, so it is noted rather than done.

### Control: is it placement, or would more columns have done it?

Having found that the mushrooms wanted size rather than composition, the same question has to
be put to the domains where placement appeared to win. Default basis, escalated:

    parallelogram_p65        cols  n_reg      lam1      lam5     lam10
    default nb=320            322    134   1.4e-09   8.2e-10   2.1e-09
    default nb=800            802    293   1.8e-10   1.0e-10   2.2e-10
    nb=320, d=0.3             320    217   1.9e-12   1.3e-12   1.8e-12   <- winner

    chevron_1_15
    default nb=320            322    207   4.6e-08   2.0e-08   8.4e-09
    default nb=800            798    441   8.6e-09   8.6e-09   2.2e-09
    nb=480, d=0.2             479    366   6.0e-10   1.8e-11   7.7e-11   <- winner

Placement wins decisively and cheaply: two orders better than the default at 2.5x FEWER
columns on `p65`, and ~500x better on `chevron_1_15` at 40% fewer. Escalating the default
basis improves things slowly and steadily -- roughly a factor of 8 over a 2.5x column increase
-- which is the signature of a basis that is spending new columns on directions it already
half-covers.

So the two bucket-2 groups really are different, and each needed the knob the other did not:

    sharp-corner group (5 domains)   placement is the lever; size barely matters
    thin-appendage group (2 domains) size is the lever; placement actively hurts

Which is a warning about generalising from either. A single "better default basis" tuned on
one group would have made the other worse -- and the local-thickness rule, which was designed
from the stadium/ellipse evidence, did exactly that to the mushrooms.

### `parallelogram_p127` needed both knobs

Placement alone at `n_basis=320` gave 7.5, half a digit short, with one mode (`lam7`) pinned at
3.1e-07 against every knob in the family. Escalating the default basis alone does little on
this domain either. Both together clear it: `n_basis=640` with `d=0.4` certifies **9.7**, up
from 4.12.

The tension screen at `n_basis=320` was a poor predictor of that -- it said `lam7` could not be
moved, and the binding mode's tension is what usually sets the certified digits. I never
screened `lam7` at 640, so this is not a contradiction, but it is a caution about the method:
**a screen is a screen.** It is right about direction and cheap enough to run dozens of, but
the end-to-end number is the result, and on two domains now (this and the mushrooms) the
screen was materially more pessimistic than the certified answer.

### Seed variance: `chevron_2_3`'s conversion is not robust

The discipline this study opened with -- fix the seed *and* report the spread -- caught a real
problem. `chevron_2_3` at `n_basis=480`, `d=0.4`, three seeds:

    seed 0    10.3        seed 1    10.3        seed 2     5.1

A 5.2-digit spread, and one seed in three lands back in bucket 2. So the honest statement is
that this configuration reaches bucket 1 on most draws and is not reliable on all of them.
Compare `parallelogram_p65` under the same treatment: 11.6, 11.6, 11.7.

Interior collocation points come from the global RNG, so a "draw" here is which interior points
were sampled. A configuration whose accuracy swings five digits on that is under-determined in
the interior block, which is the same signal FINDINGS section 1 identified and section 8 saw as
a seed spread of 0.8 on the default `parallelogram_p65` basis (dropping to 0.3 when the sources
were distributed).

**Every single-seed conversion in this study is therefore provisional until re-run.** The
remaining ones are queued.

### The seed "spread" on `chevron_2_3` was a missing eigenvalue, and a spurious one

Chasing the 10.3 / 10.3 / 5.1 spread to its cause turned up something worse than variance.

First, the eigenvalues themselves are stable: all three seeds agree to 1e-9 relative
(64.70898869 / ...73 / ...67), and the tension at a fixed true eigenvalue barely moves across
five seeds at any interior-point count. The interior draw is not what varies.

What varies is *which minima the search returns*:

    seed 0/1:  ... 194.13, 214.34, 253.73, 263.03, 294.71
    seed 2:    ... 194.13, 203.44, 214.34, 253.73, 263.03

Seed 2 picked up a spurious minimum at 203.4449 -- `sigma` there is 1.9e-07 with the good
basis, against 1e-12 to 1e-14 at genuine eigenvalues -- and it passed the `ttol=1e-3` filter,
displacing a real eigenvalue from the list of ten and dragging the certified number to 5.06.

Worse, seeds 0 and 1 **missed a true eigenvalue at 226.6204**. It is genuine beyond doubt:
`sigma(226.6204) = 2.7e-12` with the placed n_basis=480 basis (2.2e-14 at the known-true
214.3403 for scale), a fine scan of [226.0, 227.3] bottoms out there, and the n_basis=160
default basis found it in every recorded run. So the "10.31 certified digits" was computed on
an incomplete list, with every entry after the gap mis-indexed -- exactly the failure
BUCKETS.md warns certification cannot see.

**`chevron_2_3` is therefore NOT a valid conversion** pending a corrected run.

An audit of every other conversion in this study against its baseline list found no other
missing eigenvalue (`parallelogram_p127` flags at a 1e-6 threshold on 177.6949 vs 177.6947,
which is the same eigenvalue).

### The lesson: a better basis makes the search harder

The eigenvalue that was missed is found by the *worse* basis and missed by the *better* one.
That is not a coincidence. A better-conditioned basis has deeper and narrower tension minima,
and the bracket scan runs on a fixed grid of `11 * n_eigs` points -- so improving the basis
can step the search straight over a well of the very kind it was improved to produce.

Any basis study that reports digits without checking completeness is therefore reporting a
number that can silently be about the wrong spectrum. `bucket.py` grew `--pts-per-eig` for
this, and the corrected `chevron_2_3` runs use a 3x denser grid.

### CORRECTION, and the real mechanism: a good basis lowers the tension floor EVERYWHERE

I wrote above that a better basis "makes the search harder" because its minima are "deeper and
narrower". The narrowness claim is **not supported** by measurement. Widths at 100x the local
floor, around the missed eigenvalue at 226.6204:

    default n_basis=160  (finds it)     well width 0.98    floor 2.0e-05
    placed  n_basis=480  (misses it)    well width 1.16    floor 3.4e-09
    search grid spacing                            2.92

The better basis's well is *wider*, and both are narrower than the grid spacing, which the two
bases share identically. Narrowness does not separate them.

What does, is the **background**. Over the 112-point search grid:

    basis              median sigma   frac below ttol=1e-3   discrete local minima
    default nb=160         2.9e-02             1%                     13
    placed  nb=480         3.2e-07           100%                     11

A better basis lowers `sigma` everywhere, not only at eigenvalues, and two things follow:

1. **The acceptance test stops discriminating.** `solve_interval` keeps a bracket whose
   minimiser has `sigma <= ttol` with `ttol = 1e-3` absolute. At a background of 3e-7 that is
   *every point in the window*. The spurious 203.4449 has `sigma = 1.9e-07` -- indistinguishable
   from background -- and sailed through.
2. **Detection reduces to finding local minima in a nearly flat landscape.** At the grid points
   bracketing 226.6204 the default basis dips (2.0e-2, 1.7e-2, **5.3e-3**, 6.5e-3, 1.8e-2 --
   a clear discrete minimum) while the placed basis rises monotonically (2.3e-7, 3.2e-7,
   3.8e-7, 4.2e-7, 4.2e-7 -- no minimum at all). The better basis registers FEWER minima on the
   same grid, 11 against 13, despite resolving more eigenvalues when asked directly.

So the failure is not "narrow wells" but **an absolute threshold and a local-minimum test
applied to a landscape whose scale the basis changed by five orders of magnitude.**

The actionable consequence is a pipeline change, not a basis change: `ttol` and the
detection criterion should be *relative to the observed background* -- the preflight already
measures `sigma_min` and `contrast`, so the ingredients exist. A genuine eigenvalue is a dip of
several orders below the local background; a fixed 1e-3 encodes an assumption about the basis
that a better basis violates.

Until that is fixed, **any certified digit count from a strong basis must be accompanied by a
completeness check**, because the search can silently return a list that is missing entries and
padded with background wiggles. See `benchmarks/suite/run/curves/tension_narrowing.png`.

### THE ACTUAL BUG: fundamental-solution sources landing INSIDE the domain

The question that cracked this open was whether a tension of ~3e-07 across the whole search
window is *possible in exact arithmetic*. It is not, and saying so out loud is the fastest
route to the answer: `sigma` that small everywhere would mean every `lambda` sits within ~1e-05
of an eigenvalue, when Weyl gives about ten eigenvalues across [18, 336].

Chasing that contradiction:

* Two independent bases agree there is **no** eigenvalue near 203.4449 (`sigma` is flat at
  1.94e-07 across +-0.006 with the placed basis, and flat at 4.2e-02 with the default one,
  which does dip to 8.2e-06 at the true 226.6204).
* Yet Moler--Payne, computed carefully there, gives `eps = 5.67e-06`, which *guarantees* an
  eigenvalue within 1.15e-03.

A theorem and a measurement cannot both be right, so a hypothesis was being violated. Moler
--Payne requires `u` to solve Helmholtz **exactly in Omega**. It does -- unless a
fundamental-solution source sits inside the domain.

    domain              d      sources inside
    chevron_2_3        0.40    24 of 240
    chevron_2_4        0.40    20 of 240
    chevron_1_15       0.20     0
    chevron_1_125      0.15     0
    parallelogram_p65  0.30     0
    parallelogram_p127 0.40     0

`chevron_2_3`'s reentrant corner has interior angle 305 degrees (nu = 0.59), leaving a
55-degree exterior wedge. A perpendicular offset from one arm lands inside the other for **any**
offset -- 24 sources at d=0.4, and still 8 at d=0.05. Normal-offset placement is structurally
unsafe on a strongly reentrant domain.

Such a column has a pole in `Omega`. It is not a particular solution there, the MPS premise
fails, and the certified bound computed from it means nothing. Dropping just those columns
restores everything at once:

    chevron_2_3, 216/240 sources kept
      background sigma   4.96e-02, 5.98e-02, 5.78e-02   (was ~3e-07 everywhere)
      sigma at true 214.3403                 1.97e-09
      contrast                               3.0e+07

So the flat background, the inert `ttol`, the spurious minimum at 203.4449 and the missed
eigenvalue at 226.6204 were all one bug, not four findings.

**`chevron_2_3` (10.3) and `chevron_2_4` (11.4) are withdrawn** pending re-runs on a legitimate
basis. The other four conversions are unaffected -- verified, zero sources inside.

The general lesson is worth more than the two domains: **an MPS basis must be validated, not
just constructed.** A source inside the domain is silent -- no exception, no warning, and the
tension does not blow up. It gets *better*, which is what makes it dangerous: the basis can fit
boundary data with an interior pole, so `sigma` falls everywhere and every downstream number,
including a "rigorous" certificate, is quietly void. `lappy` should refuse to build a
`FundamentalBasis` whose sources are not all exterior, or at minimum warn.

### Corrected: both chevrons convert on legitimate bases, at lower numbers

Re-run with interior sources filtered out (24 and 20 columns dropped), two seeds each:

    chevron_2_3   3.65 -> 8.26, 8.26    complete, and now finds the 226.6204 it had missed
    chevron_2_4   3.30 -> 9.50, 9.50    complete

Both convert, so the placement lever is real on these domains too -- but the honest numbers are
2.0 and 1.9 digits below what the invalid bases claimed (10.31 and 11.43). That gap is the
signature of the bug rather than an incidental loss: a column with a pole inside the domain
lets the fit absorb boundary error that a legitimate basis cannot, so the tension -- and every
number derived from it -- reads better than the truth.

**Final: 32 / 10 / 2 -> 39 / 3 / 2.** Seven of the ten bucket-2 domains converted. The three
that remain are `stadium`, `stadium_L2` and `mushroom_neck01`.

### What it takes for the tension curve to track Moler--Payne

The tension is only useful if it behaves like the quantity the certificate is built from.
Moler--Payne gives a *lower* bound away from the spectrum -- `eps >= dist(lam, spec)/lam` -- so
a globally small `sigma` is not a good basis, it is a broken measurement. Measured at
`lam = 240` on `chevron_2_3`, where the nearest eigenvalue is ~13 away and MP therefore forces
`eps >~ 5e-02`:

    bdry_mult   n_bdry    sigma      eps (true MP)   eps/sigma
       1.0        458    9.08e-04      3.65e+01        40000      <- collapses
       1.5        686    5.19e-02      1.54e+00           30
       2.0        912    5.98e-02      2.35e+00           39
       5.0       2278    9.47e-02      4.44e-01            5

**Two independent things have to hold, and each fails silently.**

1. **Every column must be a particular solution in `Omega`.** A `FundamentalBasis` source that
   landed inside makes Moler--Payne inapplicable, and the symptom is a background of ~3e-07
   instead of ~5e-02.
2. **The boundary must be oversampled.** At a collocation-to-column ratio of 1.0 the fit is
   free to vanish AT the points and be huge between them: `sigma` reads 9.1e-04 while the true
   `eps` is 36.5, a factor of 4e+04, on a perfectly legitimate basis. From ratio ~1.5 upward
   `sigma` tracks `eps` to a factor of 5-40, which is the expected constant gap between a
   discrete L2 over points and an L-infinity over the curve. `lappy`'s default `mult=2` sits
   just above the cliff -- adequate, but with only a 2x margin, and nothing currently checks it.

Both failures produce the same signature, and it is cheap to test for: the preflight scan
already computes `sigma_med`. A healthy background is ~1e-2 or above; `preflight.background_suspect`
flags anything below 1e-3.

    configuration                              ratio   sigma_med    verdict
    chevron_2_3, sources inside (the bug)       2.00    2.17e-06    FLAGGED
    chevron_2_3, filtered (legitimate)          2.00    3.71e-02    ok
    chevron_2_3, legitimate but ratio-1         1.01    9.16e-04    FLAGGED
    parallelogram_p65, known-good conversion    2.01    4.84e-02    ok

Both broken configurations are caught, both good ones pass, with about 1.5 orders of margin on
either side. `bucket.py` now prints a warning and records `background_suspect` in the result.

This is the check that would have caught the withdrawn `chevron_2_3` and `chevron_2_4` results
before they were ever reported, and it costs nothing -- the scan is already being run.
