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
