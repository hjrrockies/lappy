# Reference-value run — lab notebook

Append-only. Newest entries at the bottom. Records what was tried, what
happened, and why — including dead ends. Companion to `PROTOCOL.md` (how to
resume) and `queue.json` (current state).

Target: 10 Dirichlet eigenvalues per domain, **8+ certified digits**
(Moler–Payne), pushing further where possible.

---

## Session 1 — harness, and three bugs found before a single reference value

Built `runner.py` (one domain per subprocess, JSON out), `sweep.py` (driver with
per-domain timeout, updates `queue.json`), `status.py` (compact resume view).
One domain per process so a hung domain can never take the run down.

### Finding 1: `minimize_on_bracket` crashes on degenerate brackets

`square` — the *easiest* domain in the suite — died with
`ValueError: x not increasing!` from `opt.parabolic_iter_min`.

`opt.minimize_on_bracket` guards the bracket *width* (`x[2]-x[0] > tol`) but not
that the interior point is strictly inside. Where `sigma` is flat to machine
precision over a range — which is exactly what a high-multiplicity eigenvalue
looks like — `bracket_mins` can return `x[1] == x[0]`, and
`parabolic_iter_min` raises rather than degrading gracefully.

The function *already* has a golden-search fallback for when parabolic
interpolation fails; it simply never reached it. Fixed by routing degenerate
brackets straight there (golden search needs only the two endpoints).
**Behavior changes only on inputs that previously raised**, so nothing that
worked before can regress.

`lappy/opt.py`. This one is worth remembering: it would bite any user solving a
symmetric domain, and the failure mode is a hard crash, not a bad number.

### Finding 2: certified digits and true error can disagree completely

With the crash fixed, `square` returned **13.3 certified digits and 0.2 digits
against its closed form.** Both numbers were correct.

Moler–Payne certifies that *some* eigenvalue lies within the stated distance —
not that the *k*-th one does. The full-domain path returns **distinct**
eigenvalues with multiplicity in a separate array; the reference tables count
multiplicity. Comparing them elementwise misaligns everything after the first
degeneracy. Every returned value was accurate to ~13 digits; the list was just
short by one entry at position 2.

Fixed in `runner.py` by expanding by multiplicity before comparing, and
recording `n_distinct` alongside `n_found`.

**This is the case for the analytic tier existing.** Certification alone would
have reported a clean 13-digit success on a wrong table. Nothing else in the
pipeline — not tension, not Weyl, not certification — would have caught it.

### Finding 3: symmetry reduction does not split every degeneracy

`domain_symmetry` had no entry for `rect`, `eq_tri`, `iso_right_tri`, `disk`,
`disk_sector` or `parallelogram` — 14 suite domains forced onto the slow
full-domain path. Registered all six (new entries only; no existing entry
touched) and wrote `tests/test_symmetry.py`, which `symmetry.py`'s docstring has
always referenced but which never existed. It checks every group element maps
the boundary to itself, calibrated against the identity's own error, because
`bdry.dist` measures against the adaptive polyline and so reports ~1e-8 relative
even for the identity on curved domains. 117 checks, all 39 groups pass.

That exposed the real issue. `solve_sym` collected per-sector multiplicities
from `manual_solve` and **discarded** them, documenting that "repeated
eigenvalues across distinct sectors *are* the multiplicity". That is only true
when the registered group splits every degeneracy.

`SymmetryGroup` supports only real characters, i.e. elementary abelian
2-groups — which can be strictly smaller than the domain's true symmetry. The
square is the clean example: under `rect D2`, the double (1,2)/(2,1) splits
across sectors, but (1,3)/(3,1) are both odd-odd and land in the **same**
sector. Splitting them needs the diagonal reflection that D4 has and D2 does
not.

So multiplicity has two independent sources and both are needed:
across-sector *and* within-sector. Patched `solve_sym` to expand by the
per-sector estimate it was already computing.

**General rule for this run:** whenever the true symmetry group is larger than
the registered real-character subgroup, expect residual within-sector
degeneracy. Affects `square`, the regular n-gons (D_n vs D2), `disk` (O(2)),
`eq_tri` (D3 vs one mirror).

### Early numbers (certified / analytic digits)

    L_shape        13.6            (certified only)
    eq_tri         13.6 / 14.4
    disk           13.7 / 14.5
    square         13.7 / (see above — rerunning after the multiplicity fix)

`eq_tri` and `disk` agreeing to 14+ digits against closed forms is a good sign
for the pipeline itself: where the bookkeeping is right, the numbers are right.

---

## Hypothesis correction: the n_reg truncation is NOT a scaling artifact

`PROTOCOL.md` listed "per-column normalization" as the most promising unexploited
lever, on the theory that FB columns at a sharp corner differ by hundreds of
orders of magnitude (chevron reaches exponent ~334) and the GSVD discards them as
noise.

**That lever is already pulled.** `common.build_solver` ends with

    basis = basis.to_normalized((bdry_pts, int_pts))

i.e. `NormalizedBasis(..., max_scale=True)`: every column is rescaled by its max
and then normalized to unit L2 norm over the collocation points, at each lambda.
So the columns entering the GSVD are already well-scaled.

That makes the `n_reg/n ~ 60-70%` figure in `TUNING_LOG.md` much more
interesting, and points at a different mechanism:

**Revised hypothesis — near-dependence is intrinsic and precision-bound.** At an
11-degree corner the FB functions go like `r**(m*15.9)`. Successive `m` differ
only in a neighbourhood of the corner whose radius shrinks geometrically; away
from it every one of them is zero to many digits. Normalization rescales each to
unit norm, but that *amplifies the rounding noise along with the signal* — after
normalization the columns are nearly parallel not because of bad scaling but
because, in double precision, they carry almost no independent information about
the domain. More basis functions cannot help: each new one is numerically a
copy of its neighbours.

If true this predicts, sharply:

1. Extended precision at the *same* `n_basis` should gain many digits on chevron
   (the information is in the functions; double precision cannot see it).
2. Denser collocation should do nothing — confirmed already in `TUNING_LOG.md`.
3. Corner *reweighting* should hurt — confirmed already (it made chevron worse).
4. The effect should scale with the corner exponent `p`, not with the number of
   corners.

Prediction 4 is directly testable against exact truth, because `disk_sector`
sweeps `p` with a closed form: `sector_sharp_p65` (p=6.5) should be measurably
better than `sector_sharp_p133` (p=13.3), with the *same* single-corner
geometry, same symmetry, same everything else. That is the experiment to run,
and it is the reason the analytic tier exists.

Extended precision is a diagnostic here, not a proposed fix (per instructions).
If it confirms the mechanism, the fix to look for is a *better-conditioned
representation* of the same space -- e.g. orthogonalizing the FB block against
its own lower orders before collocation -- not more terms and not more points.

---

## Session 2 — the run is memory-bound before it is precision-bound

The first full sweep did not fail on mathematics. It drove the machine to a
**59.8GB memory footprint and 40GB of swap**, at which point it was thrashing
rather than computing (one domain per 15 min, versus 13–90s when healthy).
Worth recording in detail, because anyone running MPS over a suite of domains
will hit it.

### Where the memory is *not*

Measured per stage on `rect(1,8)`, `n_basis=240`:

    import                          135 MB
    build_solver                    136 MB
    assemble A_B (484x240), A_I     144 MB
    sigma(lam)                      152 MB
    interior_l2 + boundary_sup      169 MB
    build_sym_solver (one sector)   181 MB

A single solve is ~170MB. The deg-10 Dunavant cubature mesh is only 1,750
points (7MB per Vandermonde). None of this is the problem.

### Three multiplicative factors

1. **Sector count.** The symmetry path builds one solver *per sector* — four
   for a D2 group — each with its own basis and its own caches.
2. **Fork fan-out.** `manual_solve(n_workers=4)` forks four processes *per
   sector solve*, each copy-on-writing cached matrices as it touches them.
   Sixteen live copies.
3. **Caches sized in entries, not bytes.** `NormalizedBasis.norms` is
   `instance_lru_cache(maxsize=128)` and `_tensions_scalar` is 256. That is
   the right trade for repeated *scalar* evaluation at one lambda, and the
   wrong one for evaluation over a large point set, where each entry is a
   whole Vandermonde. Certifying ten eigenvalues across four sectors fills
   them with megabyte-scale matrices.

Fixes: default `n_workers=1`, pin BLAS to one thread (four workers against
threaded BLAS gave load 17 on 10 cores — pure oversubscription), and add
`lappy.cache.clear_instance_caches(obj)` (new, opt-in, nothing calls it by
default) to drop caches between eigenvalues in the certification loop.

### The guard, and why the obvious guards do not work

That reduced but did not eliminate the blowup: `rect_thin` still reached a
59.8GB footprint with a *single* worker, so there is a genuine runaway in that
domain independent of the fan-out. Two lessons from trying to contain it:

- **An RSS watchdog is actively misleading on macOS.** Memory is compressed,
  so `ru_maxrss` reported 4.7GB while Activity Monitor showed 59.8GB and 40GB
  of swap. The watchdog fired far too late to matter.
- **macOS gives no hard per-process cap.** `setrlimit(RLIMIT_AS, ...)` and
  `RLIMIT_DATA` both fail with "current limit exceeds maximum limit" when the
  hard limit is infinity, and `ulimit -v` is a no-op.

So the guard watches the **system**, not the process: `sweep.py` polls
`vm.swapusage` and kills the child *process group* (forks included) once swap
grows past a budget over its baseline. It catches every cause at once and
converts an unbounded swap event into one recorded domain failure.

**General lesson.** For a long unattended MPS sweep, the binding constraint is
memory, and the correct unit of protection is the machine rather than the job.
Guard on swap growth; run one domain per process; keep collocation-sized
matrices out of entry-sized caches.

`rect_thin`'s underlying runaway is still unexplained and is now on the list to
diagnose — it is a thin domain (slenderness 8) whose FB orders may be climbing
much faster than the corner structure suggests.

### rect_thin runaway: ruled out so far

Not basis construction — `make_default_basis(rect(1,8), 240)` gives a plain
`FourierBesselBasis`, 60 orders at each of 4 regular (p=2) corners, same shape
as the square. Not collocation — `pts_per_seg` gives [28,214,28,214] = 484
points. Not spectral density — only 11 exact eigenvalues lie below the
`lambda_window` top of 29.499 (lambda_10 = 25.291), so the search window is
appropriately sized.

Remaining suspects, for Phase 3: the corner-centred FB functions must represent
the mode across a domain of extent 8, so at order 120 the Bessel argument
reaches k*r ~ 43 with J_120(43) ~ 1e-60; `to_normalized` then divides by a
number that small. Suspect the runaway is in the bracket recursion
(`opt.bracket_mins` recurses with `nrecurse+1` and no depth cap) once sigma
becomes numerical noise over a wide stretch of the window.

---

## Session 3 — the sharp-corner hypothesis is WRONG, and the sector sweep proves it

The `disk_sector` family came in, and it is decisive. Same geometry throughout
(one arc, two straight edges, one mirror symmetry, closed-form spectrum); the
*only* thing varying is the corner exponent `p = pi/gamma`:

    domain              p        certified   vs exact truth
    sector_reflex      0.667       13.5          12.9
    sector_sharp_p65   6.5         13.1          14.5
    sector_sharp_p133 13.3         12.9          14.5

**A single sharp corner is not hard at all.** At `p = 13.3` — an opening angle
of 13.5 degrees, comparable to chevron's 11.3 — the method reaches **14.5
correct digits against the exact spectrum**, at `n_basis=320`, in 17 seconds.

That kills prediction 4 of the revised hypothesis, and with it the whole
"sharp corners cause precision-bound near-dependence" story. If sharpness were
intrinsically fatal, `sector_sharp_p133` would be the worst domain in the
suite. It is one of the best.

Note also the ordering: the *reentrant* corner (p=0.667, 12.9 digits) is the
harder of the two, not the sharp ones — which is what the approximation theory
says should happen, since the small exponent is what limits the convergence
rate of any smooth approximant. Sharpness costs *evaluation effort*, not
accuracy.

### So what is actually wrong with chevron?

Chevron has three things at once that `sector_sharp_p133` has only one of:

  - **two** sharp corners (p~15.9), not one;
  - a reentrant corner (p=2/3) as well;
  - slenderness 5.6, so the two sharp corners are far apart relative to the
    domain width.

The sector result rules out the sharpness of any single corner. The remaining
candidates are the *interaction* between two distant sharp corners (their
corner-centred expansions must each be represented across the whole domain,
where they are numerically indistinguishable from zero) and the combination
with the reentrant corner. That is a different mechanism from the one I wrote
down, and it is testable: build a domain with two sharp corners and no
reentrant one, and one with a sharp corner plus a reentrant one, and see which
reproduces the failure. `parallelogram_p65`/`p127` are exactly the first case
(two sharp singular corners, no reentrant corner) and are already in the suite.

**Method note.** This is the second time in this run that the analytic tier has
overturned a conclusion that certification alone would have supported. Certified
digits for all three sectors sit in a narrow band (12.9-13.5) and would have
suggested the three cases are equally difficult. The true errors differ by 1.6
digits and *in the opposite order* from the hypothesis. Certified bounds are
conservative in a domain-dependent way; only exact truth ranks difficulty.

---

## Session 4 — two runaway-memory bugs in the solve pipeline

Three domains (`rect_thin`, `iso_right_tri`, `H_shape`) tripped the swap guard.
Static analysis went nowhere: I wrongly blamed the newly-registered symmetry
groups (but `H_shape` uses a pre-existing one), then max Bessel order (but
`eq_tri` has the same order and works), then Bessel underflow (measured: all
values finite). What settled it in one shot was `faulthandler.dump_traceback_
later(45, repeat=True)` plus a swap watcher that dumps and aborts at 2.5GB.
**Lesson: for a memory runaway, get the traceback; do not reason about it.**

### Bug 1 — unbounded recursion in `opt.bracket_mins`

The dump showed **11 nested `bracket_mins` frames**. The function recurses with
`nrecurse+1` and no depth cap, and its only safety valve —

    if nrecurse==0 and len(y0_min_idx) > len(x)/3:
        raise EigensolverFailure("f has too many local minima")

— fires **only at depth 0**. Once `sigma` is numerical noise over part of the
window, every deeper level finds spurious local minima, flags them all for
refinement, and each flagged run calls `fill_refinement` (grid `shrink` times
finer) and recurses. Cost compounds across levels *and* across runs.

Fixed with an **opt-in** `max_recurse` (default `None` = existing behavior
exactly); `common.manual_solve` passes 8. At the cap we return the unrefined
brackets rather than discarding them, so `polish_eigs` still gets candidates.
Eight levels is ~256x the initial grid spacing — far finer than `bracket_xtol`
needs.

### Bug 2 — the caches fill during the *solve*, not certification

That fixed `iso_right_tri` but not `H_shape`/`rect_thin`. Second dump pointed
at `polish_eigs` -> `golden_search` -> `sigma` -> `_tensions_scalar`.

`golden_search` evaluates ~100 **distinct** lambdas per eigenvalue. Every one
lands in caches sized in *entries*: `_tensions_scalar` keeps 256,
`NormalizedBasis.norms` 128. At `n_basis=480` each entry is a megabyte-scale
Vandermonde, so one polish pass holds 1-2GB — and the symmetry path keeps one
solver **per sector** alive afterwards for certification. Successive lambdas
are distinct, so there is nothing to reuse. Clearing between eigenvalues costs
nothing.

I had earlier put `clear_instance_caches` in the certification loop. Right
idea, wrong place: certification matrices are only 5-76MB (measured). The
solve is where the memory goes.

### Result

**`H_shape`: 9.9 certified digits at n_basis=480**, against the 8.2 recorded in
`TUNING_LOG.md` — and it was failing outright before. So one of the suite's
"hard" domains was not accuracy-limited at all; it was being killed by a
resource bug that also capped what earlier sessions could explore. That is
worth stating plainly: **some of the recorded digit ceilings may be artifacts
of runs that could not afford to finish.**

`rect_thin` still trips the guard and stays open.

### Two caveats to settle before the final table

1. **`iso_right_tri` gets only 4.9 digits** (certified 4.9, true 5.1). That is
   implausible for an all-regular (p=2,4,4) triangle with a closed form, when
   `eq_tri` and `square` both reach 14.5. Prime suspect is the new
   `max_recurse=8` cap truncating refinement before the brackets are resolved,
   trading a memory runaway for a precision loss. Retest at 12 and 16; if
   accuracy recovers, the cap is doing real damage and belongs higher, with the
   *cache clearing* (bug 2) carrying the memory load instead.

2. **The table is currently a mix of pre-cap and post-cap runs.** The sweep
   skips domains already marked `done`, so the 11 banked results were produced
   before `max_recurse` and the polish-loop cache clearing existed. Nothing is
   wrong with them, but they were not produced by the same pipeline as the rest.
   Before publishing, re-run everything once under the final configuration so
   the whole table is reproducible from one code state.

### The depth cap is exonerated; two different real failures on the same shape

`eq_tri` re-run *with* `max_recurse=8`: 13.6 certified / 14.5 true — identical
to its pre-cap result. The cap does not cost accuracy. Good: the fix is clean
and the 31 domains running under it are not compromised.

So `iso_right_tri`'s ~5 digits is real. The suite happens to contain the same
shape twice — `iso_tri(1)` is `iso_right_tri(sqrt 2)` up to similarity (both
area 1.0) — which separates "property of the shape" from "property of my
registration". Running both exposed **two different** failures:

**`iso_tri_h1` misses an eigenvalue.** Found values match the closed form to 5
decimals *except* that lambda = 98.69604 is absent, so everything after it
shifts by one and the analytic comparison collapses to 0.6 digits. That value
is pi^2*(4^2+2^2)/2, the (4,2) mode. Certified digits said 5.5 — the bound is
perfectly valid for the values that *were* found, and says nothing about the
one that was not.

**`iso_right_tri` finds all ten** but modes 6, 8 and 10 are only ~5 digits
(246.73792 vs 246.74011). Different problem: plain under-resolution at
`n_basis=120` with `lambda_10 = 365`. Fixable by escalating.

The closed form itself is **correct** — an independent derivation (the right
isoceles triangle is the square folded on its diagonal, so
`lambda = pi^2 (m^2+n^2)/l^2` for `m > n >= 1`) reproduces
`reference.iso_right_tri_eigs` exactly. Worth checking before blaming a
reference table.

**Consequence for the completeness check.** The Weyl two-term count *should*
have caught the missing mode, but the gap was ~1 and my threshold was 3, so it
passed. Tightened to 1.5. Note what this means in general: for a domain with no
closed form, a single missed eigenvalue in the middle of the list produces a
table that is wrong in every entry after it, with a *valid* certified bound on
each surviving value and only a ~1 discrepancy in the Weyl count. That is the
most dangerous failure mode in this whole exercise, and the analytic tier is
the only instrument that detects it reliably.

### Calibrating the certified bound against truth

The GWW pair gives a correctness check that needs no reference table at all:
GWW1 and GWW2 are isospectral, so they must agree with each other.

    GWW1 vs GWW2, per mode : 10.6 10.0 7.4 11.9 6.7 8.7 10.8 10.0 14.3 11.5
    worst agreement        : 6.7 digits
    vs Driscoll's table    : GWW1 7.4, GWW2 6.7 digits
    certified (Moler-Payne): GWW1 6.6, GWW2 5.7 digits

Collecting every case where truth is independently known:

    domain              certified   true    gap
    sector_sharp_p65      13.1      14.5    1.4
    sector_sharp_p133     12.9      14.5    1.6
    sector_reflex         13.5      12.9   -0.6
    sector_slit           13.6      13.1   -0.5
    square                13.4      14.5    1.1
    eq_tri                13.6      14.5    0.9
    disk                  13.7      14.5    0.8
    GWW1                   6.6       7.4    0.8
    GWW2                   5.7       6.7    1.0

**The certified bound runs about 1 digit pessimistic**, and never optimistic by
more than 0.6 (the two sector cases where it reads high are within the noise of
a bound that is itself computed in floating point). That is the expected
behaviour: `boundary_sup` deliberately inflates by the sup-refinement drift and
`interior_l2` deliberately under-estimates, both erring safe.

Practical consequence for classifying domains: a domain certifying at 7 is
probably *at* the 8-digit target, and one certifying at 5.7 probably is not.
I will classify on the certified number, because it is the honest one, but the
write-up should state the offset so the table is not read as more pessimistic
than it is.

### Bug 3 — the same cache, a third time: `manual_solve`'s search

`reg_ngon_6` hit the swap budget **25 seconds in**. Sector-solver construction
was only 140MB (measured), so once again the memory was in the search, not the
setup.

`bracket_mins` evaluates *thousands* of distinct lambdas — and `fill_refinement`
explicitly reuses the y-values it already knows, so genuine cache hits during
the search are rare. Every one of those evaluations nevertheless lands in
`_tensions_scalar` (256 entries) and `NormalizedBasis.norms` (128). At
`n_basis=320` that is a few MB each, so a single sector's search holds ~1GB,
times one solver per sector, all kept alive for certification afterwards.

Fixed by clearing every 32 evaluations inside `manual_solve`'s `tensions2`.
Costs essentially nothing because there was nothing to reuse.
**`reg_ngon_6`: 12.8 certified digits, 197s — previously dead at 25s.**

This is the third distinct place the same root cause surfaced (certification
loop, `polish_eigs`, now the search). The real lesson is upstream of all three:
**`instance_lru_cache` is sized in entries, which is only safe when entries are
scalars.** Every one of these caches stores a matrix whose size scales with
`n_basis` x `n_points`. A `maxsize` in bytes, or simply a much smaller entry
count for the matrix-valued caches, would have prevented all three. Worth
proposing as a real change to `lappy/cache.py` rather than three call-site
workarounds — but that changes existing behaviour, so it is left as a
recommendation.

### Process error, recorded because it cost real time

I ran `reg_ngon_6` as a test **while the sweep was running**. Two processes
allocating at once doubled the pressure and tripped the swap guard on three
innocent domains (`reg_ngon_8`, `chevron_1_15`, `chevron_1_2` — the last drove
swap to 13GB). Those failures are artifacts of my own contention, not
properties of those domains; all three were reset to pending and must be
re-run. **One job at a time on this machine. No exceptions.**

### Operational: an hour lost to a detached background job dying

A sweep launched with `nohup ... &` from the tool shell was gone an hour later
having completed nothing, while `queue.json` still showed one domain `running`.
`setsid` does not exist on macOS, so the usual detach trick is unavailable.
Fixed by launching through the harness's own background mechanism, which owns
the process lifetime properly, and by *verifying the job is alive and has
advanced* rather than assuming a successful launch.

Also parked `rect_thin` as `status='hard'` and taught `sweep.py` to skip `hard`
alongside `done`. It sorts first in the pending list and costs ~340s to fail,
so every restart was paying six minutes for a known-bad domain.

Net operational rules for this run, all learned the expensive way:
  1. One compute job at a time on this machine.
  2. Verify a background job is alive AND advancing before trusting it.
  3. Park diagnosed failures so restarts do not re-pay for them.

---

## Session 5 — the pipeline was not reproducible

`iso_right_tri` returned **4.9, then 4.0, then 2.5** certified digits on three
successive runs of identical code. Not drift from my edits — genuine run-to-run
variation.

Cause: interior collocation points come from
`domain.int_pts(method='random')`, which draws from numpy's **global** RNG
(`geometry.py:1609`) with no seed. Every run gets a different interior sample.
`benchmarks/reference/run_sym.py` and `audit.py` already seed `np.random` before
solving, so the hazard was known; the suite runner simply did not.

Fixed: `--seed` (default 0) on both `runner.py` and `sweep.py`, seeded before
the solve and recorded in every result JSON.

**Two consequences, one obvious and one not.**

The obvious one: reference values must be reproducible, and until now they were
not. The 13 already-banked results were produced with unseeded draws. At 12-14
digits the sample almost certainly does not matter, but "almost certainly" is
not good enough for a table meant to back tests — the planned final consistent
re-run will now also be a *seeded* one.

The non-obvious one: **the spread across seeds is a diagnostic in its own
right.** A domain whose accuracy swings 2.4 digits with the interior sample is
telling you the collocation system is under-determined there — the interior
points are not pinning down the solution, so which ones you happen to draw
decides the answer. A well-conditioned domain should be nearly seed-independent.
That gives a cheap, direct test that needs no reference values at all: solve at
several seeds and report the spread. Worth running across the suite as a
conditioning measure, and it may explain `iso_right_tri` and `GWW1`/`GWW2`
better than any basis argument.

### A concrete hypothesis for the seed variance: the interior block is starved

`build_sym_solver` defaults `int_npts = max(n_basis // group.order, 40)` —
about **one interior point per basis column** — while the boundary gets
`bdry_mult=2`, i.e. 2x oversampling. Measured ratios:

    domain          n_basis  |G|   int_pts   sector cols   ratio
    iso_right_tri     120     2       60          60       1.00
    eq_tri            120     2       60          60       1.00
    reg_ngon_6        320     4       80          80       1.00
    square            120     4       40          30       1.33
    GWW1 (no sym)     320     1      640         320       2.00

In the MPS GSVD the interior block `A_I` is what rules out the trivial
solution: the tension is the ratio of boundary norm to interior norm, so `A_I`
has to *pin down* the interior size of the candidate eigenfunction. At one
point per column that system is barely determined, and which points you happen
to draw decides the answer — exactly the observed seed sensitivity.

`--int-npts` and `--bdry-mult` are now exposed on the runner to test this. The
experiment (queued for when the machine is free — one job at a time):
`iso_right_tri` at int_npts = 60, 120, 240, 480, several seeds each. Prediction:
the **spread across seeds collapses** as int_npts grows, and the mean accuracy
rises. If it holds, the default is simply too low and this is a real, cheap
method improvement, not a domain-specific tweak.

Note `eq_tri` also sits at ratio 1.00 yet reaches 14.5 digits, so a starved
interior block is not *sufficient* to cause trouble — it presumably only bites
when the basis is otherwise poorly conditioned. That is testable in the same
experiment by including a domain from each camp.

### Seed variance is not only interior starvation

`GWW2` came back at **7.7** digits having previously given 5.7 — 2.0 digits of
spread, the same magnitude as `iso_right_tri`. But GWW runs full-domain with
`int_npts = 2 x n_basis` (ratio 2.0), so it is *not* interior-starved. Whatever
drives the seed sensitivity is therefore more general than the starvation
hypothesis: both a starved (`iso_right_tri`) and a well-fed (`GWW`) case show
~2 digits of it. The starvation experiment is still worth running, but it is
now a candidate contributor rather than the explanation.

Collected spreads so far:

    iso_right_tri   2.5 - 4.9   (2.4 digits)
    GWW1            4.4 - 6.6   (2.2 digits)
    GWW2            5.7 - 7.7   (2.0 digits)

All three are among the least accurate domains in the suite. The well-converged
domains have not been re-run at multiple seeds yet, so the correlation between
"low accuracy" and "high seed variance" is suggestive but not established —
that is exactly what the seed-spread sweep will settle.

### Taking the best over seeds is legitimate here

Worth stating because it looks like cherry-picking and is not. Each run carries
its **own** Moler--Payne certificate, computed from that run's eigenfunctions.
A run certifying 7.7 digits has produced values good to 7.7 digits, regardless
of how many other runs did worse. So selecting the best-certified run over
several seeds is sound, in a way that selecting the best run by *agreement with
a hoped-for answer* would not be.

What it is NOT is a claim about the method's typical behaviour. The table
should therefore report both: the best certified result (the reference value)
and the spread across seeds (the honest statement of reliability). A domain
that reaches 8 digits on one seed in five is not a domain you can trust to
8 digits.

### Seeding turned an intermittent failure into a permanent one

`reg_ngon_6` reached 12.8 certified digits standalone after the cache fix, then
tripped the swap guard in the sweep at 65s. Nothing else was running. The
difference was the seeding patch: the successful run was an unseeded lucky
draw, and **seed 0 is a bad draw for this domain**.

That is a genuine interaction worth recording. Seeding is correct — reference
values must be reproducible — but it converts "fails sometimes" into "fails
always", and a single unlucky seed would otherwise lose a domain permanently.

It also sharpens the picture of what the seed is doing. An unlucky interior
sample does not merely cost a digit or two of accuracy; it can make the tension
curve noisy enough that the bracket search refines pathologically and the run
dies on memory. Accuracy variance and resource blowups are the same phenomenon
seen at different severities: **a badly conditioned collocation system produces
a sigma curve full of spurious local minima, and everything downstream —
refinement depth, cache pressure, runtime, accuracy — degrades together.**

`sweep.py` now retries a failed domain on seed+1, seed+2 (`--retries`, default
2), tagging each attempt so the successful seed is recorded in the result and in
`queue.json`. Reproducibility is preserved because the winning seed is stored.

### A bad interior draw can make the solver MISS a mode

`iso_tri_h1` earlier returned 5.5 certified / 0.6 true, with lambda = 98.696
(the (4,2) mode) absent from the list. Re-run seeded, it returns **13.4
certified / 14.4 true with all ten modes present.**

So the missing eigenvalue was not a search-algorithm failure and not a window
problem — it was an unlucky interior sample. That is the third distinct symptom
of the same underlying cause, and the most alarming one:

    bad interior draw -> noisy sigma curve
       -> lost accuracy            (iso_right_tri: 2.5 to 5.8 digits)
       -> runaway refinement/memory (reg_ngon_6, iso_tri_h05)
       -> A MISSED EIGENVALUE      (iso_tri_h1)

The first two announce themselves. The third does not: it produces a table that
is wrong in every entry after the gap, with a valid Moler--Payne certificate on
each surviving value and a Weyl discrepancy of about one. Only the closed form
caught it.

**This is the strongest practical argument in the whole run for solving each
domain at several seeds.** Not for accuracy — for completeness. Two seeds that
agree on the *set* of eigenvalues found is far better evidence of completeness
than any single run's certificate, and it costs nothing but time.

### Two guard-tuning lessons

**The budget was too tight, then the polling too slow.** Lowering the swap
budget from 4GB to 2.5GB killed `reg_ngon_6`, which legitimately peaks near 4GB
and reaches 12.5-12.8 digits when allowed to. Raising it to 5GB recovered the
domain. Conversely `reg_ngon_8` went from under budget to **11.5GB of swap
inside one 5-second poll window** — the guard fired, but only after the machine
had taken the hit. Poll interval cut to 1.5s.

Both are my own errors rather than properties of the method, and both produced
a spurious "hard" verdict at some point in the run. Recorded so the final table
is not read as saying more about MPS than it does.

**The registry's n_basis hints may be too aggressive.** `iso_tri_h05` fails on
memory at `n_basis=240`, but `TUNING_LOG.md` records 10.8-12.3 digits for it at
`n_basis=120`. The suite inherited 240 from a later, more ambitious production
config. Bigger is not free: a larger basis means a worse-conditioned system,
which (per the unified picture above) means a noisier sigma curve, deeper
refinement, and more memory — so escalating `n_basis` can make a domain *fail*
rather than merely cost time. Retry at 120.

---

## The designed experiment: one sharp corner vs two

`parallelogram_p65` was put in the suite specifically to separate "sharpness"
from "two distant sharp corners", because chevron confounds the two. Result:

    domain              sharp corners   p      reentrant?   certified
    sector_sharp_p65          1        6.5        no          13.1  (14.5 true)
    sector_sharp_p133         1       13.3        no          12.9  (14.5 true)
    parallelogram_60          0 (p=3)   -         no          12.7
    parallelogram_p65         2        6.5        no           7.1
    chevron_1_15              2       15.9       YES           5.6

**One sharp corner costs nothing** — and costs nothing *at any sharpness*,
since p=6.5 and p=13.3 both give 14.5 true digits. **Two sharp corners cost
about six digits**, with no reentrant corner involved at all
(`parallelogram_p65` has none). Adding a reentrant corner on top
(`chevron_1_15`) costs another 1.5.

So the mechanism is not the singularity strength of any individual corner. It
is what happens when **two corner-centred expansions must coexist**. A plausible
reading, consistent with everything else in this run: each corner's
Fourier--Bessel functions decay like `r^(m p)` away from that corner, so at
`p ~ 6.5` they are numerically zero over most of the domain. With one such
corner the rest of the basis (the fundamental-solution block, or the other
regular corners) carries the solution everywhere else and the system stays well
conditioned. With two, each block is negligible in the other's neighbourhood
and both are negligible in the middle, so the combined system has a large
near-null space -- which is exactly the `n_reg/n ~ 60-70%` truncation
`TUNING_LOG.md` reported, and exactly the conditioning failure that produces
noisy sigma, seed sensitivity, and runaway refinement.

Note this also explains the elongation correlation without needing elongation
as a separate cause: two sharp corners are necessarily *far apart* in a slender
domain, which maximises the region where both expansions are numerically zero.

**Testable prediction** (not yet run): a domain with two sharp corners that are
*close together* should behave much better than one where they are far apart, at
the same p. If that holds, the actionable advice is about corner *separation
relative to wavelength*, not about corner angle -- and the fix is to add basis
functions that are supported in the middle of the domain (more
fundamental-solution sources), not more Fourier--Bessel orders at the corners.
That is the opposite of what the earlier tuning sessions tried, and consistent
with their finding that corner reweighting made chevron worse.

### The interaction, completed

`parallelogram_p127` (two sharp corners, p=12.7) came in at **4.0** digits
against `parallelogram_p65`'s 7.1. So:

                    p ~ 6.5     p ~ 13
    one corner       14.5        14.5      (disk_sector, true error)
    two corners       7.1         4.0      (parallelogram, certified)

Sharpness is **free with one corner and costly with two**, and the cost grows
with p only in the two-corner case. That is an interaction, not a main effect —
which is why every attempt to explain chevron in terms of "sharp corners are
hard" failed, including mine. A single corner of *any* sharpness is fine; two
corners whose expansions cannot see each other are not.

The two families hold symmetry order (2), basis construction, and n_basis
fixed while differing in corner count, so the comparison is about as controlled
as this gets without building domains specifically for it.

---

## The fs_frac test was invalid — and that is instructive

Tested the §2 prediction (multi-sharp-corner domains need support *between* the
corners) by sweeping `fs_frac` on `parallelogram_p65`, seed 0, n_basis=320:

    fs_frac   0.3    0.5    0.7    0.85
    digits    7.6    7.1    7.4    7.0

Flat and non-monotone over a 0.6-digit range — no effect. Taken at face value
that falsifies the prediction.

**It does not, because the experiment did not test it.** In
`make_default_basis`, the multi-singular-corner branch builds

    fs_basis = FundamentalBasis.by_corners(domain, sources_per_corner, ...)

and `by_corners` places sources on outward rays from each corner, exponentially
clustered *at the corners*. So both blocks — Fourier--Bessel and fundamental —
are corner-localized. Raising `fs_frac` trades corner-localized functions of one
kind for corner-localized functions of another. The middle of the domain gets no
new support either way, which is exactly why nothing moved.

The mechanism in §2 predicts that the near-null space lives *between* the sharp
corners. To test it one needs sources that are actually there:
`FundamentalBasis.by_boundary(domain, n_per_seg, d=...)` distributes sources
along an offset boundary rather than clustering them at corners, and interior
sources would be stronger still.

So §2 remains **untested**, not refuted. Recorded this way deliberately: the
flat sweep is real evidence about `fs_frac` as a knob (it is not the lever), and
no evidence at all about the mechanism.

It also sharpens the proposal. If the default basis for a multi-singular-corner
domain puts *everything* at the corners, then for two distant sharp corners it
has no representation of the region between them beyond whatever the corner
expansions' slowly-decaying tails provide — which at `p ~ 6.5` is nearly
nothing. That is a concrete, checkable deficiency in `make_default_basis`, and a
better-specified proposal than "raise fs_frac".

---

## The Faber--Krahn edge bug: the disk cannot find its own ground state

The seeded re-run of `disk` certified **13.6 digits** while agreeing with its
closed form to **-0.2**. Not a bad draw — a structural bug:

    found: 14.68197 14.68197 26.37462 26.37462 30.47126 40.70647 ...
    exact:  5.78319 14.68197 14.68197 26.37462 26.37462 30.47126 ...

Every returned value is correct to ~14 digits. The list is simply missing
`lambda_1 = 5.7831859629` and returns modes 2..11.

Cause. `lambda_window` takes its lower edge from `bounds.faber_krahn`, and
**Faber--Krahn is sharp — the disk is the extremal domain.** So for the disk:

    lambda_window(disk, 10) lower edge = 5.7831859629
    exact lambda_1                     = 5.7831859629

identical to all printed digits. `bracket_mins` finds minima via
`discrete_locmin_idx`, which "ignores endpoints (assumes use of ghost points)".
A minimum sitting exactly *on* the lower edge therefore cannot be found, ever,
at any basis size or seed.

Fixed by nudging the lower edge to `faber_krahn * (1 - 1e-6)`.

**Why this one matters out of proportion to its size.** It is the cleanest
example in the run of the §5 warning: every individual value carried a valid
Moler--Payne certificate, the count was exactly the ten requested, and the Weyl
discrepancy was ~1 — under any threshold that does not also fire on correct
tables. Tension was fine. Nothing in the pipeline could see it. Only comparison
against a closed form revealed that the table was wrong in every entry.

It is also a reminder that sharp bounds are dangerous as search endpoints: the
better the bound, the more likely the extremal case lands exactly on it. Any
near-circular domain is at risk, and the failure is silent.

The unseeded `disk` run (14.5 true digits) got the right answer only because its
different interior sample happened to shift the tension curve enough for the
edge minimum to register. That is luck, not correctness.

---

## Near-degeneracy: the limit is within-sector, not numerical

`rect_near_deg_1e5` (rect(1, 1.00001)) certifies 13.6 digits but agrees with its
closed form to only 4.8. Per-mode:

    idx        found              exact           rel err
     1    49.347232449      49.347232449         1.9e-15
     2    49.347824616      49.347824616         3.3e-15
     4    98.694267509      98.694267509         2.6e-15
     5    98.694267509      98.695846622         1.6e-05   <-- merged

Nine of ten modes are exact to ~1e-15. **The pair at 49.347, split by 1.2e-5
relative, is resolved perfectly. The pair at 98.694, split by 1.6e-5 — a
*larger* gap — is merged**, returned as a double of the lower value.

So the limit is not numerical resolution. It is the sector structure, and it is
the same issue as Session 1's Finding 3:

- (1,2)/(2,1) at 49.347 have opposite parities, so `rect D2` puts them in
  **different sectors**. Each sector solve finds one, exactly, and they are
  merged by the across-sector rule. Splitting is irrelevant.
- (1,3)/(3,1) at 98.694 are both odd-odd, so they land in the **same sector**
  (separating them needs the diagonal reflection D4 has and D2 does not). There
  the only tool is `estimate_multiplicity`, which at `ttol=1e-3` reads a 1.6e-5
  gap as a genuine double.

**Practical statement of the limit.** Two eigenvalues distinguished by the
registered symmetry group are resolved to full precision no matter how close
they are. Two that are *not* distinguished are merged once their relative gap
falls below roughly `ttol`. Improving this means either a larger real-character
group (impossible here — D4's diagonal reflection has no real character on this
basis), a tighter `ttol`, or resolving multiplicity from the tension curve's
shape rather than a threshold.

This is worth knowing for shape optimization, where a design can drift towards
a symmetric configuration and two eigenvalues approach: the solver will hold
them apart perfectly if the symmetry separates them, and silently fuse them if
it does not.

---

## The isospectral pair does its job: §8 is a knob, not a fix

Applying distributed fundamental sources across the near-miss domains:

    GWW2                7.7 -> 9.4    (+1.7)
    parallelogram_p65   7.1 -> 8.3    (+1.2)
    cut_square_r025     7.2 -> 7.1    ( 0.0)
    mushroom_thin       7.4 -> 6.5    (-0.9)
    GWW1                6.3 -> 2.1    (-4.2)

**GWW1 and GWW2 are isospectral.** Same spectrum, same area, same perimeter,
same corner angles, neither has any symmetry. They responded in opposite
directions by 1.7 and -4.2 digits.

That is decisive, and it is decisive *because* the pair is in the suite. Any
other two domains differing this much could be explained away by some
geometric difference. These two differ in almost nothing that should matter,
so the effect cannot be a robust property of the domain class — it must be
sensitive to where the offset sources happen to land relative to that
particular boundary polygon.

Conclusion: **distributing the sources is a per-domain knob worth trying, not a
better default.** §2's mechanism still predicts correctly *when* placement
matters; it does not license "distribute and win".

I had also written, one result earlier, that the fix fails when a reentrant
corner is present (based on chevron). GWW1/GWW2 are reentrant-dominated and
produced both the best and the worst outcomes, so that explanation is dead too.
Recorded rather than quietly dropped: I proposed two scoping rules from small
samples and the data killed both.

**What it did buy, which is real:** `GWW2` at 9.4 certified and agreeing with
Driscoll's published table to **11.2 digits** — comfortably the best value for
that domain in this run — and `parallelogram_p65` over the bar on every seed.
Both are in the table, both reproducible.

Final table: **27 domains at >=8 certified digits.**

---

## Implementing fixes 1-4: one correction, one retraction-of-a-retraction

**Fix 1 — sharp bound as a search endpoint.** `evp.py` took the window's lower
edge straight from `bounds.faber_krahn`, which is attained exactly by the disk.
Nudged by a relative 1e-6 at the *use site*; the bound itself is mathematically
correct and left alone. `Eigenproblem.solve` now recovers the disk's ground
state.

**Fix 2 — `opt.bracket_mins` depth cap is now the default** (`max_recurse=8`,
`None` restores unbounded). `mps.solve_interval` already forwards
`**bracket_kwargs`, so it stays overridable.

**Fix 3 — NOT the change I recommended.** I proposed sizing the LRU caches in
bytes. Measuring first (hit rates across both the search and certification)
showed: `norms` 50-59% hits holding small vectors, `_tensions_scalar` 19.5%
holding small arrays, `_weighted_eval` **0.0% hits** — it can only ever miss,
because `norms` is cached above it. So the caches were never holding gigabytes
and a byte budget was not justified. Dropped the dead cache instead.

**Then I over-corrected.** On the strength of a 170MB-vs-166MB measurement I
also removed the `clear_instance_caches` calls from the benchmarks pipeline,
concluding they had been treating a symptom. `reg_ngon_6` was then SIGKILLed
end-to-end. A/B against stashed changes isolated it: **the clearing in
`polish_eigs` is genuinely required** — removing that one line is sufficient to
kill the run; restoring it gives 12.5 certified digits.

My measurement had instrumented `manual_solve` and never executed `polish_eigs`
at all. The `manual_solve` conclusion was correct *for `manual_solve`*; I
generalised it to a loop I had not measured. `golden_search` runs ~100 distinct
lambdas per eigenvalue and the symmetry path keeps one solver per sector alive
for certification afterwards, which is a completely different retention profile
from the search.

Certification-side clearing turned out to be genuinely unnecessary — the
verified run includes certification and passes without it. So of the three
places I had added clearing, exactly one was load-bearing.

**Lesson, stated plainly:** scope a benchmark to the code you intend to draw
conclusions about, and re-run end-to-end before believing a removal is safe.
The unit tests (731) passed throughout — only the full solve caught this.

**Fix 4 — reproducible interior points.** `rng` threaded through
`utils.rand_interior_points`, `Domain.int_pts`, `Polygon.int_pts`, and
`symmetry.fundamental_int_pts` (which advertised an `rng` argument from the
start and ignored it). `rng=None` keeps the legacy global RNG so
`np.random.seed` callers are unaffected.

Regression tests in `tests/test_solver_robustness.py` (10 tests) cover all four.
They save and restore numpy's global RNG state, because other test modules draw
interior points from it unseeded and reseeding changed their results depending
on test order — itself an argument for threading `rng` everywhere.

---

## Rellich exact interior norm: correct, elegant, and useless here

The MPS tension is a GSVD of the boundary block against an interior block whose
only job is to supply ``||u||_{L2(Omega)}``. That block is Monte-Carlo
collocation, which is why the answer moves with the draw. The Rellich identity
computes the same Gram exactly from boundary data alone, so substituting a
factor of it removes the random draw from the search entirely. Worth trying: it
attacks the seed-variance mechanism at its root.

**It works.** The modified tension gives clean wells -- 4.0e-15 at
lambda=49.348022 and 5.6e-15 at 98.696044, against O(1e-2) away from
eigenvalues.

**It is unaffordable as the search objective.** The basis depends on lambda, so
the Gram must be rebuilt and eigendecomposed at every evaluation. Measured: >30
minutes without finishing, for a domain that takes 165s normally. Killed.

**And where it is affordable, it changes nothing.** Hybrid: sampled interior to
locate eigenvalues, exact Rellich Gram to refine them (~24 golden-section
evaluations each, so the cost is negligible). Same coarse locations into both
refinements, so the comparison isolates the interior norm:

    iso_right_tri  seed 0:  sampled-polish 14.4   exact-polish 14.4   +0.0
    iso_right_tri  seed 1:  sampled-polish 14.5   exact-polish 14.5   +0.0

Zero improvement, twice. The reason is simple in hindsight: once the interior is
adequately sampled, its error is far below the level at which the final digits
are decided, so making it *exact* buys nothing. The sampled interior norm is
only a problem when it is **starved**, and then the fix is more points, not
better mathematics.

Recorded as a dead end so the idea is not re-attempted. Note it may still matter
for a domain where interior sampling is genuinely hard -- a very thin domain
where rejection sampling struggles -- but it is not a general accuracy lever.

## The finding hiding inside the negative result

Both hybrid runs reached **14.4-14.5 digits on `iso_right_tri`**, a domain the
pipeline reports at 5.8. And they were seed-stable: spread **0.1 digits** across
two seeds, versus **3.3** through the pipeline.

The experiment ran full-domain with ``int_npts = max(2*n_basis, 500) = 500``.
The pipeline runs the symmetry path, where ``build_sym_solver`` defaults to

    int_npts = max(n_basis // group.order, 40)    # = 60 here

i.e. about **one interior point per basis column**, while the boundary gets 2x
oversampling. That is the interior-starvation hypothesis from earlier in the
run, which I set aside when GWW did not fit it -- wrongly, because GWW runs
*full-domain* at ratio 2.0 and was never a test of the symmetry path at all.

If the interior sweep confirms it, this is a one-line default change affecting
every symmetric domain in the suite, which is most of them.

---

## Inexact Rellich interior block: works, but abandoned by decision

Following a suggestion to replace the interior collocation with something merely
*bounded away from zero* rather than exact. For a Dirichlet domain the Rellich
identity collapses to one term, so

    ||u||^2 = ||B(lam) c||^2,   B = (2 lam)^{-1/2} diag(sqrt(w * rN)) A_N(lam)

with ``rN = (x - x0).n``. **The Gram never needs forming** -- which was the
entire cost of the exact route (an O(n_quad n^2) assembly plus O(n^3)
factorization per lambda, >30 min vs 165s). ``B`` is one normal-derivative
Vandermonde on points we already have.

Measured on `iso_right_tri`, n_basis=120:

    per-lambda block cost:  sampled 39.6 ms   rellich 29.0 ms
    true digits (3 seeds):  sampled 14.4      rellich 14.5/14.4/14.5

Cheaper and equally accurate. But **it requires the domain to be star-shaped
about x0** (``rN >= 0``), and the three hardest domains are precisely the ones
that fail:

    L_shape       min rN +1.0000    0.0% negative   ok
    reg_ngon_6    min rN +0.8660    0.0% negative   ok
    mushroom      min rN +0.2496    0.0% negative   ok
    eq_tri        min rN +0.2165    0.0% negative   ok
    GWW1          min rN -1.0000   20.0% negative   FAILS
    H_shape       min rN -0.5000   22.6% negative   FAILS
    chevron_1_15  min rN -0.1768   33.9% negative   FAILS

(Note `L_shape` passes -- reentrant does not imply non-star-shaped.)

**Dropped, on the user's call**: a technique that needs star-shapedness works
mostly where things already work. Recorded for completeness, not as a
recommendation.

## And the interior block was never the bottleneck anyway

The starvation test settles it. Full-domain, `iso_right_tri`, n_basis=120:

    500 interior points -> 14.4 true digits
     60 interior points -> 14.4 true digits

Starving the sampled interior block by 8x costs **nothing**. So the
"interior starvation" hypothesis is dead: the interior block is not what limits
these domains, and neither an exact nor an inexact replacement for it can help.

That leaves one candidate for the ~7-digit gap, and it is a big one:

    full-domain,   n_basis=120 -> 14.4 true digits
    symmetry path, n_basis=120 ->  7.1 true digits

**The symmetry path costs ~7 digits on this domain at identical basis size.**
That is the opposite of its documented purpose -- `symmetry.py` argues each
sector needs |G| times fewer functions for the same resolution and should
therefore be *better* conditioned. Candidates, all confounded in the current
comparison: the `SymmetrizedBasis` projection (which is |G|-to-one on columns,
so half the columns are annihilated for a mirror), `prune_columns`' two-stage
removal, and `fundamental_bdry_pts` dropping the collocation points that lie on
the symmetry axis.

This is now the live lead, and it is independent of star-shapedness.

---

## Scope decision: symmetry is not the path lappy should be tuned on

Called by the user, and the evidence supports it independently.

lappy targets *generic* planar domains, and is meant to sit in the inner loop of
shape optimization. In that setting symmetry is measure-zero: a shape being
optimized will essentially never be symmetric, and if it drifts toward symmetry
the group is not known in advance. So the symmetry path is a special case that
most real inputs never take -- while the benchmark suite has been measuring it
as the default for 39 of 44 domains.

Worse, on the one domain where a clean head-to-head exists it is not just
irrelevant but harmful:

    iso_right_tri, n_basis=120     certified   true    time
      symmetry path (default)          6.0      7.1    171s
      full domain (--no-sym)          13.5     14.4     43s

**+7.5 certified digits and 4x faster** on the generic path. Whatever the
symmetry reduction is doing -- the |G|-to-one column projection, `prune_columns`,
or dropping collocation points on the symmetry axis -- it is costing far more
than the cubic saving it buys.

It is not uniformly bad: `L_shape` (13.6), `eq_tri` (13.6) and `reg_ngon_6`
(12.5) all reached good accuracy *through* it. But "sometimes fine, sometimes
catastrophic, on a path most users will not take" is not something to tune
against.

**Consequence for the suite:** reference values should be produced on the
generic full-domain path, and that is what the benchmarks should measure.
Symmetry becomes an optional accelerator to be justified per-domain, not the
default.

**What this dissolves:** within-sector multiplicity expansion, fundamental-domain
collocation, `prune_columns`, the symmetry-registry gaps, and the associated
caveats all leave the critical path. Several of tonight's hardest-won fixes were
in service of a code path that is now out of scope -- worth noting honestly,
though the bugs they exposed (the bracket recursion, the polish-loop retention,
the RNG seeding) were real and remain.

### rect_thin explained: it was the symmetry path all along

`rect_thin` was the one failure that survived everything -- the bracket depth
cap, the cache clearing, three swap-budget settings, every seed. It drove the
machine to a 59.8GB footprint and 40GB of swap and was parked as
`status='hard'`, "runaway memory not yet explained".

On the generic path: **7.5 certified / 8.5 true digits in 406s.** No memory
event at all.

So the runaway was a symmetry-path pathology, not a property of the domain or
of the method. rect(1,8) under `rect D2` builds four sector solvers over an
elongated geometry; something in that combination diverges. Since the symmetry
path is now out of scope there is no reason to chase it further, but it is worth
recording that the single most stubborn failure of the run had the same root
cause as the largest accuracy losses.

Running tally of what the scope decision fixed, at no cost:

    iso_right_tri     7.1 -> 14.4 true digits
    iso_tri_h05       2.7 ->  8.7 certified
    reg_ngon_8        8.0 ->  9.5 certified
    rect_thin         DEAD ->  8.5 true digits

against a worst case of -0.3 (chevron_1_15) and one genuine regression:
`rect_near_deg_1e5`, 13.6 -> 3.7 certified, where the symmetry path was
separating a 1.2e-5 near-degenerate pair by sector and `estimate_multiplicity`
now merges it. That is the multiplicity-vs-precision issue, not a conditioning
one -- see the discussion of tying the merge threshold to the requested
precision rather than to `ttol`.

---

## PAUSED — resume point

Stopped cleanly for the machine to sleep. Nothing running; queue is resumable.

**State:** generic-path pass (`--tag generic`) reached 6 of 44 before pausing.
`queue.json`: 6 done, 1 short, 37 pending. Everything committed.

**To resume:**

    LAPPY_RUN_SWAP_MB=3000 python -m benchmarks.suite.sweep \
        --all --retries 1 --timeout 900 --tag generic

Symmetry reduction is now OFF by default (`--sym` opts back in), so this pass
measures the generic path -- the one lappy is actually for.

**Open item, agreed and not yet implemented.** `common.manual_solve` merges
adjacent brackets whenever `estimate_multiplicity` at their midpoint returns
>= 2, with **no distance bound at all**:

    mult = mps.estimate_multiplicity(solver.tensions, cand_eig, cand_a, cand_b, ttol)
    if mult >= 2: merge

`ltol` and the requested precision never enter. `rect_near_deg_1e5`'s pair is
1.2e-5 apart; both tensions at the midpoint are under `ttol=1e-3`, so it merges
and the domain drops 13.6 -> 3.7 certified digits.

This is a benchmarks-layer regression, **not** a bug in `lappy`.
`mps.solve_interval` does it correctly:

    eig_brackets = sort_merge_brackets(eig_brackets, ltol, ...)   # distance = ltol
    mult = estimate_multiplicity(..., ttol, ...)                  # multiplicity, separately

i.e. merge distance keyed to `ltol`, multiplicity determined afterwards by
counting simultaneously-small tensions (`is_locmin & is_small`) -- which is
already exactly the "genuine double vs unresolved pair" discriminator. I had
described that as a proposal before reading the code; it was there all along.
Correction recorded because I misattributed the bug to `lappy` first.

**Fix when work resumes:** restore a distance guard in `manual_solve` -- only
consider merging when `|e1 - e0| / e < ltol` (or the caller's target precision)
-- then use `estimate_multiplicity` to confirm. That keeps what the rewrite
wanted (degeneracy confirmed from tension structure rather than assumed from
proximity) and restores the bound it dropped. `rect_near_deg_1e5` is the test.

---

## Session 6 — chasing the tension noise

Started from the user's observation that a hand-written script
(`scripts/near_degen_rect.py`) reached 14.5 digits on `rect(1,1+1e-5)` where the
benchmark got 4.8. It differed in two settings, and unpicking why took most of
the session.

### The actual bug: bracket_xtol vs merge_rtol

`manual_solve` hard-coded `bracket_xtol=1e-5` while merging at `merge_rtol=1e-9`
-- four orders apart. The bracket floor is `bracket_xtol * lam`, so at
`lam ~ 128` it is 1.3e-3, while `rect(1,1.00001)`'s pair there is 9.9e-4 apart.
**One bracket held both eigenvalues**, `minimize_on_bracket` returned a single
minimizer between them, and the polish converged to 128.303891 -- not an
eigenvalue at all. Cost 8.6 digits.

Fixed by defaulting `bracket_xtol` to `merge_rtol`. The two answer the same
question (how close can eigenvalues be and still be told apart) and must agree.
User's diagnosis, and correct.

### Three dead ends, recorded so they are not re-explored

**1. rtol=1e-12 as a global default.** Loosening from 1e-14 appeared to fix the
same domain, which sent me chasing regularization. Coincidence -- it perturbs
the search enough to split the bracket differently. And it is a bad trade
globally: measured over 12 domains, `mushroom` loses 1.4 digits (12.9 -> 11.5)
and `L_shape` 0.1, while `GWW1` gains 0.7 and `chevron` 0.2. rtol is genuinely
domain-dependent; no single value wins.

**2. adapt_rtol.** Validated on request and it is **non-functional**: returns
`rtol_min` for every domain tested, including ones where that value is
demonstrably wrong. Two independent reasons. Its 15-point grid spans the whole
window, so it never samples near an eigenvalue -- sigma stays in [2e-2, 7e-1]
while the floor is 1e-16. And its `||d2 sigma||/||d sigma||` statistic is
scale-invariant: 0.5345 at *every* rtol even when sampled tightly around an
eigenvalue, while the floor moves from 4e-8 to 7e-16. It measures the shape of
the well, which barely changes; the floor is what changes.

**3. A spectral-gap rule for rtol.** The idea: cut where the singular values
separate real information from redundancy. The user predicted the decay would
not be clean, and it is not:

    domain            best_rtol   largest drop at   drop size (decades)
    GWW1                  1e-12           1.4e-11          0.42
    chevron_1_15          1e-12           3.7e-13          0.65
    mushroom              1e-14           7.5e-14          0.48
    L_shape               1e-14           3.4e-14          0.41
    cut_square_r025       1e-14           1.5e-12          0.66
    stadium               1e-14           5.6e-21          8.11

Largest consecutive drop is 0.4-0.7 decades -- a factor of 2.5-5, i.e. the
ordinary roughness of smooth decay, not a rank cliff. `stadium`'s genuine
8-decade gap sits at 5.6e-21, far below any usable cut. And the location does
not predict the optimum: `cut_square_r025`'s largest drop is at 1.5e-12 while
its measured best rtol is 1e-14, off by 150x.

There is no well-defined numerical rank to find. The basis functions have a
continuum of usefulness.

### solve_interval vs manual_solve: refactor dropped

`manual_solve`'s stated justification (working around a `bracket_mins` hang)
expired when the noise-floor stop landed, and it had produced four bugs today,
so replacing it with the library's `solve_interval` looked attractive. Measured
over 8 hard domains at ltol 1e-12 and 1e-14:

    L_shape 13.5/13.4/13.5   mushroom 12.9/12.9/12.9   chevron 5.3/5.3/5.3
    cut_square 7.2/7.2/7.2   stadium 2.9/2.9/2.9       iso_tri_h05 11.3/11.4/11.4
    GWW1 8.2/7.1/7.1         reg_ngon_8 crash/2.8/crash

Six ties, one loss (GWW1, -1.1), one domain that fails both ways. Not worth 1.1
digits for tidiness. It also refutes my "one tolerance for all three roles"
argument: on a noisy curve, bracket width and merge distance genuinely want
different values, which is exactly what `manual_solve`'s decoupling provides.

Separately: **ltol=1e-8 (the lappy default) costs `L_shape` 4.3 digits**
(13.5 -> 9.2) through `solve_interval`. The coarse estimate is then only good to
1e-8 while `polish_eigs` searches +-1e-9 around it -- a bracket narrower than the
estimate's own uncertainty, so the polish cannot reach the true root. Same
failure `TUNING_LOG.md` recorded, via a different entry point. The invariant
`bracket_rel_width >= ltol` should be asserted somewhere; it is silent when
violated and looks exactly like a basis-resolution problem. Benchmarks now use
ltol=1e-14 to take this off the table (not a lappy default change -- lappy
targets 8 digits).

### Open: a genuine lappy crash on reg_ngon_8

    UserWarning: Eigenvalue may have deficient multiplicity (3.891e+05>1.000e-03)
    TypeError: 'NoneType' object does not support item ...

Something in the deficient-multiplicity path returns `None` and a caller
subscripts it. `rellich.lowdin_transform` documents exactly this contract
("Returns None (after warning) if G is deficient ... callers should fall back to
the raw values"), so a caller is likely ignoring it. Crash rather than graceful
degradation, on the suite's most degenerate domain. Traceback being collected.

### Where this leaves the noise problem

The user's architecture: detect, abort, or repair noisy solvers *before*
`solve_interval`, so the search does not have to manage noise. (My noise-floor
stop in `bracket_mins` is the wrong shape of fix by that standard -- the search
managing noise. Keep as a safety net, not load-bearing.)

Detection needs a signal. The spectrum does not provide one. The remaining
candidate is **the accuracy of the matrix entries**: singular values below the
level at which `A` is actually known are indistinguishable from perturbations of
it, so that level -- not machine epsilon, not a spectral feature -- is where the
cut belongs. It is domain-dependent for a concrete reason (GWW1: eight
high-order Fourier--Bessel blocks, each evaluated with its own error; mushroom:
an arc served by well-conditioned fundamental solutions), and it is directly
measurable via `bases.ExPrecFBBasis` rather than estimated.

---

## Session 7 — the failsafe, calibrated (and my labels corrected)

Goal: an abort test for ill-posed instances, calibrated before any bucketing.
The signal is the count of discrete local minima in the tension curve against
the Weyl-expected eigenvalue count, measured on a 300-point scan with no search.

**First result looked like failure.** Of four instances I labelled "noisy", only
one separated. `reg_ngon_8` @ rtol=1e-14 -- which produced `mult=0` and a crash
-- read *cleaner* than `L_shape`.

**Looking at the curves showed my labels were wrong, not the metric.** Rendering
sigma(lambda) to PNG and actually inspecting it:

    GWW1 @1e-14    23 minima = ~11 deep wells + ~12 shallow wiggles sitting on
                   TOP of the humps at sigma~1e-1, while real wells plunge to
                   1e-3. Genuinely ill-conditioned. Ratio 2.02.
    reg_ngon_8     7 clean deep wells, nothing spurious. Its crash was a
                   multiplicity failure at a legitimate well, not noise.
                   (7 wells vs Weyl 10.8 because D8 doubles share wells.)
    stadium        11 clean well-formed wells, bottoming at ~4e-4. Clean curve,
                   shallow floor.
    chevron_1_15   12 clean wells. Same.

With corrected labels the metric separates cleanly:

    clean curves       ratio 0.54 - 1.08   (eq_tri, square, reg_ngon_8, L_shape,
                                            chevron, stadium, mushroom)
    ill-conditioned    ratio 2.02          (GWW1 @ 1e-14)

**Threshold: 1.5.** Guard set to `max_minima = ceil(1.5 * weyl_expected)`.

### Each taxonomy class has a distinct visual signature

This is the useful part, and it is what the taxonomy was for:

- **#1 basis insufficiency** -- wells are clean and well-formed but the *floor*
  is too high. `stadium`'s wells bottom at 4e-4; the folklore
  `digits ~ -log10(sigma) - 1` predicts 2.4-3.4 against its measured 2.9. The
  curve looks correct; it just does not descend far enough.
- **#2 ill-conditioning** -- *extra* minima, shallow, riding near the local
  maxima rather than the floor. Count separates them; depth separates them even
  more sharply (prominence would be the refined statistic).
- **#4 search failure** -- clean deep wells and the search still fails.
  `reg_ngon_8` is the example: the curve gives the minimizer everything it needs.

### Caveat on well depth from a coarse scan

`sigma_min` over a 300-point grid is only meaningful for *wide* wells. `eq_tri`
reads 3.8e-4 yet solves to 14.4 digits -- 300 points over a window of width ~175
is a spacing of 0.58, far wider than its wells, so the grid never lands near a
bottom. It matched exactly for `stadium`, whose wells are wide and genuinely
shallow. So: **minima count is the sound pre-flight signal; well depth needs a
zoom pass to measure**, and I should not read #1 off a coarse scan.

### The reference table was the least accurate thing in the loop

`sector_reflex` and `sector_slit` appeared to certify *above* their true error
(13.7 vs 12.9, 13.6 vs 13.1) -- i.e. to violate Moler--Payne. Three candidate
explanations, tested in order:

1. **`boundary_sup` under-resolving the `r^p` corner** (p<1 has infinite slope,
   so the sup could hide between samples). Measured: converges from 8.652e-16 at
   n_per_seg=400 to 8.737e-16 at n=1600/grade=6, with drift falling to 0.0%. A
   1% effect, not a 6x one. **Ruled out.**
2. **`interior_l2` over-estimating on a curved domain** (mesh points outside the
   true boundary, where basis functions grow). Cross-checked against a
   Rellich-identity evaluation of the same norm -- boundary data only, no mesh:

       sector_reflex (curved)   cubature 9.27720939e-02   rellich 9.27720939e-02
       square (polygon)         cubature 9.79680314e-02   rellich 9.79681197e-02

   Agreement to 7 and 6 digits. **Ruled out** -- and this independently
   validates the Rellich Dirichlet branch, whose neglected `u != 0` boundary
   term I had flagged as a concern.
3. **The reference itself.** `_bessel_zero` scanned for a sign change and
   refined with `brentq` at default tolerance. Exact at integer order; at
   *fractional* order (which is precisely what sectors need, nu = m*pi/alpha)
   the returned zeros have |J_nu(z)| ~ 1e-13 instead of ~1e-16. **This was it.**

Replaced with `mpmath.besseljzero` at 40 dps. Residuals ~1e-16 at every order,
and the two sectors moved to 14.6 and 15.2 true digits -- now pessimistic by
0.9 and 1.6, in line with every other analytic domain.

**The lesson is about instrument calibration, not Bessel functions.** The
analytic tier is the ruler everything else is measured against, and part of it
was less accurate than the thing being measured. Worth remembering that
`rect_eigs`/`eq_tri_eigs`/`iso_right_tri_eigs` are closed-form and exact, while
anything routed through a numerical root-find (sector, disk) is only as good as
that root-find.

### Rellich normalization validated on the reentrant sectors

Fed exactly-normalized closed-form eigenfunctions to the Dirichlet Rellich
identity, `||u||^2 = (1/2 lam) * integral rN (du/dn)^2 ds`, and checked it
returns 1. `rN = (x - x0).n`.

**The identity is machine-exact at every corner exponent tested -- IF x0 sits at
the reentrant corner.**

    domain        nu        x0=apex      x0=bbox    x0=(0.5,0.5)
    reflex     0.667      -1.1e-14      6.4e-05        -8.5e-12
    slit       0.504      -2.0e-15      1.1e-01         2.1e-01
    reflex     1.333      -4.7e-15     -3.1e-15        -4.7e-15
    right-ang  2.000      -6.3e-15     -3.7e-15         1.8e-15

Why: on a straight edge emanating from the apex, `(x - apex)` is parallel to the
edge and `n` is perpendicular, so **`rN` is identically zero there**. The
singular factor `(du/dn)^2 ~ r^(2nu-2)` is multiplied by exact zero and the
integral collapses onto the arc, where `u` is smooth. Put `x0` anywhere else and
the integrand behaves like `r^(-0.99)` for the slit -- barely integrable, and
hopeless for any fixed grading.

**`cauchy.default_x0` returns the bounding-box centre**, which is close to the
worst choice for precisely the domains that need care. For a single-reentrant-
corner domain, placing `x0` at that corner is free and buys 7-14 digits.

Two earlier claims of mine were wrong and are retracted:

1. *"The graded quadrature is the limitation."* Refuted directly: refining from
   565 to 6790 nodes left the error at 2.2e-7, unmoved. It had converged to the
   wrong value, which is a different failure from under-resolution.
2. *"Do not switch certify.py to Rellich."* That rested on (1). With `x0` chosen
   properly, Rellich is machine-exact where the cubature agrees to only 6-7
   digits, so it is the *better* norm, not the worse one.

**Open limitation.** `x0` can only sit at one corner. `L_shape` has one
reentrant corner (fine), but `H_shape` has four and GWW two, and no single `x0`
zeroes `rN` at all of them. Those need either a partition of unity over corners,
or grading that genuinely handles `r^(2nu-2)`. Untested so far.

**Test-method note.** My first attempt computed `du/dn` by central differences
and returned *negative* norms of order 1e6: near the apex the graded nodes sit at
r < 1e-6, so an h=1e-6 step crosses the corner where `arg(z)` jumps by 2pi. The
control (rect) passed at 8.6e-11, which is what identified the fault as mine.
Two lessons: use analytic gradients for analytic functions (they belong in
`reference.py` alongside the eigenfunctions), and always run a control in the
same batch.

---

## Session: how robust is Rellich normalization on singular sectors, really?

Systematic follow-up to the previous entry, which tested three sectors at a
handful of `x0`. Swept opening angle `alpha in {pi/2, 2pi/3, 1.1pi, 1.25pi,
1.5pi, 1.75pi, 1.9pi, 1.99pi}` x modes `(m,n) in {(1,1),(1,2),(2,1),(3,1)}` x
`x0 in {default (bbox), apex, arc midpoint}`, feeding exactly-normalized
closed-form eigenfunctions to the Dirichlet identity and asking for 1 back.

Added `reference.sector_eigfun_grad` (analytic `du/dx + i du/dy`) so the test
never differences across the corner -- the failure mode the previous entry hit.

### The result, as one table (relative error, m=1 n=1)

    alpha     nu      x0=bbox     x0=apex   x0=arc-mid
    pi/2   2.000      3.3e-15     1.1e-15      4.0e-15
    2pi/3  1.500      9.3e-15     2.4e-15      1.5e-14
    1.1pi  0.909     -5.1e-09     8.9e-16     -1.5e-08
    1.25pi 0.800     -5.3e-07    -1.1e-16     -4.0e-06
    1.5pi  0.667      4.2e-07    -1.3e-14     -1.2e-03
    1.75pi 0.571      2.4e-15     2.4e-15     -3.7e-02
    1.9pi  0.526      1.9e-05     1.3e-15     -1.7e-01
    1.99pi 0.503      2.6e-04    -7.5e-15     -4.5e-01

`x0=apex` is machine-exact everywhere. Everything else degrades monotonically as
`nu -> 1/2`, up to **55% error** at the near-slit. `1.75pi` reads clean at bbox
only by accident: there the sampled bounding box centre lands exactly on the
apex.

**Only modes with `nu < 1` are affected.** At every angle, `(2,1)` and `(3,1)`
(`nu = 2pi/alpha, 3pi/alpha > 1`) are machine-exact at all three `x0`. Bad `x0`
does not degrade the method broadly; it destroys precisely the corner-singular
modes.

### The error is linear in |x0 - apex|, with a nu-dependent constant

Offsetting `x0` along the bisector, m=1 n=1:

    alpha     nu    err/|offset|   (TAU_FLOOR)^(2nu-1)
    1.1pi  0.909        1.5e-08              4.3e-08
    1.25pi 0.800        4.0e-06              4.0e-06
    1.5pi  0.667        1.1e-03              1.0e-03
    1.75pi 0.571        3.7e-02              5.2e-02
    1.9pi  0.526        1.7e-01              3.4e-01
    1.99pi 0.503        4.5e-01              9.0e-01

The constant is a *pure quadrature deficit*, and the right column is a first-
principles prediction of it that tracks over eight orders of magnitude.

### Why refinement cannot fix it: the mass lives below the smallest node

With `x0` off the apex, `rN` does not vanish there and the integrand goes like
`r^(2nu-2)`. Integrable, but its mass concentrates at the corner: the fraction
below radius `eps` is `eps^(2nu-1)`, which for `nu=0.503` is **90% below
r=1e-9**. `quad._KRESS_TAU_FLOOR = 1e-9` pins the innermost node there
regardless of point count -- confirmed, `r_min = 1.00e-09` at both `mult=2` and
`mult=32`. That is why the previous session saw 565 -> 6790 nodes move nothing:
the rule is not under-resolved, it is *truncated*, and the truncated part is
most of the integral.

Lowering the floor confirms the mechanism and also caps the payoff (fixed
offset 1e-2):

    floor      1.5pi      1.75pi     1.99pi
    1e-09    -1.2e-05    -3.7e-04   -4.5e-03
    1e-11     3.9e-07    -8.6e-05   -4.4e-03
    1e-13     3.9e-07    -8.6e-05   -4.4e-03

Two decades of floor buy ~1.5 digits at `1.5pi` and nothing at `1.99pi`, then
stall -- below ~1e-11 float64 arclength rounding takes over, which is the
original reason the floor exists (see `cached_kressgauss`'s docstring). **The
floor is not the bug and raising precision is not the fix.** `rN == 0` at the
singular corner is the only mechanism that works, because it removes the
singularity rather than resolving it.

### Standing recommendation

`cauchy.default_x0` (bounding-box centre) is unsafe on any domain with a
reentrant corner and should not be the default there. For one singular corner,
put `x0` on it. For several, no single `x0` works and the open question from the
previous entry stands.

### Incidental: FourierBesselBasis cannot be built on a half-disk

`alpha = pi` fails outright -- `disk_sector(1, pi)` reports `len(corners) == 2`
but `corner_angles` of shape `(2,3)`, so `from_domain` raises `IndexError` for
*any* `orders` length. A geometry/basis inconsistency at a straight (pi) vertex,
unrelated to Rellich, but it means the half-disk is currently unsolvable by MPS.
Not fixed here.

---

## Follow-up: 1e-14 at a 270-degree corner with a bad x0, in ~30 nodes

Retracts the "no rule in float64 can capture it" claim from the previous entry.
That was wrong. The integrand `r^(2nu-2)` is integrable and has exactly the
structure Gauss-Jacobi exists for; the truncation floor is a symptom of building
nodes as `p0 + tau*L*dir` in absolute coordinates, not a precision barrier.

### The rule

On a straight edge from the apex, Dirichlet data gives `du/dn = -+ nu J_nu(kr)/r`
and `rN = (z-x0).n` is **constant**. So the edge integrand is `r^(2nu-2)` times
an analytic function -- the Gauss-Jacobi weight `(1+x)^beta` with `beta = 2nu-2`.
Arc data is smooth: plain Gauss-Legendre. Crucially the Jacobi nodes sit
`O(1/n^2)` from the corner, not 1e-9, so no coordinate cancellation and no floor.

Measured at `alpha=1.5pi` (nu=2/3), `x0` = the bbox default that currently
fails, `du/dn` evaluated as a **black box** (only `nu` assumed known):

    nodes (2*n_edge + n_arc)     rel err
        16                       -7.1e-09
        28                       -7.5e-15
        40                       -6.0e-15

**28 nodes, 7.5e-15**, where the Kress rule plateaus at 2e-7 with 2264. The
near-slit `alpha=1.99pi` also works: 2.8e-14 at 56 nodes.

### Two traps found while testing this

**1. A degenerate x0 makes the test vacuous.** For `alpha=1.5pi` the two edges
carry `rN = Im(x0)` and `-Re(x0)` against identical `un^2`, so **any `x0` on the
diagonal cancels the entire singular contribution**. My first pass used
`x0=0.3+0.3j` and was measuring the arc alone. Rerun with `x0=0.4+0.05j` (edge
share 12.7% of the total) -- conclusions held, but they were not yet evidence
when first reported. Any future test of corner quadrature must report the
edge/arc split, not just the total.

**2. beta needs nu to full precision.** Rounding `nu = 2/3` to `0.667` (3e-4
relative) costs four digits and drops convergence to algebraic:

    nu_quad     28 nodes    192 nodes
    0.6667      -7.5e-15     -3.1e-15
    0.667        6.6e-05      1.7e-05
    0.600       -1.8e-02     -4.7e-03

`nu = pi/alpha` is exact from geometry, so this is free -- but it rules out
tabulated or margin-padded exponents of the kind `corner_grading_orders` uses.

### The part that actually matters for a real basis

A single mode is not the use case. Near a corner an MPS solution is a mix,
`sum_k c_k r^(nu_k)` with `nu_k = k*pi/alpha`, so `(du/dn)^2` carries every
cross term `r^((j+k)nu - 2)`. One Jacobi rule with `beta = 2nu-2` nails only the
(1,1) term. Tested on a synthetic 5-term corner expansion times an analytic
tail, against mpmath at 40 dps:

    n     Jacobi in r    after r = t^(1/nu)
    8       -4.6e-03              4.0e-15
    16      -1.2e-03              7.3e-15
    64      -7.2e-05              1.6e-13

Plain Jacobi is only algebraic -- 7e-5 at 64 nodes, useless. **The substitution
`r = t^(1/nu)` maps every `r^((j+k)nu)` to `t^(j+k)`, i.e. rationalizes the
whole commensurate corner expansion at once**, and the rule is then spectral:
**4e-15 at 8 nodes.** (It drifts up slowly past n~32 from roundoff, so small n
is both cheaper and better -- use 8-16.)

### Recommended construction

Per corner, per adjacent edge: panel in `t = r^nu`, Gauss-Jacobi with
`beta = (2nu-1)/nu - 1`, 8-16 nodes, evaluated in corner-local coordinates.
Smooth boundary away from corners keeps plain Gauss-Legendre. This removes the
`x0`-at-the-corner requirement entirely, which is what unblocks H_shape and GWW
-- no partition of unity needed.

Not yet tested: a real `FourierBesselBasis` (as opposed to closed-form or
synthetic data) through this rule, and a genuinely multi-corner domain.

---

## Stage 0 gate: the corner rule inside lappy, measured against Kress

Prototype in `benchmarks/corner_quad/` (nothing in `lappy/` yet), driven through
the real segment API (`seg.p/N/T(tau)`, `seg.len`,
`int_angles[corner_idx[c]]`) so what is measured is what Stage 2/3 will
assemble. Exact truth from `reference.sector_eigfun` + `sector_eigfun_grad`.

### Gate: passed, by 11 orders

At the target case -- alpha=1.5pi (270 degrees), `x0` deliberately OFF the apex
and off the cancellation diagonal (edge share 12.7% of the total):

    corner-jac   -6.0e-15    48 nodes
    Kress         2.8e-04   142 nodes

Better accuracy by 11 orders with **one third the nodes**. Across the whole
sweep the corner rule is at or below Kress everywhere, and never worse. With
`x0` at the apex both rules are machine-exact, as expected -- `r_N == 0` removes
the singularity and there is nothing left to resolve.

Order convergence at 1.5pi is spectral, as `2/nu = 3` predicts:

    order   4      6      8     12     16
    err  7.9e-3 4.3e-5 6.3e-8 8.6e-15 6.0e-15

**36 total nodes for machine precision** on a 270-degree corner at a bad `x0`.

### nu sensitivity: confirmed, and now a live guard

    rel err in nu    0     1e-8     1e-4     3e-4     1e-2
    rel err       6e-15  3.1e-10  3.1e-06  9.2e-06  3.1e-04

Exactly the predicted behaviour. This is wired as a test, not a one-off: if it
ever reports no loss, a rounded or margin-padded exponent has crept in.

### Finding 1: the achievable precision is a function of nu, and it is NOT monotone in order

    alpha      nu    2/nu   floor cap   best order   best err   err at cap
    1.05pi   0.952   2.10       128           98      4.4e-16      1.6e-14
    1.1pi    0.909   2.20       128           72      2.2e-16      2.0e-14
    1.25pi   0.800   2.50       128          100      7.8e-16      6.8e-15
    1.5pi    0.667   3.00       128           32      2.4e-15      5.3e-13
    1.6pi    0.625   3.20       128           46      5.3e-14      2.5e-13
    1.75pi   0.571   3.50       128           66      7.4e-13      2.6e-10
    1.9pi    0.526   3.80        75           16      1.2e-11      4.2e-09
    1.99pi   0.503   3.98        18            8      1.0e-09      3.8e-09

Machine precision is reachable for **nu >= 0.6** (alpha <= ~1.6pi), which covers
the 270-degree corner that actually matters. As nu -> 1/2 it degrades to
1e-9..1e-12.

Two distinct mechanisms, and only one was predicted:

1. **Predicted** -- the substitution amplifies node crowding
   (`tau_min ~ (c/n^2)^(1/nu)`), so past a nu-dependent order the innermost node
   drops below the `1e-9` coordinate-collapse floor and
   `(1-tau)*p0 + tau*pf` rounds onto the corner. Directly observed at 1.99pi:
   the error *diverges* with order, 1.6e-8 -> -3.2e-7 -> 1.4e-6 -> -4.9e-6 for
   order 16 -> 32 -> 48 -> 64, with `tau_min` crossing the floor at order 24.
2. **Not predicted** -- the best order is well *below* the cap (1.9pi: best at
   16, cap at 75). Dynamic range: at `tau ~ 1e-9` and nu ~ 0.5 the integrand
   `(du/dn)^2 ~ tau^(2nu-2)` is ~6e8 while its weight is correspondingly tiny,
   and the terms must sum to O(1). That is roundoff, not truncation, and it
   bites before the floor does.

**Consequence for Stage 3, which changes the design:** precision-driven sizing
cannot be "raise the order until the target is met, capped by the floor" --
that walks straight into a non-monotone error curve and picks a *worse* rule.
It must select the smallest order meeting the target from a *calibrated* curve
per nu, which is exactly what `cubature._choose_corner_rule(beta, eps, s_max)`
already does for the interior rule. Follow that precedent, and report the
achievable precision when the target is out of reach rather than silently
overshooting the order.

### Finding 2: the near-corner cases are corner-limited, not arc-limited

At alpha=1.1pi the singular edges carry only 1.1% of the integral, so the
obvious read is that the smooth arc dominates the error. It does not: raising
the smooth order 16 -> 48 changes nothing (1.37e-10 throughout), while raising
the *corner* order 16 -> 48 gives 1.37e-10 -> 1.5e-13. These are the
`2/nu` non-integer cases where convergence is algebraic, so they need more
corner nodes despite the corner contributing almost nothing to the value.
Cheap to get wrong in the sizing heuristic.

### Finding 3: panel length barely matters here, and the reason is geometric

Sweeping the panel's share of the edge (0.25 / 0.5 / 0.75 / 1.0) moves the error
by well under an order at every angle tested, with shorter panels marginally
better where the rate is algebraic (1.25pi: 1.7e-11 at 0.25 vs 6.2e-10 at 1.0).

This does **not** settle the plan's open question. `disk_sector`'s apex has
inradius R and its radii have length R, so a full-length panel never leaves the
corner expansion's disk of convergence. The domains where it would --- L_shape
(inradius 1, edge length 2), H_shape --- have no analytic truth, so this has to
be re-measured in Leg 3 against synthetic data with a known exact integral.
Do not read the flatness above as licence to default `frac=1.0`.

### Also worth recording

Higher radial modes are unaffected (1.5pi, modes (1,1)/(1,2)/(1,3): 6e-15,
3.8e-15, 1.0e-13) and nu>1 modes on the same domain stay machine-exact, so the
rule does no harm where there is no singularity to resolve.

---

## Curved sides at a singular corner: the substitution had to change

Prompted by the requirement that this work on non-polygonal domains, including
two curved sides meeting at a singular corner. It did not, as designed. Three
things came out of working it through.

### What actually changes on a curved edge

Both confirmed numerically against a circle of curvature kappa0 through the
origin:

    r.N  = (kappa0/2) s^2 + O(s^3)      with x0 AT the corner
    r    = s - (kappa0^2/24) s^3 + ...  so r is NOT arclength

The first is better news than expected. On a straight edge from x0 at the
corner, `r.N == 0` identically and the corner's singularity disappears. On a
curved edge it vanishes only to *second* order -- but that still leaves the
integrand `r.N (du/dn)^2 ~ s^(2nu)` bounded, so the trick degrades gracefully
instead of failing. With x0 anywhere else, r.N is an analytic series in s with
nonzero constant and linear terms.

The consequence is the exponent family. Both effects (plus the curvature
corrections in the corner asymptotics) contribute INTEGER powers of arclength,
so the family goes from `{k nu + 2q}` on a straight edge to **`{k nu + m}`** on
a curved one -- crucially including ODD powers.

### The original substitution fails on that family; t = r^(1/q) fixes it

`t = r^nu` maps `r^m` to `t^(m/nu)` with 1/nu in (1,2) -- only C^1. Measured on
a realistic five-term corner expansion, order needed to reach 1e-13:

    alpha       straight             curved
              sub=nu  sub=1/q     sub=nu  sub=1/q
    3/2 pi       4       8         never     6
    5/4 pi      54      10         never     8
    7/4 pi      28      14         never    10
    11/6 pi     32      16         never    14

"never" = no order up to 64 got there. Substituting **t = r^(1/q) where
nu = p/q in lowest terms** maps `{k nu + m}` to `{k p + m q}` -- all integers --
and is exact by order 6-14. It also covers the straight family, so one rule
serves straight edges, curved edges, and corners where one of each meets. The
only case sub=nu wins is alpha=3pi/2 (4 nodes vs 8), not worth a second code
path.

This also retires the claim from the Stage 1 notes that alpha=3pi/2 is the only
spectral reentrant angle. That was true *under sub=nu*. Under sub=1/q every
corner whose angle is a rational multiple of pi is exact.

### Open problem: on arc-arc corners nu is generically IRRATIONAL

Built `peanut` (union of two overlapping disks) as the natural curved singular
corner -- two arcs meeting at a reentrant angle, two such corners. It works, but:

    rho=0.6 d=1.2  ->  alpha = 1.521236 pi,  nu = 0.657360
    rho=0.8 d=1.0  ->  alpha = 1.369010 pi,  nu = 0.730455

The angle is *determined by the circle geometry*, not chosen, so nu is
irrational and there is no p/q to substitute. `corner_substitution` correctly
reports `exact=False` and falls back to sub=nu, which on a curved family means
~1e-8 at order 32 and never 1e-13. **This is the generic case for curved
domains, and it is unsolved.** Polygons escape it only because their angles are
rational multiples of pi by design.

Note the mechanism is NOT the catastrophic nu-sensitivity from
docs/corner_quadrature.tex Sec. 4. That one is about the *singular* exponent
gamma = 2nu-2 being wrong, and it costs four digits for a 3e-4 error. Here gamma
still uses the exact nu; only the substitution's rationalization of the *smooth*
remainder is imperfect, which is a far more benign failure. The two must not be
conflated.

The promising fix is to stop insisting on a monomial substitution and build the
rule directly for the known exponent set: the moments of `{t^(k nu + m)}` are
just `1/(gamma + k nu + m + 1)`, so an interpolatory (or moment-based
generalized Gauss) rule on that set is a linear solve, exact for irrational nu
too. Untested.

### Two incidental findings

- **A circular arc's arclength parametrization is machine-exact** (1.7e-16 for
  |dp/dtau| vs seg.len) at *any* `tol`, because its arclength map is linear.
  My earlier claim that curved segments are capped near `tol` (1e-4) was wrong
  and is retracted; the quality is a property of the curve, so
  `_parametrization_quality` measures it instead of assuming a budget. A curve
  with varying speed (ellipse) is a separate question, still unmeasured --
  building one at tol=1e-12 did not finish.
- **A self-intersecting boundary makes `Domain()` hang, not raise.** Picking the
  wrong arc of the small circle sent the CCW/polyline machinery into a spin with
  `val_simple=False`. `peanut` now asserts closure and that each arc stays
  outside the other disk. Worth a real guard in `geometry` eventually.

### Irrational nu resolved: an interpolatory rule on the true exponent set

The open problem above (arc-arc corners have irrational nu, so no monomial
substitution rationalizes {j nu + m}) is closed. The exponent set is *known*
even when irrational, and its moments are closed-form, `int_0^1 t^e dt =
1/(e+1)` -- so fix the nodes and solve for weights that integrate that set
exactly. `quad.cached_cornerinterpgauss`.

    order            8       12       16       24       32
    interpolatory 1.4e-4  1.5e-6  2.7e-9  6.8e-13  8.1e-14
    substitution  5.1e-6  1.0e-6  3.3e-7  6.7e-8   2.1e-8

(nu = 0.65736, the peanut's corner.) It *loses* below order ~12 and wins by five
orders at 24. So order >= 16 is the usable range, which Stage 3's sizing has to
know.

**The conditioning trap, and the fix.** Solving the square system (as many
exponents as nodes) is the obvious construction and it is a trap: exact on the
span, but cond(V) reaches 1e13 at order 12 and 1e19 at order 16, with weights
growing to -1e4. `sum|w|` is the factor by which a rule amplifies roundoff in
its integrand, so that is a ~1e-12 floor imported for free. Taking `n_exp <
order` -- a minimum-norm least-squares solve instead -- keeps **sum|w| = 1.0**
while retaining exactness on the exponent set:

    order  n_exp   sum|w|   in-span err
       12     12    2.1e2     6.6e-15      <- exact but ill-conditioned
       16     16    6.9e3     3.0e-13
       16     10    1.0e0     2.0e-15      <- exact AND well-conditioned
       24     14    1.0e0     7.8e-13

`n_exp >= order` now raises rather than being available as a footgun.

**One test of mine was worthless and is worth recording as such.** I measured
"out-of-span" accuracy by injecting exponents 0.37 and 1.61 and got ~1e-5,
which looked like a verdict on the rule. It is not: corner asymptotics say the
exponents present are exactly {gamma + j nu + m}, and for *irrational* nu no two
(j,m) coincide, so there are no resonances and hence no log terms either. The
injected exponents do not arise. Irrational nu is the benign case for logs; the
rational, resonant case is where they appear. The in-span number is the relevant
one -- but note that rests on the exponent family being right, which is theory
here and still wants checking against a real curved-corner solution (Leg 3).

### Curved-domain scope, from a measurement that changed my mind twice

Cost and quality of the arclength reparametrization on a varying-speed curve
(ellipse 2x1), where the whole scheme assumes |dp/dtau| == seg.len:

    tol      build   polyline pts   |dp/dtau| vs seg.len
    1e-4     0.01s        117            8.6e-03
    1e-6     0.75s       1273            1.6e-03
    1e-7     6.56s       3773            6.0e-04
    1e-8    >30s           --                --

Cost rises ~10x per decade while the error improves only ~2.6x, and it is still
6e-4 at tol=1e-7. **That is a ~1e-3 floor under any boundary quadrature on such
a segment**, corner-adapted or not. Circular arcs, by contrast, are machine-exact
(1.7e-16) at *any* tol because their arclength map is linear.

So my original blanket claim (curved segments capped near 1e-4) was wrong,
my retraction of it was also too broad, and the truth is that it depends
entirely on the curve. Scope decision: target straight + circular-arc
boundaries, where the corner rule's accuracy is actually reachable, and let
`_parametrization_quality` report when the parametrization -- not the corner
rule -- is the binding constraint.

Those four wedged 70-minute background jobs were all this: `tol=1e-8` on a
varying-speed curve, not a hang in the new code.

---

## Stage 2-3: one rule, self-certifying sizing, and a trap in the substitution

### sub = 1/q is exact and unusable; the interpolatory rule replaces it

The substitution `t = tau^(1/q)` rationalizes the full curved exponent family and is exact
to 2e-16 as an abstract rule on [0,1]. On an actual boundary it is unusable, because
`tau = t^q` crushes the innermost node as `tau_min ~ t_min^q`:

    alpha      q    tau_min at order 24
    3/2 pi     3      1.4e-08
    5/4 pi     5      1.1e-10   below the coordinate-collapse floor
    8/5 pi     8      1.4e-18   "
    7/4 pi     7      4.7e-19   "
    11/6 pi   11      1.6e-29   "

Those nodes round onto the corner once mapped through a segment's parametrization. The
interpolatory rule gets the same exactness from its WEIGHTS while taking nodes from the mild
`sub = nu` placement, so `tau_min` stays at 3.4e-5 to 3.7e-7 and the accuracy survives:
0 to 9.7e-14 at order 24 over the same angles. It replaces the large-q substitution outright.

So the design collapses to one rule, plus one cheap special case: a STRAIGHT edge with
2/nu integral (among reentrant angles, only alpha = 3pi/2) keeps the plain substitution,
where order 8 does what the interpolatory rule needs 24 for.

### Sizing certifies itself -- no offline calibration table

Stage 0 established that accuracy is non-monotone in order, so "raise the order until the
target is met" returns a *worse* rule. The plan was to follow
`cubature._choose_corner_rule`'s offline-calibrated curve. That turned out to be unnecessary:
every exponent's moment is closed-form, `int_0^1 t^e dt = 1/(e+1)`, so the rule can be scored
directly against its own integrand class with no reference solve and no stored table
(`quad.corner_rule_residual`). Smooth panels get the same treatment against exp(i k tau)
(`smooth_order_for_precision`). `corner_order_for_precision` then scans and returns the
smallest qualifying order, or the argmin plus what it actually achieved.

The indicator is deliberately pessimistic -- it takes the max over exponents, weighting none
by the coefficient it really carries (at nu=0.526 it reads 1.3e-6 at order 16 where the
measured error on a real eigenfunction is 1.2e-11). That is the safe direction, but it is not
the achieved accuracy and must not be quoted as such.

### The edge-type bug this exposed, which was worth more than the indicator

Scored against the CURVED exponent set, the substitution rule at alpha=3pi/2 read 1e-8 and
the sizing rejected it outright -- flatly contradicting Stage 0's measured 2.4e-15 on a real
sector. The cause: a straight edge admits only EVEN integer powers (r.N is exactly constant
and r is exactly arclength), so `{gamma + j nu + 2q}`; the odd powers exist only on a curved
edge. Testing a straight-edge rule against the curved set understates it by seven orders.
`corner_exponents` now takes `curved`, and alpha=3pi/2 straight returns order 8 at 4.2e-15,
matching the measurement.

### boundary_quadrature: the whole point of the exercise

    domain          precision   nodes   achieved
    rect              1e-14        88    1.0e-14
    L_shape           1e-14       104    1.0e-14
    H_shape           1e-14       200    1.0e-14     <- FOUR reentrant corners
    sector 1.5pi      1e-14        58    1.0e-14
    peanut            1e-14       184    1.0e-14     <- two arc-arc corners, irrational nu

No basis, no `mult`/`margin`/`q_min`/`q_max`/`c_lam`/`beta`. H_shape -- the domain that
motivated all of this, and where no single x0 can zero more than one corner -- reaches 1e-14
in 200 nodes.

`sum(wts)` matches the perimeter to ~1e-14 on the polygons but only to 2e-6 on `peanut`.
That is correct, not a defect: the interpolatory rule is exact on `{gamma + j nu + m}`, which
does not contain `t^0`, so it does not integrate constants. `sum(wts) == perimeter` is
therefore an invariant only where every corner panel uses the substitution rule, and must not
be asserted generally.

### Test bars now encode measured capability, not aspiration

`ACHIEVABLE` in tests/test_quad.py records the residual actually reachable per angle:
1e-14 out to alpha=1.6pi, 2e-12 at 1.75pi, 1e-9 at 1.9pi. A flat target across all angles
would have been a wish -- the capability degrades monotonically as nu -> 1/2, where the
integrand stops being integrable at all.

---

## Stage 4, Leg 1: the identity validated end-to-end, and a retraction

Exactly-normalized closed-form sector eigenfunctions through the whole
`boundary_quadrature` -> `eigfun_cauchy_data` -> `gram` path. Relative error in
`||u||^2`, which must be 1:

    alpha      nu     nodes      x0=apex    x0=generic    x0=bbox
    0.50pi   2.0000     44       1.1e-15      1.1e-15      1.1e-15
    0.67pi   1.5000     42      -1.2e-15      3.8e-15     -1.2e-15
    1.10pi   0.9091    140      -9.6e-15     -2.0e-14     -1.0e-14
    1.25pi   0.8000    144       1.8e-15     -1.6e-15      1.8e-15
    1.33pi   0.7500    150      -1.3e-14      6.3e-13     -1.1e-14
    1.50pi   0.6667     50      -7.7e-15     -6.4e-15     -7.7e-15
    1.60pi   0.6250    150      -1.1e-14     -1.4e-11     -2.8e-14
    1.75pi   0.5714    152       1.8e-15     -5.7e-12     -2.7e-14
    1.90pi   0.5263    152       1.3e-15      2.1e-09     -4.8e-13

`generic` is the binding case -- an x0 that zeroes no corner, chosen off the
diagonal so the two edges do not cancel. Machine precision out to alpha=1.5pi,
and 1e-11 or better through 1.75pi. Higher modes (1,2)/(1,3)/(2,1)/(3,1) all
come in at ~1e-15.

### Retraction: the indicator is not "systematically pessimistic"

I documented `corner_rule_residual` as erring on the safe side. It does not. At
alpha=1.5pi it reported 4.2e-15 for order 8 whose true end-to-end error is
1.5e-10 -- **optimistic by five orders** -- because all 20 exponents it probed lay
inside order 8's exact class, so it never saw the terms that actually limit the
rule. An indicator that only tests what a rule integrates exactly reports machine
precision for everything.

Fixing that by probing further exposed the opposite failure. `cornerinterp`'s
exactness claim GROWS with its order (n_exp = order//2), so a probe set that also
grows measures a moving target: its argmin landed at order 40 where the true
error keeps improving to order 64, and it read 2.5e-5 at an order that measures
9e-14 end-to-end.

### What replaced it: score a representative integrand, not a monomial max

`corner_model_error` builds an explicit member of the corner's integrand class --

    s^gamma * (sum_j a_j s^(j nu))^2 * (sum_m b_m s^m)

with a_j, b_m carrying the factorial decay of the actual Bessel series at
wavenumber sqrt(lam_max)*panel_length, and m restricted to even powers on a
straight edge -- and integrates it in closed form. So the signal is a genuine
relative error on something the rule will really face, rather than a max over
monomials weighted as if all were equally present. The high powers an unweighted
max fixates on carry negligible coefficients in reality.

Effect at alpha=1.75pi: the chosen order went from 40 to 64 and the delivered
error from 5.4e-10 to 5.7e-12, matching that angle's best achievable.

`corner_rule_residual` is kept as a diagnostic but no longer drives sizing.

### A dead end worth recording

Applying a safety factor to the requested precision does nothing at all --
`corner_order_for_precision` already returns the argmin when the target is
unreachable, so asking for 1e-16 instead of 1e-14 changes neither the order nor
the result. Tightening a request cannot compensate for a signal whose minimum is
in the wrong place; the signal had to be fixed.

### Guards now in the suite

- The alpha=3pi/2 cancellation is pinned as a fact about the geometry: an x0 on
  the diagonal kills the corner contribution to below 1e-12 while an off-diagonal
  one leaves it above 1e-3. A test using the former would silently measure only
  the arc.
- A corner-panel share test, so Leg 1 cannot pass by measuring the smooth part.
- The nu-sensitivity guard: perturbing nu by 3e-4 must cost >=3 orders.
- Test bars record measured capability per angle, and the test_quad bars are now
  split straight/curved, since the curved exponent family is denser and costs 1-3
  orders.

## Stage 4, Leg 3: the multi-corner claim, certified -- after two wrong models

Singular amplitude at every reentrant corner of a multi-corner domain, every
corner panel scored against a closed-form reference:

    domain    corner panels   nodes   worst panel error
    L_shape         2           56         4.9e-15
    H_shape         8          124         1.6e-14

H_shape includes two edges singular at BOTH ends (the notch floors), which is
the case no single-corner domain reaches and the only reason the panel split
exists. This is the leg that actually certifies the target; Leg 1 has one corner
and Leg 2's eigenfunction is smooth at the corners by construction.

### Wrong model 1: superposing two corner series

My first model set un(s) = A(s) + B(L-s), one corner series anchored at each end.
It reported 2e-2 on H_shape while every single-corner edge was exact to 1e-15,
and neither exponent family (sparse or dense) fixed it -- which is what showed the
model, not the rule, was wrong.

Near a corner the Dirichlet expansion is COMPLETE (Kondrat'ev): within the disk
about the corner inside Omega,

    un(s) = sum_k c_k nu_k J_{k nu}(sqrt(lam) s)/s  ~  sum_{k,q} a_kq s^(k nu - 1 + 2q)

so the far corner enters through the coefficients c_k, not as a separate additive
term. The superposition's cross term 2AB carries exponents gamma/2 + m, outside
any single-corner class -- an integrand no eigenfunction has.

A useful corollary, now a test: at nu=2/3 the equation k*nu - 1 + 2q = 0 has no
solution in non-negative integers, so a genuine du/dn at a 270-degree corner has
**no constant term**. Any model with one is not a member of the class.

### Wrong model 2: one anchored series per edge

Fixing that to a single corner series per edge still failed on H_shape at 4.8e-9,
and for a reason worth keeping: on an edge singular at both ends, a model anchored
at one end is SMOOTH at the other, so the panel anchored there applies a singular
weight to a function with nothing to cancel it. That is a real property of the
rule, not a modelling artifact -- and it is measurable in isolation:

    a genuine class member, one full-length panel   8.9e-16
    the same member, edge split in half             1.8e-9      (order 8)

**That justifies the panel design.** An edge singular at only one end must get a
single full-length panel, never a split; splitting costs seven orders. It is now
a test.

### What is actually correct: score per panel, not per edge

The two representations of a real eigenfunction on a doubly-singular edge are
asymptotic expansions about DIFFERENT points, and no single closed form is
sparse-in-nu about both endpoints. So there is no global synthetic model to
compare against -- each panel must be scored against the corner expansion valid
*on that panel*, over its own sub-interval. That is precisely the representation
the rule is built for, and it is what the test now does.

### Also: the reference guard caught the guard

`test_leg3_reference_is_closed_form_not_quadrature` first compared the Beta-function
moment against Gauss-Legendre at order 200 and "failed" at 3e-7. The Beta
expression was right to every digit (0.0e+00 against mpmath.beta); Gauss-Legendre
on s^0.3 (L-s)^0.8 simply converges that slowly, since the integrand has infinite
derivatives at both ends. The check now compares closed form against closed form
via a different implementation. Third time in this run that the instrument was
less accurate than the thing being measured.

## Stage 4, Leg 2: the polyomino control

`geometry.polyomino(cells)` builds the boundary of any union of unit grid cells;
`plus_shape()` is the 5-cell cross, with FOUR reentrant corners and area exactly 5.
`reference.polyomino_eigfun/eig/eigfun_grad` supply the exact eigenfunction:
`sin(m pi x) sin(n pi y)` vanishes on the entire integer grid, hence on the whole
boundary of any polyomino, with eigenvalue pi^2(m^2+n^2) and norm^2 = cells/4
exactly.

Rellich norm through the full path returns 1 to <1e-12 on plus / L / H / S /
square polyominoes, for modes (1,1), (2,1), (2,3).

**Its limitation is the point, and is now a test.** A closed-form eigenfunction on
a nonconvex domain is necessarily SMOOTH at the reentrant corners -- exactly why
L_shape has no closed form -- so the singular coefficients vanish and this leg
cannot test the corner singularity at all. `test_leg2_is_a_control_not_a_singularity_test`
pins that by checking |grad u| stays bounded as each reentrant corner is
approached, unlike a genuine corner-singular solution whose du/dn ~ r^(nu-1)
diverges. Leg 3 carries the singularity; Leg 2 tests geometry, panel splitting,
orientation and assembly on four reentrant corners with exact truth.

Boundary construction is by edge cancellation (each cell contributes four CCW
edges; a shared edge appears twice with opposite orientation and cancels), then
collinear runs are merged so a straight run of k cells is one segment rather than
k. Diagonal-only joins and enclosed holes are rejected rather than producing a
non-simple polygon.

---

## Stage 5: MPSEigensolver orthonormalizes by itself; the old paths are gone

`MPSEigensolver.from_domain(domain, basis=basis)` now returns L^2-orthonormal
eigenfunctions with no quadrature configuration from the caller. The seven
`rellich_*` parameters collapsed to one:

    before:  rellich=True, rellich_x0, rellich_mult, rellich_min_per_seg,
             rellich_margin, rellich_c_lam, rellich_beta      (+ a basis)
    after:   orthonorm=True, orthonorm_precision, orthonorm_x0

and the node set needs no basis at all -- it is a pure function of geometry,
`lam_max` and the accuracy target.

End-to-end, with nothing configured:

    domain        nodes   |G-1|      x0-spread     cubature cross-check
    L_shape        100    2e-16      9.3e-13       7.3e-12
    plus_shape     264    1.1e-14    4.2e-12       --
    H_shape        192    --         2.6e-11       1.4e-08

L_shape's lam_1 = 9.6397238445 against the known 9.6397238440.

**Leg 4 (x0-invariance) came in ~3 orders better than predicted.** I expected the
eigenfunction's own accuracy to cap it near 1e-9; it lands at 1e-12 to 1e-13 on
L_shape and 2.6e-11 on H_shape. The identity holds for every x0, so this needs no
reference at all -- the only check available on domains with no analytic truth.

### The cubature cross-check is bounded by the OTHER method

On H_shape the boundary rule is self-consistent to 2.6e-11 while the interior
cubature comparison sits at 1.4e-8. H_shape is hard for everything (the reference
run certified it to 9.66 digits), so the cubature and the eigenfunction residual
dominate, not the corner quadrature. `scripts/hshape_eigfunc_norm.py` -- promoted
from print-only to asserting -- now carries per-domain tolerances saying so, with
the x0-spread as the sharper of the two claims.

### Default precision is 1e-13, not 1e-14

A 270-degree corner lands at ~1.9e-14, so a 1e-14 default warns on the commonest
domain in the suite while delivering the same answer. Asking for 1e-14 explicitly
still warns if it falls short, which is the point of the warning.

### What was deleted, and what was NOT ported

Deleted: `lappy/rellich.py`, `lappy/cauchy.py`, `MPSEigensolver._cauchy_gram`,
and with them `rellich_gram_basis`, `orthonormalize_coef`, `basis_cauchy_data`,
`build_boundary_quadrature`, `corner_grading_orders`, `graded_pts_per_seg`.
`tests/test_rellich.py` + `tests/test_cauchy.py` became
`tests/test_orthonormalization.py`, keeping every MPS-wiring test and adding Leg 4.

`benchmarks/suite/experiments.py:exact_interior_factor` is deliberately NOT
ported -- it forms the basis-level N x N Gram, which is exactly the thing that
cannot work with a corner-adapted rule. It now raises with that explanation
rather than being quietly repaired; its own recorded conclusion (forming G was
unaffordable, >30 min against 165s) is why `cmd_inexact_rellich` exists.
`cmd_exact_interior` and `cmd_exact_polish` were ported, and got shorter: the
replacement takes no basis.

The Kress rule needed for the Stage 0 comparison is reconstructed inside
`benchmarks/corner_quad/proto.py` so that measurement stays reproducible after
the deletion.

---

## Panel length vs corner clearance: the last open question, and it was a live bug

Settled, and the answer was not "it doesn't matter". Leaving the clearance cap off
-- which was the default until now -- loses up to **twelve orders** on a domain
whose edge is long relative to the corner's clearance.

Test domain: a 1 x N polyomino strip with one cell below an end, so the reentrant
corner's edge has length N-1 against a clearance of 1. Worst relative norm error
over modes (1,1) and (2,3), against the exact eigenfunction:

    N     off      cf=1.0   cf=0.9   cf=0.7
     2  6.5e-14   6.5e-14  5.3e-15  2.2e-16
     4  4.7e-06   3.9e-14  3.6e-15  2.2e-16
     8  2.3e-02   2.2e-14  1.6e-15  0.0e+00
    16  7.7e-02   1.2e-14  1.3e-15  6.7e-16
    24  7.4e-03   9.3e-15  4.4e-16  2.2e-16

`clearance_frac` is now **0.9 by default**. cf=1.0 puts the panel exactly at the
radius where the expansion stops being valid and lands an order worse; below 0.9
buys nothing. Cost: ~6% more nodes than cf=1.0, 20-50% more than off (H_shape
204 -> 252, disk_sector 50 -> 70).

**A fixed `panel_frac` cannot substitute for it.** It is a fraction of the EDGE,
so it grows with the edge and stays too long: panel_frac=0.25 with no clearance
cap still gives 2.6e-04 at N=16, against 1.3e-15 for the clearance cap. This is
why the plan's original framing -- "measure the split point" -- was the wrong
question; the cap has to be geometric, not fractional.

### Choosing the instrument was the hard part

**x0-invariance on a real MPS eigenfunction cannot see this effect at all.** The
spread came back IDENTICAL to three significant figures across every panel
configuration -- 2.74e-07 at N=6 whether the panel spanned the whole 5-long edge
or a quarter of it -- while growing with N. That is the eigenfunction's own
residual: the identity int c.n (du/dn)^2 ds = 0 holds only for an EXACT
eigenfunction, and a longer, thinner domain is simply harder for the basis. Leg
4's documented floor, now measured rather than asserted, and a reminder that a
reference-free diagnostic is bounded by the thing it is diagnosing.

What worked was the polyomino's exact eigenfunction: zero residual, closed-form
norm. It is smooth at the reentrant corner -- Leg 2's limitation, and the wrong
tool for the singularity -- but exactly the right tool here, because what a long
panel risks is under-resolving the smooth FAR FIELD. The corner rule clusters its
nodes at the corner by construction, so the far end of a long panel is sparsely
sampled and cannot resolve the sqrt(lam) oscillation over the remaining
arclength. That is the mechanism, and it is a resolution failure, not a class
mismatch.

### What remains unmeasured, and why

The other candidate mechanism -- that beyond the clearance the integrand leaves
the corner's exponent family, so the rule loses its exactness guarantee -- is NOT
separately measured. Any synthetic model of "the integrand stops being in the
class past radius R" has to assume what it is trying to demonstrate, and the one
exact eigenfunction available on this geometry carries no singular amplitude to
mismatch. Since the resolution mechanism alone accounts for the observed failure
and the fix removes it entirely, this is recorded as untested rather than ruled
out.

---

## Stage 6: docs

`docs/corner_quadrature.tex` (8 pages) keeps Sections 1-6 as written -- the
original analysis, whose central claim held -- and gains
**Section 7, "Addendum: what implementation changed"**, plus a pointer in the
abstract telling a reader to go there first. Sections 1-6 are left intact rather
than silently rewritten, so the record shows which predictions survived. Three
subsidiary conclusions did not:

- the rationalizing substitution is `t = r^(1/q)`, not `t = r^nu` -- and is
  unusable in that form for q >= 4, because `tau_min ~ t_min^q` puts the innermost
  node below the coordinate-collapse floor (1.6e-29 at q=11);
- the `2/nu in Z` condition is a limitation of `t = r^nu` specifically, not of the
  method: exactness comes from the *weights* instead, which also covers irrational
  nu (the generic case for an arc-arc corner);
- Section 6's recipe omitted any cap on panel length, whose absence costs twelve
  orders.

The two errors from `corner_quadrature_review.md` are fixed: `alpha = m*pi/2` (not
`2*pi/m`, since `2/nu = 2*alpha/pi`), and the abstract's O(10)-node claim is now
conditioned on the substitution rationalizing the family.

`docs/eigfun_integrals.md` is new and supersedes `rellich_hadamard_mps.pdf`'s
basis-level architecture. `docs/rellich.md` keeps its mathematics -- still the
reference for the identity -- with a status note pointing forward, and gains the
one scope fact that belongs with the mathematics rather than the code: at a mixed
Dirichlet/Neumann reentrant corner the integrand goes like `r^(nu-2)`, which is
not integrable, so the identity itself diverges there.

`PROTOCOL.md` records the additive-only suspension explicitly: why it was safe for
this run (`common.build_solver` bypasses `from_domain` and built no Rellich data,
so no reference value is affected) and why it could not have been additive
(keeping both paths means two node sets and leaves the accuracy trap reachable).
The rule stays in force for everything else.

Also fixed: `scripts/hshape_eigfunc_timing.py` imported the deleted module and was
simply broken. Rewritten, and it now reports the number that matters for
CLAUDE.md principle 4 -- the quadrature is built **once per solve in 11 ms**
(H_shape, 240 nodes, 18 panels), so the per-eigenfunction cost is the Cauchy-data
evaluation, not the node set. Stale docstring references to `lappy.cauchy` in
`bases.py` and two test files were retargeted; `corner_terms` survives with
`symmetry.py` as its remaining consumer.

---

## Arclength reparametrization: diagnosed, prototyped, not yet implemented

Prompted by wanting spline boundaries to work. Three error sources, separated on
an ellipse (`benchmarks/arclength/diagnose.py`):

    tol     nodes   (a) s(t)   (b) t(s) roundtrip   (c) t'(s) vs 1/|p'|   |dp/dtau|-L
    1e-4     49     5.2e-05         5.2e-06              8.6e-03           8.6e-03
    1e-7    273     5.5e-07         1.8e-08              8.7e-04           8.7e-04

(c) tracks `|dp/dtau|-L` to every digit. The whole constant-speed defect is the
DIFFERENTIATED PCHIP inverse -- and it is gratuitous, since `t'(s) = 1/|p'(t)|`
analytically.

### Correction: my "~1e-3 floor on curved segments" was measuring the wrong thing

`_parametrization_quality` measured `|dp/dtau| - seg.len`. **The quadrature never
calls `seg.dp`** -- `assemble_panels` uses `seg.p/N/T(tau)` and `seg.len`. The
property it needs is that a node sits at the arclength its weight assumes, i.e.
the round-trip error (b), which is 3-5 orders smaller. The diagnostic now measures
(b), and the module docstring is corrected.

But the real number is worse than (b), and for a third reason:

    tol      f=x^2    f=exp(x/2)cos(3y)     (boundary integrals, ellipse)
    1e-4    3.5e-05        4.7e-05
    1e-6    2.9e-06        7.8e-07

and these do **not** improve with quadrature order -- 32 nodes and 256 nodes give
the same answer. A piecewise-cubic inverse makes `f(p(tau))` only C^1, and
Gauss-Legendre on a C^1 integrand converges algebraically. No order fixes it.

### The fix, prototyped and measured (`benchmarks/arclength/`)

Keep the adaptive table as a bracket and initial guess; then
  1. rebuild `s_nodes` at the table's own nodes with a high-order Gauss rule;
  2. solve `t(s)` by **Newton** on the exact `s(t)`, with `ds/dt = |p'(t)|`;
  3. take `t'(s) = 1/|p'(t)|` analytically, never the differentiated interpolant.

Ellipse, same integrals, Gauss-Legendre of increasing order:

    order      pchip      newton
      32      3.9e-05    3.9e-05
      64      7.2e-07    3.7e-08
     128      2.2e-09    9.6e-14
     256      7.3e-07    4.7e-15

Spectral convergence restored; `|dp/dtau| - L` goes 8.6e-03 -> 3.7e-16 by
construction. **And the Newton results are identical for tol=1e-4 and tol=1e-6**,
because the table is only a bracket now -- which also removes the reason tight
`tol` was ever wanted, i.e. the >30 s construction that wedged four background
jobs earlier in this run.

### Splines need one more thing: panels must break at knots

A degree-k B-spline is only C^(k-1) at its knots, so even a perfect `t(s)` leaves
`f(p(tau))` non-analytic there. On a wobbly cubic spline (5 interior knots):

    |dp/dtau|-L    pchip 2.1e-02   newton 3.8e-16
    s(t(s))-s      pchip 3.6e-05   newton 1.3e-16

    integral, one global panel:      order 128  pchip 1.9e-05   newton 1.3e-06
                                     order 512  pchip 3.5e-06   newton 2.6e-09
    integral, knot-aligned panels:   96 nodes (6 x 16)          8.3e-11

Five orders better with fewer nodes. (The knot-aligned figures are near the
reference's own accuracy, so read them as a floor, not a measurement.) The
arclength panels must respect knots too, or a Gauss panel integrates `|p'|` across
a derivative break.

### Cost, and why it is affordable

The prototype is 657x slower per call, but that is a Python loop over points in
`s_of_t`; vectorized it is ~4 Newton iterations x one Gauss rule, order 100x a
PCHIP call. It is paid ONCE per solve -- the node set is built once and reused for
every lambda -- against a current whole-quadrature build of 11 ms. Better still,
`t(s)` can be evaluated once at the quadrature nodes and cached.

Not implemented: this changes `ParametricSegment`, which everything else in lappy
builds on.

### Vectorized: the cost is ~1 ms per solve, and the knobs are cheaper than expected

The prototype's 657x slowdown was entirely a Python loop over points inside
`s_of_t`. Vectorized -- one `speed` call per Newton iteration over an
(n_points x gauss_order) block -- the picture is
`benchmarks/arclength/vectorized.py`:

    knob calibration (round-trip error)
                    G=6       G=8      G=12      G=24
    iters=1      4.0e-09   4.0e-09   4.0e-09   4.0e-09      (ellipse)
    iters=2      1.8e-16   1.8e-16   1.8e-16   1.8e-16
    iters=3      1.8e-16   1.8e-16   1.8e-16   1.8e-16

**`gauss_order` does not matter at all** -- G=6 is indistinguishable from G=24,
because the remainder span is a fraction of one adaptive-table panel. And Newton's
quadratic convergence is plainly visible: 4e-9 at one step, machine precision at
two. The spline needs three (1.3e-16); a circular arc is exact at one, since its
PCHIP guess is already the answer.

So the tuned configuration is `gauss_order=6, anchor_order=8, iters=3`, and it is
*identical in accuracy* to the expensive one -- boundary integrals at 1.4e-13
(order 128) against PCHIP's 6.9e-05, which no order improves.

    cost
      construction    0.11 ms per segment      (once per solve)
      evaluation      265-530 ns per point,
                      i.e. +0.13 ms (ellipse) / +0.22 ms (spline) for 250 nodes

Against a whole-quadrature build of ~11 ms that is **under 5%**, and it is paid
once per solve rather than per lambda. Per-call it is 30-120x a PCHIP evaluation,
which is the number that looks alarming and is the wrong one to look at: the
absolute cost is a fraction of a millisecond.

Trading eight orders of accuracy on curved boundaries for ~1 ms per solve.

---

## Arclength reparametrization: implemented, default-on, plus segment-declared panel breaks

`ParametricSegment` now SOLVES the arc-length inverse instead of interpolating it. The
adaptive table supplies bracketing nodes and an initial guess; `_t_of_s` runs a
safeguarded Newton on the exact `s(t)`, and `t'(s) = 1/|p'(t)|` is analytic.
Segments declare `break_ts` (native parameter) / `break_taus` (arc length) where a
panel must break, `SplineSegment` reports its interior knots, and
`eigfun_integrals._split_at_breaks` honours them -- splitting only a corner panel's
OUTER part, so the anchored singular end is never cut off.

    segment          roundtrip   |dp/dtau|-L   build    t(s) per 1000 pts
    circular arc      1.9e-16      3.8e-16     0.0 ms       0.13 ms
    ellipse           6.1e-16      3.7e-16     0.0 ms       0.34 ms
    cubic spline      9.5e-16      2.5e-16     0.8 ms       0.65 ms

`boundary_quadrature` on a spline-boundary domain now returns **8.9e-16** at every
requested precision, with `sum(w) - perimeter` exactly 0. Knot alignment does what
it promised: at 120 nodes, knot-panelled 3.3e-12 against 6.7e-07 for one global
panel.

### Three of my own errors on the way, all worth recording

**1. My test spline was degenerate, and it invented a phenomenon.**
`make_interp_spline(..., bc_type='periodic')` fed COMPLEX points silently discards
the imaginary part (ComplexWarning; `c.dtype` comes back float64). My "spline" was
therefore a back-and-forth segment on the real axis -- which genuinely has cusps,
`min|p'| = 0` at three parameters. I spent a while diagnosing "near-cusp behaviour
in splines", built a safeguarded Newton for it, and measured a 130x slowdown, all of
which were properties of a broken test object. Fitting the real (n,2) array
`SplineSegment` expects gives `min|p'| ~ 4.4-5.2`: no cusps, and 0.65 ms per 1000
points rather than 50.

The safeguard stays -- a bracketed fallback is nearly free (the bracket comes from
monotonicity of s, and well-conditioned points exit in 3 iterations) and near-cusps
are perfectly possible in splines fitted to real data. But it is insurance, not a
response to a demonstrated failure, and the notebook should say so.

The 130x cliff is now a `RuntimeWarning`: `_complex_vectorize` falling back to its
per-point Python loop was silent, and it is almost always a sign the segment was
built wrong. That warning would have saved the whole detour.

**2. An active-set bug that cost four orders.** Iterating only the unconverged
points is necessary -- without it a couple of stragglers drag the whole vector
through all 60 iterations (62 ms per 1000 points). But my first version evaluated
`f` at the current `t`, then stepped EVERY active point, including ones already
within tolerance, before dropping them -- moving converged points back off the root.
Round-trip went 6e-16 -> 8e-3. Fixed by not stepping converged points.

**3. `adaptive_polyline` must not use the exact map.** It makes very many small
recursive `p` calls while choosing nodes, and the solve's per-call overhead turned
segment construction into a timeout. It only selects WHERE nodes go, to `tol`, so it
now runs on the cheap guess map; `polyline_pts` still evaluates them through the
exact `p`, so nothing downstream inherits the guess's error.

### Calibration note

`_REMAINDER_ORDER = 6` is insensitive (a fraction of one table panel; order 6 equals
order 24). `_ANCHOR_ORDER` is not, and 8 was too low: the total length still moved
8e-11 between tol=1e-3 and tol=1e-6, leaving exactly the tol-dependence the solve
exists to remove. At 12 that is 5e-15, for no measurable cost.

---

## Certification moved to the boundary; all 44 domains re-run; buckets unchanged

`eps = sqrt(area) ||u||_inf,dOmega / ||u||_L2` had one boundary quantity and one
interior one. With `MPSEigensolver`'s corner-adapted boundary quadrature both are
now boundary integrals: `certify.boundary_l2` reads the norms off the Rellich Gram
of the (already L2-orthonormal) eigenfunction cluster. `build_solver(orthonorm=True,
lam_max=...)` attaches the node set; `bucket.py` and `runner.py` pass it by default;
`--no-orthonorm` / `l2_source='cubature'` reproduce the old path exactly.

**Result of the full re-run (44 domains, each at ITS best recorded `n_basis`,
tag `orth`): 32 / 10 / 2, identical to the previous run. No domain changed
bucket. No eigenvalue lost a digit.**

    worse:   ellipse_a2   -0.06 digits    (the only movement in either direction
                                           attributable to the change)
    better:  ellipse_a3   +0.05 digits
    "better": sector_reflex +1.8, sector_slit +2.3, sector_sharp_p65 +0.45
              -- NOT ours. Those jsonl records predate the `_bessel_zero` ->
              `mpmath.besseljzero` reference fix; BUCKETS.md's corrected table
              already lists 14.6 / 15.2, which is what the re-run reproduces.

### Why the numbers did not move, and why that is the expected result

`eps` is **scale-invariant**: it is a ratio in which `u`'s normalization cancels.
Orthonormalizing `u` does not change `eps` at all -- it changes what the
denominator *costs* and what it can be trusted to be. So "no change" was the
prediction, and getting it is the check that the new path is wired correctly.

Confirmed directly: re-certifying the same eigenvalues at four node densities
(`l2_source='boundary'`, no fallback) gives **identical digits to three decimals**
across a 4-6x refinement.

    ellipse_a2        46 nodes 13.541    250 nodes 13.541
    GWW1             204 nodes  8.920    572 nodes  8.920
    right_trapezoid   84 nodes 11.374    346 nodes 11.374
    reg_ngon_7        98 nodes 10.737    336 nodes 10.737

### What was actually bought

Not digits -- soundness and cost:

* **`spiral_t25`'s cubature mesh does not build** (>90s, killed). The interior
  norm was a hard dependency of certification on a mesh that some domains do not
  have.
* **The mesh was on the wrong side of the bound for `cut_square_r025`**:
  `sum(w) = 0.9509685` against `area = 0.9509126`. The bound needs an
  UNDER-estimate of `||u||_L2`; that mesh over-integrates. (Elsewhere it
  under-resolves and errs safe: disk loses 1.6% of its area, stadium 0.2%.)
* **Speed, where the mesh was large**: disk 128s -> 58s, mushroom 235s -> 110s,
  mushroom_thin 207s -> 120s. Its mesh is 101,700 points against a 166-node
  boundary rule. Suite total 5801s -> 5473s despite the boundary path *adding* a
  cubature cross-check on 22 domains.
* **Cluster orthogonality is now an output.** Degenerate pairs come back with
  off-diagonal Gram entries <= 2.4e-16, which the old path never measured.

### The x0-spread is a real signal, and `BoundaryQuad.precision` is not

The identity holds for every reference point `x0`, so disagreement across `x0` is
pure quadrature error and costs three extra Gram evaluations. It flagged **22 of
43** domains above 1e-8 -- while `bq.precision` advertised 1e-13 on 40 of them.
`precision` is a statement about the rule's model integrand, not about what is
actually integrated, and it should not be used as an error estimate. Worst case:

    sector_slit    x0-spread 6.9e-01     bq.precision claims 1e-13

`sector_slit` is the sharp case: its nu=0.504 corner is INADMISSIBLE and gets
demoted to a smooth rule (`boundary_quadrature` does warn), but `bq.precision`
does not record the demotion. The fallback caught it, used cubature for those
columns, and the domain still certified 13.4 / true 15.4, bucket 1.

### Where the spread comes from -- partly diagnosed, one part open

Tested and **ruled out**: the basis. `fs_frac=0.5` vs `0.0` on GWW1 changes the
basis from 316 to 332 columns and moves the spread by <1%. So it is a property of
the node set against the integrand, not of the approximant. (My first hypothesis
was that FundamentalBasis poles just outside the boundary create near-singular
features the geometry-only node set cannot know about. Wrong.)

**Smooth boundary: plain under-resolution, and it converges spectrally.**

    ellipse_a2    46 nodes 1.1e-05    78 nodes 1.6e-08
                 136 nodes 2.4e-13   250 nodes 2.3e-14

So the smooth-panel Nyquist sizing is thin by roughly a factor of 3 in node count
for this integrand -- unsurprising in hindsight, since the rule is sized for an
eigenfunction and the Rellich integrand is a PRODUCT of derivative data.

**Polygons: only algebraic, and not explained.** GWW1 goes 3.7e-05 -> 7.0e-07
over a 2.8x refinement; right_trapezoid 4.3e-05 -> 4.9e-07. Neither the trace
term (no correlation with `eps` across domains: GWW1 eps=1.2e-9 and
right_trapezoid eps=4.2e-12 have the same spread) nor node count explains it.

One structural asymmetry is worth recording as the leading suspect for whoever
picks this up: `default_x0` deliberately sits AT a singular corner, where `r.N`
vanishes on both edges and removes that corner's singularity outright. Every
probe `x0` is generic, so the probes re-activate a singularity that the
production configuration cancels exactly. If that is the mechanism, the spread is
a **stress test rather than the error of the value actually used**, i.e.
conservative in the right direction -- which is consistent with the digits not
moving at any node density. Not tested; the test is probe `x0` placed at other
singular corners of a multi-corner domain.

---

## The quadrature was missing its advertised precision, and `nu < 1` was why

The boundary-certification run left a loose end: `bq.precision` claimed 1e-13 on 40 of 44
domains while the measured x0-spread exceeded 1e-8 on 22 of 43. Chasing that down took four
dead hypotheses, two real mechanisms and one instrument change.

### Killed, in order, so nobody repeats them

| hypothesis | test | result |
|---|---|---|
| dropped `u != 0` trace terms | full 4-kernel Rellich form vs the Dirichlet branch | changes G by 5e-13, spread unmoved |
| FundamentalBasis poles near the boundary | `fs_frac` 0.5 vs 0.0 (316 vs 332 cols, GWW1); `fs_d` 0.25..4.0 (ellipse) | <1% change; identical to 3 s.f. |
| basis composition (FB vs FB+FS) | basis audit across the suite | `right_trapezoid` pure FB and dirty, `reg_ngon_6` mixed and clean |
| the MPS residual dominating the spread | fixed node set, `n_basis` 60 -> 480 | `sup|u|` falls six orders, spread identical to 3 s.f. |

The last one deserves emphasis because it was the most plausible: the eigenfunctions in
question are resolved to sigma ~ 1e-16 and `sup|u|` ~ 1e-15, which is exactly the regime
where the rule's promise is supposed to hold.

### Mechanism 1: the criterion was wrong, not the rule

Per-panel attribution (`benchmarks/eigfun_quad/sizing_audit.py` -- refine one panel, difference
the integral, repeat) localised it immediately. On GWW1 every corner-adapted panel BEATS its
model (ratios 0.01-0.3); every failure is a `legendre` panel at ratios 1e8-1e9.

`corner_specs` marked a corner singular when `nu < 1`, i.e. reentrant, with the comment "a
smooth rule is already exact" otherwise. The Rellich integrand is `r^(2nu-2)`, so what matters
is whether that exponent is an INTEGER. At nu=4/3 (a 135-degree corner) the eigenfunction is
bounded, its normal derivative `r^(1/3)` even vanishes -- and the integrand `r^(2/3)` defeats
Gauss-Legendre, which converges on `tau^gamma` at `n^-(2 gamma + 2)`. Fitted 2.64, 3.31, 4.96
against predicted 2.67, 3.33, 5.00.

Both directions matter. "nu is not an integer" is ALSO wrong: it hands the corner rule to
nu=6.78 corners, where `r^11.55` is already smooth, and makes them worse (1.2e-11 -> 8.9e-10)
because the corner rule is not exact on constants. `quad.smooth_power_error` asks the smooth
rule what it can deliver, which settles both cases.

### Mechanism 2: on a curved boundary, the parametrization can cost more than the eigenfunction

`int (r.N) ds = 2|Omega|` -- divergence theorem, no lambda, no basis, no eigenfunction -- is
the sharpest test in the subject and isolates the geometry completely. An ellipse fails it at
the node count the oscillation model asks for (1.3e-05 at 46 nodes) while disk, stadium,
mushroom, cut_square and every polygon are machine exact. Normalized arclength on a
varying-speed curve is analytic but its strip of analyticity narrows with eccentricity.

NOT the arclength table's tolerance, which was the natural suspect: identical to three digits
at tol 1e-4, 1e-6, 1e-8 while build cost goes 0.0s, 0.9s, 65s. Vary the suspected cause and
check that the symptom responds, before optimizing it.

### Instrument change: prefer refinement to x0-invariance

x0-invariance is a fine detector -- it found all of this -- but it is not an error bar,
because lambda enters the identity and an inexact lambda reintroduces x0-dependence. It
misleads in both directions: `L_shape` spread 3.3e-12 against quadrature 1.4e-13; `H_shape`
spread 4.1e-11 against quadrature 1.0e-08. `verify_gram` (refine, difference) measures the
quadrature alone. That also overturned a comment in `scripts/hshape_eigfunc_norm.py` blaming
H_shape's 1.4e-8 cubature discrepancy on the cubature -- the boundary rule was the weaker
method there.

That script was also unseeded, so its x0-spread ranged 1.2e-14 to 6.3e-12 across draws,
straddling its own 1e-12 bar. It passed or failed by luck.

### What the re-run showed, and what it could not

44 domains, defaults flipped: 32/10/2, no switches, no regressions, four small gains on
exactly the targeted domains (see BUCKETS.md). Cost 2.0x nodes for +1.5% wall time.

The certified digits could not have moved: `eps` is scale-invariant. `right_trapezoid`'s
quadrature improved ten orders and its bound is unchanged to two decimals. The suite is the
regression test; `verify_gram` is where the result is.

### Still open

1. **`stadium`.** Exact parametrization, integral nu, and its integrals sit at 1.5e-04
   because its boundary data carries far more harmonics than `2 sqrt(lam) L` predicts. No
   geometry-only rule can anticipate that. `smooth_safety` is the blunt lever (1.5e-04 ->
   1.0e-08 at 3x); a posteriori verification is the honest one.
2. **The corner model is pessimistic for convex non-integer nu**, by up to seven orders
   (`iso_tri_h05` reports 3e-08, delivers 4.8e-16). It drives orders to the cap, which is
   where reg_ngon_5/7/8's 7-9x node counts come from. Tightening it is cheap accuracy.
3. **`H_shape` has a floor near 3e-10** that neither more smooth nodes nor deeper refinement
   moves. Not diagnosed.

---

## H_shape's floor, and the ngons' node bill: two diagnoses

Both were left open by the `verified` re-run. Neither is a defect in the corner rule; both
are the *sizing* being wrong in opposite directions.

### H_shape: nothing exotic, just smooth panels ~8 nodes short

Killed first, cheaply, so nobody repeats them:

* **catastrophic cancellation** in the boundary sum. Measured `sum|w f| / |sum w f|`: H_shape
  1.59, GWW1 1.14, L_shape and square 1.00. Under a digit lost anywhere. Dead.
* **the integrand being out of class.** Fitting `log|du/dn|` against `log r` on the edge
  leaving each singular corner recovers the exponent `nu-1` the rule assumes: H_shape's four
  reentrant corners give -0.3340, -0.3329, -0.3340, -0.3329 against -0.3333. Dead.

That second measurement threw off a bonus. **L_shape fits +0.3334, not -0.3333** -- for that
mode the leading singular coefficient vanishes and the next term `r^(2nu-1) = r^(1/3)` takes
over. L_shape is partly clean *by luck of the mode*, which is worth knowing before treating it
as the easy control it looks like.

Per-panel attribution then localised H_shape immediately: every corner-adapted panel BEATS its
model (ratios 0.01-0.12); the error sits in four `legendre` panels of order 14, each ~600x
worse than the smooth model claims. Their self-convergence (everything else held fixed) is
spectral and fast:

    order        14        20        28        40
    panel  9   2e-13     2e-15     1e-17     3e-18
    panel 16   2e-13     2e-15     2e-17     2e-17

So order 14 is simply too low by about 8 nodes. No floor, no barrier -- the same
under-prediction as the ellipse, milder.

**But the amount it is short by is a property of the BASIS, not the domain.** Varying only the
fundamental-solution placement, at comparable eigenfunction quality, moves the order-14 error
across four orders:

    fs_C  fs_sigma   sigma(lam)   err at order 14
    10.0       1.0      5.9e-11           2.8e-11
     3.0       1.0      5.2e-11           1.6e-13
    10.0       0.3      4.2e-10           3.9e-15
    10.0       3.0      1.2e-10           5.8e-15

A geometry-only node set cannot know where a basis put its poles. So H_shape belongs with
`stadium` in one class, not two: **the approximant's own singularity structure sets the
smooth-panel requirement, and only an a posteriori check can see it.** `verify_gram` is the
instrument; `smooth_safety` is the blunt lever (H_shape 1.0e-08 -> 4.6e-10 at 2x, +9% nodes).

An interim measurement that misled me for one step, recorded because the mistake is easy:
comparing `|G(panel i at order o) - G_ref|` looks like a per-panel convergence study and is
not -- it includes every OTHER panel's error, so it plateaus at their sum and reports a floor
that is not there. Hold everything else fixed and self-converge the one panel.

### reg_ngon_5/7/8: the corner model is a worst case, and it is being used as a predictor

These pay 7-9x nodes (reg_ngon_7: 98 -> 868) because `corner_order_for_precision` drives the
order to its cap of 64. Measured against the truth, the order is 2x more than needed:

    reg_ngon_7   order    12        24        32        48        64
                 model   2.6e-04   5.9e-03   1.6e-03   6.8e-05   3.9e-06     <- non-monotone!
                 true    1.6e-08   8.8e-12   3.6e-13   1.6e-13   5.1e-14

The model never reaches 1e-13, so the scan returns its argmin, which is the cap. The true
error is monotone and clears 1e-13 by order 32-48.

Not ill-conditioning: the interpolatory weights are clean (`sum|w|/|sum w|` = 1.00-1.23 at
every order and every nu tested), and the exponent collisions that `nu = p/q` creates
(nu=7/5 collides at j=5, nu=4/3 at j=3) are evidently handled.

It is the model integrand's **amplitudes**. `corner_model_error` builds a radial series to
depth `n_j=6` with coefficients `(k/2)^(2q)/q!^2`, which at k~4-8 peak around q~4 and put
large weight on exponents a real eigenfunction barely populates. Calibrated against EXACT
sector eigenfunctions at convex non-integer nu (the case that matters here), with a generic
x0 so nothing cancels:

    nu=1.4     order      8        12        16        24        32        48
    TRUE              2.5e-04   3.1e-06   2.1e-07   6.0e-10   1.5e-11   2.8e-14
    n_j=2             1.0e-03   1.0e-04   3.2e-05   1.9e-06   7.9e-08   2.7e-10
    n_j=6 (default)   1.6e-03   3.6e-04   4.4e-04   2.1e-04   3.1e-05   7.1e-07

`n_j=2` bounds the truth at every order and every nu measured (1.4, 4/3, 5/3 x six orders)
while being 4-5 orders tighter than the default, and it decays at roughly the right rate
instead of stalling. That is the shape a predictor should have.

**Not changed yet, deliberately.** Retuning `n_j` moves the order chosen on every singular
corner in the suite, including the reentrant ones this calibration did not cover, so it needs
its own validation against the sector bars and a re-run before it becomes a default. Recorded
here with the data so that work starts from a measurement rather than a guess. Note also that
the current setting errs conservative -- it buys nodes nobody needed, which is the safe
direction.

## The Hadamard contract, and the corner-moving weight it immediately broke

`tests/test_shape_derivative.py` (12 tests, ~2s) is the permanent contract promised in
`docs/scope_and_downstream.md` §4: rectangle edge translation against closed form, a degenerate
cluster compared by matrix EIGENVALUES (the eigenspace basis is arbitrary), dilation recovering
Rellich, and the sector radius. All pass at machine precision.

The corner-moving case did not. On a reentrant sector with the EXACT eigenfunction, so any
error is the quadrature's alone, `dlam/dalpha` was capped near 1e-06 and did **not** improve
with refinement (6.2e-06 at 90 nodes, 3.9e-06 at 166, still ~1e-06 at precision 1e-14).

The cause is not parity, which is how it first presented. `sub = nu` rationalizes the
eigenfunction's own family, but sends a weight `r^m` to `t^(m/nu)` -- a non-integer power with
a SMALL exponent, on which Gauss decays only as `n^(-(2m/nu+2))`. `sub = 1/2` reverses which
half is exact: integer `m` becomes the exact polynomial `t^(2m)`, and the Bessel family becomes
`t^(2 j nu)`, non-integer but with exponents growing by `2 nu` per term, which Gauss eats.

    alpha      nodes  even       nodes  integer
    0.75 pi     178   3.2e-08      86   4.3e-14
    1.50 pi      90   6.2e-06      94   3.0e-14
    1.75 pi     188   2.9e-06     152   2.8e-14

Six to eight orders, at FEWER nodes in half the cases. It assumes nothing about `nu`, so it
also covers the generic arc-arc corner where `corner_substitution` reports no exact
substitution exists (verified to 1e-15 at nu = 1/1.37 and nu = 1/phi). Landed opt-in as
`boundary_quadrature(..., weight_family='integer')`; the default is unchanged.

**`cornerinterp` cannot be rescued, and it is a NODE problem.** Two alternatives were measured
and rejected first. `sub = 1/q` on the dense family works (4e-15 even at nu=4/5) but needs
rational `nu`. Rebuilding `cornerinterp` on the dense exponent set fails outright -- 2.0e-06 at
order 32, WORSE than on the sparse set -- and raising `n_exp` runs `cond(V)` to 4.4e18. It is
not arithmetic: reconstructing the same rule at 60 dps makes it worse still, `sum|w|` reaching
4e10. The Jacobi nodes are simply wrong for that family; fixing it means solving for nodes AND
weights jointly (Ma-Rokhlin). `sub = 1/2` made that unnecessary.

## The corner order model is wrong in BOTH directions; `n_j=2` is not the fix

Following the previous entry's recommendation, `n_j=2` was tested -- and it is not enough,
because the previous calibration covered only convex `nu`. Measured against closed-form truth
over 3822 configurations (`benchmarks/eigfun_quad/corner_model_calibration.py`: alpha/pi in
[0.6, 1.9], k in [0.5, 10.6] which is the measured suite range, every order 6..64, restricted
to the band where the value actually decides an order):

    model                 worst optimism   %opt >2x   median |log10 ratio|
    unsigned n_j=6              1.0e-09      32.7%           1.68    <- default
    unsigned n_j=2              4.5e-09      29.4%           1.46
    max(signed, unsigned)       7.6e-06      13.0%           1.20

The two regimes fail oppositely, which is exactly why neither was noticed -- each looked fine
on the half of the range the other covered:

    reentrant nu=0.75, order 16   true 3.7e-09   model 4.1e-10   OPTIMISTIC 9x
    convex    nu=1.4,  order 24   true 6.0e-10   model 2.1e-04   PESSIMISTIC 3e5x

Two causes. Unsigned coefficients drop the `(-1)^q` of the real J_nu series, inflating
high-order content (pessimistic). Fixed depth `n_j=6` cuts AT the series peak `q ~ k/2` once
`k > 8`, making the model integrand smoother than the real one (optimistic); `k` reaches 10.6
in the suite, so this is live.

**Three candidate replacements were built and all three rejected on measurement.** Signed +
k-adaptive had the best aggregate and fixed `chevron_2_3`'s spurious shortfall (568 -> 152
nodes) -- but cancellation shrinks the model integral, so a relative error against it
understates, and it FAILED an exact test: H-polyomino 288 -> 232 nodes, claiming 1e-14 while
delivering 2.7e-12. The `max(signed, unsigned)` envelope fixed that but the pessimistic branch
then dominates: claimed precision collapsed on 12 suite domains and nodes inflated 43-73% on
six. `lappy/quad.py` therefore keeps its current model; only the INSTRUMENT landed.

**A perfect corner model would still not deliver the advertised number.** `chevron_1_2` claims
1e-13 while `verify_gram` measures 4.9e-08, and that gap is IDENTICAL under every corner model
tried. The error is not at the corners -- it is the smooth panels, whose requirement is set by
the basis's own pole placement (see the H_shape entry above), which no geometry-only model can
see. The open proposal is to demote `bq.precision` to a sizing heuristic and let `verify_gram`
certify. Not decided; recorded in `docs/orientation.pdf` §5.

### Method notes, both of which cost time here

**`mpmath.quad` is not usable as truth for corner integrands -- fifth occurrence.** The first
calibration harness reported `nu=0.5714` errors pinned at 5.4e-07 across every order; the
reference itself was only good to 5.3e-07. The harness now self-checks at two precisions, and
its truth is a closed-form double series (`exact_corner_integral`), not quadrature.

**Do not monkeypatch a module global to build a before/after table.** Twice in this session a
comparison script captured the "before" function AFTER a previous loop iteration had already
overwritten it, producing confident and entirely wrong tables -- once claiming a basis result,
once claiming `chevron_2_3` at 272 nodes/2.1e-13 when a clean process measured 240/3.1e-02.
Run one model per process, or capture every function object before any patching.

**Aggregate statistics must not outrank an exact test.** The signed model won on 3822
configurations and lost on one polyomino with closed-form truth. The polyomino was right. Gate
candidates on Legs 1-3 FIRST, then consult the grid.

## `sizing_precision`, and what `cornerinterp` actually does on a dense corner family

Two decisions and one measurement session, continuing directly from the entry above, which left
the `bq.precision` demotion as an open proposal and recommended validating `weight_family=
'integer'` before it became a default. Both are now settled, in opposite directions.

**The demotion landed (f6baf59).** `BoundaryQuad.precision` is now `sizing_precision`. Value and
semantics unchanged -- still the model bound that chose the orders, still `inf` on a demoted
corner -- but it no longer reads as an accuracy claim, the warning says "could not size for",
and `verify_gram` is named as the certificate. The `bq_precision` key in `buckets.jsonl` is
deliberately unchanged so existing records still parse.

**`weight_family='integer'` must NOT become the default, and the earlier claim for it was
overstated.** The previous entry's reasoning was that `sub = 1/2` sends the Bessel family to
`t^(2 j nu)`, "which Gauss resolves at once". That holds for a SPARSE member -- the single
(m,n) sector mode, whose squared normal derivative has exponents spaced by 4. A real corner
series has cross terms spaced by `2 nu`, and there `sub = 1/2` converges only algebraically.
L_shape corner panel, nu = 2/3, Leg 3 synthetic series against closed-form truth:

    order            8         16         32         64        128
    sub = nu    7.8e-16    2.7e-15    1.5e-14    6.1e-14    9.2e-14
    sub = 1/2   4.1e-06    2.8e-07    1.8e-08    1.2e-09    7.2e-11

About `n^-4.7`, never reaching machine precision at a usable order, where `sub = nu` is exact at
order 8. Flipping the default fails 12 tests including all of Leg 3 (3.6e-06 against a 1e-12
bar, at the SAME node count). It is the original corner-moving defect with the roles swapped.
The two families are a genuine trade, not a better and a worse: `sub = nu` for the
eigenfunction's own dense family (Rellich, Gram), `sub = 1/2` for an integer-power weight on top
of a sparse one (Hadamard corner-moving). Leg 1 alone cannot see this, because its
eigenfunction IS the sparse case -- where 'integer' ties or wins on half the nodes. That is
exactly how the claim came to be overstated, and it is the same shape as the `n_j=2` mistake:
a calibration that saw only half the range. The rationale now sits in the code at the branch.

The wrapper for downstream (`douse`) use is still wanted; it should preset `weight_family` at
the call site rather than move the default.

### The coverage hole this exposed, and closing it (7342b7f)

Asking why Leg 1 and Leg 3 disagreed turned up something worth more than the original question.
**Leg 3 -- the certifying leg -- only ever exercises one corner rule at one exponent.**

    L_shape      cornerjac    nu=0.6667
    H_shape      cornerjac    nu=0.6667
    sector1.5    cornerjac    nu=0.6667
    sector1.1    cornerinterp nu=0.9091      <- Leg 1 only, and Leg 1 is SPARSE

`corner_rule_spec` picks 'cornerjac' only for a straight edge with integer `2/nu`, which among
reentrant angles is essentially `alpha = 3pi/2` alone. Every other corner -- any non-right
polygon angle, every curved edge -- runs 'cornerinterp'. So the rule generic domains actually
use had never met a dense multi-term family with real singular amplitude. `chevron(h1, h2)`
supplies one: a nonconvex quadrilateral whose reentrant `nu` varies continuously, straight
edges throughout, so the corner series stays exactly the right class and the reference stays
closed form.

**It holds down to nu ~ 0.77, and has a ceiling below nu ~ 0.6.**

    nu = 0.888    4.0e-15      chevron(0.2,1)
    nu = 0.772    3.6e-14      chevron(0.5,3)
    nu = 0.615    1.5e-12      chevron(1.5,4)
    nu = 0.587    5.2e-10      chevron(2,3)      ceiling
    nu = 0.556    4.1e-10      two_notch, two generic reentrant corners at once

**More order makes the ceiling WORSE, which makes the order cap load-bearing.** Forcing the
order past `corner_order_for_precision`'s `order_max=64`:

    order        chevron(2,3) nu=0.587    two_notch nu=0.556
       32               9.5e-11               4.7e-10
       64               4.9e-10               4.1e-10
      128               1.2e-08               1.0e-06
      192               2.3e-07               9.1e-06

against controls at nu = 0.888 and 0.772 that stay flat at 1e-15/1e-14 to order 192. So it is
specific to the near-slit regime and is not a general high-order defect. `sum|w|` holds at
0.285 and the weights stay positive, so this is NOT the weight blow-up that killed cornerinterp
on the dense exponent set -- it is accuracy loss in the interpolatory solve itself. Raising the
cap is not the fix; the real levers are panel subdivision or a joint node/weight solve
(Ma-Rokhlin). Note also that the sizing model picks the wrong side of the optimum here: order
32 is both more accurate and cheaper than the 64 it chooses. That is the next cheap win.

Not silent, which is what makes this a known limitation rather than a trap:
`boundary_quadrature` warns and records a shortfall on both domains. But the reported bound is
450x optimistic on chevron(2,3) (1.15e-12 claimed, 5.2e-10 delivered) and 90x pessimistic on
chevron(1.5,4) (1.33e-10 claimed, 1.5e-12 delivered) -- a third independent confirmation that
the corner model errs in BOTH directions, now on the rule axis rather than the nu axis.

**Assessment, for the record.** Rellich orthonormalization on generic non-pathological domains
is in good shape and now has exact-truth evidence at the rule those domains actually run, for
nu >~ 0.65 (reentrant angles to about 1.55pi). Beyond that there is a ~1e-10 ceiling with no
lever currently available. What is still missing for "to a target precision" is a closed loop:
`verify_gram` is correct and is called by NOTHING in `lappy/` -- it appears only in tests and
one script. Sizing model plus a posteriori measurement, with no refinement between them.

### Method notes

**A geometrically-damped corner series is not a class member at large nu.** `_corner_series`
uses `0.4^j` as a stand-in for the expansion's true `1/Gamma(k nu + q + 1)`. At `nu = 22` that
demands exponents 21, 43, 65, 87 with O(1) amplitudes -- a squared integrand of degree 174 --
where the physical k=1 amplitude is ~1e-21. It convicted `cornerinterp` at 1.7e-02 on corners
the physical series integrates to 4.4e-16, and I reported that as a finding before checking it.
`_bessel_corner_series` now builds amplitudes from the Bessel series itself (in log space; `k
nu` passes 20 at a sharp corner), carrying both the Gamma damping and the `(-1)^q` alternation.
Harmless for L_shape and H_shape at nu = 2/3, so the existing legs never saw it. **The control
is what caught it:** L_shape was carried through the new harness precisely so a wrong harness
would be visible, and it reported 6.7e-16 while everything else burned.

**A canary that perturbs the shared quantity measures nothing.** Perturbing `corner_specs`' nu
to test whether the new bars have teeth moved the rule AND the model I was scoring against, so
they stayed consistent and the error did not budge -- which reads as "cornerinterp is
insensitive to nu, unlike cornerjac", an interesting and entirely false finding. Holding the
model at the true nu, at nu = 0.7721 order 28: correct 1.7e-13, rule nu off by 3e-4 gives
7.0e-06, a smooth Legendre rule gives 1.7e-02. The bars do have teeth. Same family as the
monkeypatch note above: when a before/after shows NO difference, suspect the harness first.

**One panel per adjacent edge, and they are not alike.** The first ceiling test asserted a
single panel at the reentrant corner and failed. A corner has one panel on each adjacent edge,
and at chevron(2,3) the same corner at the same order measures 5.2e-10 on one and 5.2e-14 on
the other. A test looking at one of them could have reported either story; the bar is the worst
of the panels.

## Correction: "cap the corner order near the slit" was a lam=1 artifact

The previous entry closed with a recommendation -- that `corner_order_for_precision` picks the
wrong side of the accuracy optimum for near-slit corners, since order 32 beat the chosen 64 on
both accuracy and node count, and called it the next cheap win. **It is wrong, and no code
changed.** Recording it because the recommendation is in the previous entry and in a commit
message, and because the mistake is the same species as the two already logged here.

Every measurement behind it was taken with `boundary_quadrature(dom, lam_max=1.0)`. A corner
order serves two demands: resolving the singular exponent family, and resolving oscillation at
wavenumber `k`. At lam=1 there is no oscillation, so only the first is exercised, and the
saturation point sits early. It moves right, fast, with lam -- true optimum order, measured
against closed-form truth on the same panels:

                              lam=1   lam=100   lam=1000
    chevron(2,3)   nu=0.587      24        88        128
    chevron(0.5,3) nu=0.772      40       128        128

At lam=100 the chosen order of 64 is if anything too LOW, and a fixed cap at 32-40 would have
been a clear regression on every domain in the suite, which runs lam into the hundreds. The
model's disagreement with the truth is still real -- the model error falls monotonically to
1e-16 while the true error saturates and then degrades -- but the fix is not a cap, and the
"more nodes AND less accuracy" framing only holds below the saturation point's own lam.

What survives at every lam: the near-slit ceiling itself (~1e-9 to 1e-10 at nu ~ 0.56-0.59,
whatever the order), and the degradation PAST the optimum wherever the optimum happens to sit.
`tests/test_eigfun_integrals.py::test_leg3b_more_order_does_not_rescue_a_near_slit_corner_at_low_k`
is renamed accordingly and carries the table, so the qualifier cannot go missing again.

**Method note, fourth of its kind.** The first three were: mpmath.quad as truth, a monkeypatched
before/after, and a geometrically-damped series at large nu. This is the same shape --
measuring in a regime that is not the operating regime and generalizing. The tell was available
before the measurement: `_score_corner_panels` hardcodes `lam_max=1.0`, and nothing in the
suite runs at lam=1. Ask what the harness holds FIXED before reading a curve off it.
