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
