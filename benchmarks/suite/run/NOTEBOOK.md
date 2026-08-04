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
