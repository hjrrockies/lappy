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
