# Overnight reference-value run — RESUME PROTOCOL

**If you are a fresh/compacted context: read this file first, then `queue.json`,
then the tail of `NOTEBOOK.md`. Everything you need is on disk. Do not try to
remember anything.**

## The job

Produce certified reference values for the first 10 Dirichlet eigenvalues of
every domain in `benchmarks/suite/domains.py`, and write down *why* each domain
reaches the precision it does.

- **Target: 8 correct digits**, verified by the Moler--Payne certified bound
  (`benchmarks/reference/certify.py`), not by the tension heuristic.
- Push past 8 where possible. Domains that cannot reach 8 are recorded as
  officially **hard**, with a written explanation of the mechanism.
- Extended precision may be used to *diagnose*, never to produce a reported
  value.

## Ground rules (from the user)

1. New code in `lappy/` is allowed, but **must not change existing behavior or
   defaults**. New functions / classes / opt-in kwargs only. Failed experiments
   stay in `benchmarks/`.
2. Commit to branch `reference_values_run` as work completes. Never push.
3. Breadth first: bank certified values for everything at known-good configs,
   then attack the hard domains with the remaining time.

## Context-survival rules (IMPORTANT)

The run is longer than one context window. Obey these or the run degrades:

- **Never print a full result array into the transcript.** Runners write JSON to
  `results/`; read back only one summary line per domain (`status.py` does this).
- **Update `queue.json` and append to `NOTEBOOK.md` after every domain.** These
  are the memory. If it is not written down, it did not happen.
- Long solves run as **background subprocesses with a timeout**, one domain per
  process. A hung domain must never block the run.
- Prefer `python -m benchmarks.suite.status` over re-deriving state by hand.

## Layout

    benchmarks/suite/run/
      PROTOCOL.md    this file
      queue.json     per-domain state machine; the single source of truth
      NOTEBOOK.md    append-only lab notes (findings, dead ends, explanations)
      results/       <key>.json, one per completed attempt
      logs/          <key>.log, stdout/stderr of each attempt

## Per-domain state machine (`queue.json`)

Each entry: `status`, `attempts`, `best_digits`, `best_result`, `notes`.

    pending   -> not yet attempted
    running   -> a subprocess is live (record its pid + start time)
    done      -> certified >= 8 digits; result file written
    hard      -> exhausted reasonable ideas below 8 digits; explanation in notes
    failed    -> crashed or timed out; see logs/

## Workflow per domain

1. Solve (symmetry-reduced when `domain_symmetry` supplies a group).
2. Certify with Moler--Payne. **The certified `eps` is the number that counts.**
3. Cross-check:
   - analytic domains: direct error against `truth_fn` (the strongest check);
   - Weyl two-term count for completeness (no missing/spurious eigenvalues);
   - GWW1 vs GWW2 must agree (isospectral).
4. Record. If < 8 digits, diagnose before escalating:
   - `common.diagnose()` -> resolution-limited vs conditioning-limited;
   - `n_reg/n` ratio: if ~60-70% and insensitive to collocation density, the
     basis is intrinsically near-dependent (a *conditioning* problem);
   - only then escalate `n_basis`.

## Ideas queue (levers not yet exploited)

`common.build_solver` deliberately bypasses `MPSEigensolver.from_domain`, so
these are all currently OFF in the reference pipeline:

1. **Rellich L2-orthonormalization** (`lappy/rellich.py`,
   `from_domain(rellich=True)`) — never used in reference production.
2. **Cauchy data / singularity subtraction** (`lappy/cauchy.py`) —
   `build_solver` passes `cauchy_data=None`.
3. **Per-column normalization.** `TUNING_LOG.md` records `n_reg/n ~ 60-70%`
   *intrinsically*, insensitive to collocation density. Hypothesis: at a sharp
   corner the FB exponents reach ~334 (chevron), so columns differ by hundreds
   of orders of magnitude and the GSVD discards them as noise. This would be a
   **scaling artifact, not a resolution limit** — the single most promising
   idea in the queue. See `bases.NormalizedBasis` / `to_normalized`.
4. **Pivoted-QR column selection**, as `symmetry.prune_columns` already does for
   symmetrized bases; generalize to plain bases.
5. **Extended precision as diagnostic** (`bases.ExPrecFBBasis`): if a domain
   gains many digits in extended precision at the *same* basis, it is
   conditioning-limited; if not, it is truncation-limited. Decides which of the
   above to spend time on.

## Instruments

The analytic tier is the measuring device, not busywork. `reference.sector_eigs`
takes an arbitrary opening angle, so `disk_sector` sweeps the corner exponent
`p` from 0.504 (near-slit) to 13.3 (sharp) **against exact truth**. Test any
proposed fix there first: it gives a true error curve vs `p`, instead of one MPS
run compared against another.
