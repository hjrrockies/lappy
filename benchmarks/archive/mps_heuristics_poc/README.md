# The `docs/mps_heuristics.pdf` proof of concept, retired 2026-08-13

`heuristics.py` lived at `lappy/heuristics.py` and was the first implementation of the paper's
closed-form basis recipe; `test_heuristics.py` (31 tests) pinned its formulas. Both are archived
rather than deleted because neither was ever committed, so a delete would be unrecoverable.

**Superseded by `lappy/basis_plan.py`.** The evidence is in `benchmarks/basis_lab/HEURISTICS.md`
(1154 measurements of this recipe) and `benchmarks/basis_lab/PLAN_LAB.md` (the redesign). In short,
matched against this recipe on certified digits:

| domain | this recipe | basis_plan | note |
|---|---|---|---|
| square | 106 cols / 13.1 | 104 / 14.1 | |
| L_shape | 240 / 13.2 | 191 / 13.9 | |
| reg_ngon_6 | 318 / 11.2 | 174 / 11.1 | same accuracy, half the columns |
| iso_tri_h4 | 459 / 3.6 | 76 / 5.9 | |
| iso_tri_h16 | 480 / 5.9 | 173 / 5.7 | |

and the `precision` argument, measured *inert* here (flat to 1.5 digits across four decades on 12
of 18 domains), now responds monotonically.

To run the archived code, copy it back to `lappy/heuristics.py`. `benchmarks/basis_lab/heur.py`
still reports its recorded ledger (`run/heur/*.jsonl`) without it; only the stages that *build*
bases need the module.
