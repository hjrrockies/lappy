"""`solve(k)` must return the FIRST k eigenvalues, not merely k of them.

WHY THIS FILE EXISTS. The failure it guards is silent, and it is the worst kind for a downstream
shape-optimization package: `solve(k)` returns k values, every one a genuine eigenvalue accurate
to 11-13 digits, the count is right, no warning is raised -- and one is missing from the bottom,
so every index above it is shifted by one. An optimizer tracking lambda_3 across a shape family
would follow a confident wrong gradient across the swap.

THE SWEEP IS THE POINT. Both genuine failures found sat next to passing cells at the same
domain -- `right_trapezoid` was wrong at k=10 and correct at k=9 and k=11 -- so a test that
checks one `k` per domain passes while the contract is broken. That is exactly how this
survived. `benchmarks/basis_lab/PLAN_LAB.md` records the identical methodological miss in the
`sharp_ref` sweep ("a knob sweep should cover the target ladder, not one rung").

WHAT THE CAUSE TURNED OUT TO BE. The initial scan grid was too coarse. Swept over ten suite
polygons x k in 2..10 at precision 1e-10:

    ppl=5    9 of 90 cells wrong        ppl=10   2 of 90        ppl=20   0 of 90

`H_shape` is why the default is 20 rather than 10: it drops lam_5 = 14.30522996 at k=7 and k=9
even at ppl=10, because its sigma well there is narrow -- 1.5e-10 at the mode, 3.4e-03 just
0.025 away -- inside a cluster of three modes spanning 0.4.

Nothing subtler was involved, and in particular the Weyl-count audit already in
`_solve_dir_neu` could not have caught any of it: measured per-gap expected counts overlap
completely between correct and incorrect cells (correct cells reach 2.87 expected modes in a
gap, incorrect ones span 2.27-2.67), because multiplicity confounds the two-term Weyl count at
these wavenumbers. So this file pins the OUTCOME, not a detector -- there is no validated
detector to pin, and `ppl=20` is therefore validated rather than proven: a domain with a
tighter cluster than H_shape's could still defeat it.

`reg_ngon_6`'s reference table was itself short a mode when this file was written; it is
fixed now (see `benchmarks/suite/run/reference_values.py`) and scored normally here.
"""
import warnings

import numpy as np
import pytest

from lappy import Eigenproblem
from benchmarks.basis_lab.heur import reference_eigs
from benchmarks.suite.domains import SUITE

# Four domains rather than the ten the diagnosis swept: the sweep is ~10 s per cell, so the full
# grid costs ten minutes of suite time and most of it re-confirms domains that never failed.
# Kept: `right_trapezoid` (the headline failure), `eq_tri` (the other genuine one, at k=5),
# `reg_ngon_6` (degenerate spectrum, and the corrected-reference case), `square` (regular
# control). `L_shape`, `parallelogram_60`, `reg_ngon_5`, `iso_tri_h1`, `GWW2` were clean at every
# k and every ppl and are covered elsewhere; `H_shape` has its own targeted test below rather
# than a full sweep, because it is the slowest domain in the suite.
DOMAINS = ['square', 'eq_tri', 'right_trapezoid', 'reg_ngon_6']
KS = tuple(range(2, 11))
PRECISION = 1e-10     # where the failures were measured; 1e-8 hides right_trapezoid's
RTOL = 1e-6           # far looser than the ~1e-11 these solves reach: "same eigenvalue?", not "how good?"

def _truth(key):
    return reference_eigs(key, max(KS))[0]


def _first_mismatch(eigs, truth, k):
    """1-based index of the first returned value that is not the eigenvalue it should be, or None.

    A short return counts as a mismatch at the first missing slot: `solve(k)` promising k values
    and delivering fewer is the same broken promise, reported the same way.
    """
    if len(eigs) < k:
        return len(eigs) + 1
    for i in range(min(k, len(truth))):
        if abs(eigs[i] - truth[i]) > RTOL * abs(truth[i]):
            return i + 1
    return None


def _solve(key, k, **kw):
    dom = SUITE[key].build()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(Eigenproblem(dom, precision=PRECISION).solve(k, **kw), dtype=float)


@pytest.mark.slow
@pytest.mark.parametrize('key', DOMAINS)
def test_solve_returns_the_first_k_for_every_k(key):
    """No index may shift, at any k in 2..10. Swept, because the failures neighboured passes."""
    truth = _truth(key)
    bad = {}
    for k in KS:
        eigs = _solve(key, k)
        i = _first_mismatch(eigs, truth, k)
        if i is not None:
            got = eigs[i - 1] if i <= len(eigs) else float('nan')
            bad[k] = f'first wrong at index {i}: got {got:.10f}, want {truth[i-1]:.10f}'
    assert not bad, (f'{key}: solve(k) returned the wrong set at k={sorted(bad)}\n'
                     + '\n'.join(f'  k={k}: {v}' for k, v in sorted(bad.items())))


@pytest.mark.slow
def test_the_right_trapezoid_regression_specifically():
    """The cell that exposed the whole class, pinned on its own so a failure names it directly.

    k=10 dropped lam_3 = 44.9484877814 while k=9 and k=11 were correct to 11-13 digits -- so the
    search was not short of resolution in general, it stepped over one bracket at one grid.
    """
    truth = reference_eigs('right_trapezoid', 12)[0]
    for k in (9, 10, 11):
        eigs = _solve('right_trapezoid', k)
        assert _first_mismatch(eigs, truth, k) is None, \
            f'k={k}: {np.array2string(eigs[:6], precision=7)} vs {np.array2string(truth[:6], precision=7)}'
        assert abs(eigs[2] - 44.9484877814) < 1e-8, f'k={k}: lam_3 = {eigs[2]:.10f}, dropped again'


def test_the_default_scan_grid_is_the_validated_one():
    """ppl=20 is the only value measured at 0 of 90; ppl=10 leaves H_shape wrong at k=7 and k=9.

    A unit test rather than a solve, so the guard survives even if the sweeps above are
    deselected for time.
    """
    import inspect
    assert inspect.signature(Eigenproblem.solve).parameters['ppl'].default == 20


@pytest.mark.slow
def test_a_coarser_grid_is_what_broke_it():
    """The diagnosis itself: at ppl=5 the failure returns, at the default it does not.

    Pins the CAUSE, so that if someone lowers the default again this file says why it matters
    rather than only that something broke.
    """
    truth = reference_eigs('right_trapezoid', 12)[0]
    # `weyl_audit=False` is what pins the CAUSE. The audit added later repairs this case, so with
    # it on the coarse grid no longer misbehaves -- which is the point of it, and would otherwise
    # make this test read as "the cause has changed".
    coarse = _solve('right_trapezoid', 10, ppl=5, weyl_audit=False)
    assert _first_mismatch(coarse, truth, 10) == 3, \
        'ppl=5 no longer reproduces the lam_3 drop; the cause has changed and this file is stale'
    assert _first_mismatch(_solve('right_trapezoid', 10, weyl_audit=False), truth, 10) is None
    # ...and the audit repairs the coarse grid rather than merely detecting it
    assert _first_mismatch(_solve('right_trapezoid', 10, ppl=5), truth, 10) is None


@pytest.mark.slow
def test_the_H_shape_cluster_needs_the_finer_grid():
    """The cell that set the default at 20 rather than 10.

    lam_5 = 14.30522996 sits 0.37 above a pair at 13.9276/13.9316, and its sigma well is narrow:
    1.5e-10 at the mode against 3.4e-03 only 0.025 away. ppl=10 steps over it and returns
    17.70673522 in that slot -- which is itself a genuine SIMPLE eigenvalue (10.5 certified
    digits as mult=1, garbage as mult=2), so nothing downstream looks wrong.
    """
    truth = reference_eigs('H_shape', 10)[0]
    for k in (7, 9):
        eigs = _solve('H_shape', k)
        assert _first_mismatch(eigs, truth, k) is None, \
            f'k={k}: {np.array2string(eigs[:6], precision=7)}'
        assert abs(eigs[4] - 14.30522996) < 1e-7, f'k={k}: lam_5 = {eigs[4]:.8f}, dropped again'
