"""Cross-check every polygon reference table against an independent full-domain solve.

WHY THIS EXISTS. `benchmarks/suite/run/reference_values.REFERENCE` is the ground truth that
basis studies, certified-accuracy claims and (soon) a downstream shape package are all scored
against, and until now nothing checked it. Two of its entries were each short a mode. Both were
found by accident and by an argument from outside the codebase -- `reg_ngon_8` by Faber--Krahn
while measuring something else, `reg_ngon_6` by certifying a disputed cluster during a solver
review -- never by a test.

A reference that is wrong in this particular way is the worst kind: the values it does list are
accurate to 12+ digits, so anything comparing against it sees small errors on most entries and
one large one, which reads as a solver problem rather than a data problem. The cost is measured
in weeks of debugging the wrong component.

THE MECHANISM, so the guard is aimed at something real. 24 of 27 entries came from a
symmetry-reduced solve, and `benchmarks/reference/symsolve.solve_sym` documents the hazard
itself: the registered group is "the largest elementary abelian 2-subgroup with real
characters", which can be a PROPER SUBGROUP of the domain's true symmetry, and then degeneracies
the full group would split survive inside one sector and have to be recovered by that sector's
own multiplicity estimate. Exactly two suite entries reduce a domain by a proper subgroup --
`reg_ngon_6` (D6, reduced by D2) and `reg_ngon_8` (D8, reduced by D2) -- and exactly those two
are the ones that were wrong. Every `|G|=2` mirror/half-turn entry is correct, and `rect D2` is
correct because D2 *is* a rectangle's full group.

So the instrument is a full-domain solve, which shares no code with the symmetry path.
"""
import warnings

import numpy as np
import pytest

from lappy import Eigenproblem
from lappy.asymp import weyl_count_check
from benchmarks.suite.domains import SUITE
from benchmarks.suite.run.reference_values import REFERENCE

K = 10
RTOL = 1e-6      # "is this the same eigenvalue?", not "how accurate is it?"


def _polygon_keys():
    keys = []
    for key in sorted(REFERENCE):
        try:
            dom = SUITE[key].build()
        except Exception:
            continue
        if dom.bdry.is_polyline:
            keys.append(key)
    return keys


POLYGONS = _polygon_keys()


@pytest.mark.slow
@pytest.mark.parametrize('key', POLYGONS)
def test_the_reference_table_matches_an_independent_full_domain_solve(key):
    """No index may differ. A table short a mode shows up as every entry above it shifted."""
    ref = np.asarray(REFERENCE[key]['eigs'], dtype=float)
    dom = SUITE[key].build()
    k = min(K, len(ref))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = np.asarray(Eigenproblem(dom, precision=1e-10).solve(k), dtype=float)

    m = min(len(got), len(ref), k)
    bad = next((j for j in range(m) if abs(got[j] - ref[j]) > RTOL*abs(ref[j])), None)
    assert bad is None, (
        f'{key} ({REFERENCE[key].get("method", "?")}): reference and full-domain solve first '
        f'differ at index {bad + 1}\n'
        f'  reference : {np.array2string(ref[:m], precision=7, max_line_width=200)}\n'
        f'  full solve: {np.array2string(got[:m], precision=7, max_line_width=200)}\n'
        f'  Weyl deviation, reference : '
        f'{np.array2string(np.round(weyl_count_check(ref[:m], dom), 2), max_line_width=200)}\n'
        f'  Weyl deviation, full solve: '
        f'{np.array2string(np.round(weyl_count_check(got[:m], dom), 2), max_line_width=200)}\n'
        f'  (a table missing a mode drifts NEGATIVE in the deviation; that is which side is '
        f'wrong)')


@pytest.mark.slow
@pytest.mark.parametrize('key', POLYGONS)
def test_no_reference_table_lags_the_weyl_count(key):
    """A cheaper, weaker check that needs no solve, kept because it runs on entries the solve
    cannot reach and because it is the argument that caught `reg_ngon_8` originally.

    The two-term deviation oscillates and is O(sqrt(lam)), so no single value means anything --
    a persistent DOWNWARD drift does. Both bad tables ran negative over their last five entries
    (reg_ngon_6 to -0.72, reg_ngon_8 to -1.00) where every correct one does not.
    """
    ref = np.asarray(REFERENCE[key]['eigs'], dtype=float)
    dom = SUITE[key].build()
    dev = weyl_count_check(ref, dom)
    tail = dev[-5:] if len(dev) >= 5 else dev
    assert np.mean(tail) > -0.5, (
        f'{key}: the Weyl count deviation drifts negative over the last entries '
        f'({np.array2string(np.round(tail, 2))}), which is what a table missing a mode looks '
        f'like. Cross-check against a full-domain solve.')


def test_every_polygon_entry_is_actually_covered():
    """A guard on the guard: if `SUITE` or `REFERENCE` changes shape and `_polygon_keys` starts
    returning nothing, both tests above would vacuously pass."""
    assert len(POLYGONS) >= 15, POLYGONS
    assert 'reg_ngon_6' in POLYGONS and 'reg_ngon_8' in POLYGONS, POLYGONS
