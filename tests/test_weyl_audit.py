"""`solve` returns the FIRST k eigenvalues, and now audits that claim before making it.

The rescue loop `solve` already had asks "did I find ENOUGH?" and stops once the count is met, so
a scan that steps over one bracket and picks up a higher mode instead returns k accurate values
that are not the first k -- count right, values right, nothing raised. `solve`'s own docstring
records two such cases, and `douse` hit the same failure in production at a reentrant corner,
where the surviving values had a tension of 4.7e-12 because the badly resolved mode was the one
that went missing.
"""
import warnings

import numpy as np
import pytest

from lappy import Eigenproblem, Polygon
from lappy.asymp import weyl_deficit
from lappy.evp import WEYL_DEFICIT_TOL, _deficient_windows


def square_exact(n, L=1.0):
    m = np.arange(1, 4*n)
    return np.sort(((np.pi/L)**2*(m[:, None]**2 + m[None, :]**2)).ravel())[:n]


def unit_square():
    return Polygon([0, 1, 1 + 1j, 1j])


def test_the_detector_separates_a_clean_set_from_one_with_a_mode_deleted():
    """The calibration `WEYL_DEFICIT_TOL` rests on, measured on an exactly known spectrum."""
    dom, exact = unit_square(), square_exact(10)
    assert weyl_deficit(exact, dom) < WEYL_DEFICIT_TOL
    for drop in range(8):
        assert weyl_deficit(np.delete(exact, drop), dom) > WEYL_DEFICIT_TOL, drop
    # THE BLIND SPOT, pinned deliberately. Entries 8 and 9 are one degenerate level (167.783
    # twice); deleting one copy leaves the level standing, so no cut can see the shortfall and
    # the deficit reads clean at +0.319. Every cut lies BELOW the top level by construction --
    # which is why a caller must ask for more eigenvalues than its objective consumes.
    for drop in (8, 9):
        assert weyl_deficit(np.delete(exact, drop), dom) < WEYL_DEFICIT_TOL


def test_a_clean_set_asks_for_no_rescan_at_all():
    """The audit must cost nothing on the healthy path -- it runs on every solve."""
    dom = unit_square()
    assert _deficient_windows(square_exact(10), 1.0, dom) == []


def test_the_window_is_everything_below_the_crossing_not_the_gap_beside_it():
    """Weyl's count oscillates, so the crossing can sit well above the gap the mode is missing
    from: on H_shape the crossing was at 20.5 for a mode absent at 14.3."""
    dom, exact = unit_square(), square_exact(10)
    windows = _deficient_windows(np.delete(exact, 4), 1.0, dom)
    assert len(windows) == 1
    lo, hi = windows[0]
    assert lo == 1.0 and hi > exact[4]


def test_a_domain_with_no_corners_does_not_break_the_audit():
    """`weyl_count_poly` refuses a domain it cannot read angles off, and `solve` is not a polygon
    method. The disk reached the audit as a TypeError before the fallback existed."""
    from lappy.evp import _audit_count
    from lappy.geometry import disk
    assert np.isfinite(_audit_count(10.0, disk(1.0)))


@pytest.mark.slow
@pytest.mark.parametrize('name,k,bad_ppl', [('right_trapezoid', 10, 5), ('H_shape', 7, 10)])
def test_the_audit_repairs_the_cases_the_docstring_records(name, k, bad_ppl):
    """Both are documented in `solve`: `right_trapezoid` drops lam_3 at ppl=5, `H_shape` drops
    lam_5 = 14.30523 at ppl=10. The fix for both used to be "raise ppl and hope"."""
    from benchmarks.suite.domains import SUITE
    dom = SUITE[name].build()
    evp = Eigenproblem(dom, precision=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        truth = np.asarray(evp.solve(k, ppl=20, weyl_audit=False), dtype=float)
        without = np.asarray(evp.solve(k, ppl=bad_ppl, weyl_audit=False), dtype=float)
        with_audit = np.asarray(evp.solve(k, ppl=bad_ppl, weyl_audit=True), dtype=float)
    assert not np.allclose(without, truth, rtol=1e-6)      # the documented failure still bites
    assert np.allclose(with_audit, truth, rtol=1e-6)       # and the audit repairs it


@pytest.mark.slow
def test_the_audit_leaves_a_correct_solve_alone():
    """A false positive costs a re-scan that finds nothing; it must never change the answer."""
    from benchmarks.suite.domains import SUITE
    dom = SUITE['right_trapezoid'].build()
    evp = Eigenproblem(dom, precision=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        off = np.asarray(evp.solve(8, ppl=20, weyl_audit=False), dtype=float)
        on = np.asarray(evp.solve(8, ppl=20, weyl_audit=True), dtype=float)
    assert np.allclose(off, on, rtol=1e-12)


@pytest.mark.slow
def test_solve_interval_can_report_the_brackets_it_REJECTED():
    """A mode whose tension minimum misses `ttol` is dropped, and the survivors can then look
    perfectly healthy. `return_rejected` is how a caller learns it happened."""
    from lappy.mps import MPSEigensolver
    from lappy.asymp import weyl_est
    dom = unit_square()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        solver = MPSEigensolver.from_domain(dom, lam_max=weyl_est(6, dom), prec=1e-13)
        eigs, mults, fe, rejected = solver.solve_interval(1.0, weyl_est(6, dom), 120,
                                                          return_rejected=True)
        assert len(eigs) >= 3 and rejected == []           # nothing rejected on a clean square
        # with an impossible tension threshold EVERY bracket is rejected, and each one is named
        _, _, _, all_rejected = solver.solve_interval(1.0, weyl_est(6, dom), 120, ttol=1e-30,
                                                      return_rejected=True)
    assert len(all_rejected) >= 3
    assert all(lam > 0 and sig > 0 for lam, sig in all_rejected)
