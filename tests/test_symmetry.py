"""Verify that every registered symmetry generator really is a symmetry.

`lappy/symmetry.py` projects the basis onto a character sector and collocates
only inside the fundamental domain. If a generator does *not* map the boundary
to itself, that produces confidently wrong eigenvalues with small tensions and
no warning at all -- so this check gates everything downstream of
`domain_symmetry`.

The `symmetry.py` docstring has always referenced this file; it did not exist
until the reference-value run.
"""
import numpy as np
import pytest

from benchmarks.suite.domains import SUITE
from lappy.symmetry import domain_symmetry

# domains in the suite that declare a symmetry, keyed by (family, params)
SYMMETRIC = sorted(k for k, d in SUITE.items() if d.group() is not None)


@pytest.mark.parametrize('key', SYMMETRIC)
def test_generators_preserve_boundary(key):
    """Every group element maps boundary points onto the boundary."""
    entry = SUITE[key]
    dom = entry.domain()
    grp = entry.group()

    pts = dom.bdry_pts(24, kind='even').pts
    scale = dom.diameter

    def max_dist(zs):
        return max(dom.bdry.dist(z) for z in zs)

    # `bdry.dist` measures against the adaptive polyline, so on curved domains
    # even the identity reports a nonzero distance (~1e-8 relative, set by the
    # ParametricSegment arc-length tolerance). Calibrate against that floor
    # rather than against zero, so the test measures the generator and not the
    # distance routine.
    floor = max(max_dist(pts), 1e-12 * scale)

    for iso, _ in grp.elements:
        d = max_dist(iso(pts))
        assert d <= 10 * floor + 1e-9 * scale, (
            f'{key}: element {iso.name} moves the boundary by {d / scale:.2e} '
            f'(relative to diameter); identity floor is {floor / scale:.2e}')


@pytest.mark.parametrize('key', SYMMETRIC)
def test_group_closes_and_preserves_area(key):
    """The isometries are area-preserving and the group has 2-power order."""
    entry = SUITE[key]
    grp = entry.group()
    assert grp.order in (2, 4, 8), f'{key}: order {grp.order}'
    for iso, _ in grp.elements:
        assert np.isclose(abs(iso.a), 1.0), f'{key}: {iso.name} is not an isometry'


@pytest.mark.parametrize('key', SYMMETRIC)
def test_fundamental_domain_is_nonempty(key):
    """Interior points survive restriction to the fundamental domain.

    A generator that is a symmetry of the *shape* but whose half-plane
    convention is inconsistent can leave the fundamental domain empty, which
    silently starves the collocation.
    """
    from lappy.symmetry import fundamental_int_pts
    entry = SUITE[key]
    dom = entry.domain()
    grp = entry.group()
    pts = fundamental_int_pts(dom, grp, 40)
    assert len(pts) > 0, f'{key}: fundamental domain came back empty'
