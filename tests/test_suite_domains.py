"""Checks on the curated benchmark domain suite (``benchmarks/suite/``).

Cheap: builds every domain and derives its geometry, but runs no eigensolves.
The point is that the suite's *claims* about its domains stay true as the
geometry code evolves.
"""
import numpy as np
import pytest

from benchmarks.suite import domains as D
from benchmarks.suite.features import TAGS, corner_data


def test_registry_validates():
    """Declared tags match derived geometry, and every field is well-formed."""
    problems = D.validate()
    assert not problems, '\n'.join(problems)


def test_core_spans_every_tag():
    covered = set().union(*(D.SUITE[k].tags for k in D.CORE))
    assert TAGS - covered == set(), f'CORE misses {sorted(TAGS - covered)}'


@pytest.mark.parametrize('key', sorted(D.SUITE))
def test_domain_builds(key):
    dom = D.SUITE[key].domain()
    assert dom.area > 0
    assert dom.perimeter > 0


@pytest.mark.parametrize('key,expected', [
    ('square', [2, 2, 2, 2]),                 # all regular
    ('reg_ngon_6', [1.5] * 6),                # all singular, mild
    ('L_shape', [2 / 3, 2, 2, 2, 2, 2]),      # one reentrant
    ('sector_reflex', [2 / 3, 2, 2]),         # reentrant, exactly solvable
    ('eq_tri', [3, 3, 3]),
])
def test_corner_exponents(key, expected):
    """p = pi/gamma is the suite's organizing quantity; pin it down."""
    p = np.sort(corner_data(D.SUITE[key].domain())[1])
    assert np.allclose(p, np.sort(expected))


def test_chevron_is_sharp_and_reentrant():
    p = np.sort(corner_data(D.SUITE['chevron_1_15'].domain())[1])
    assert np.isclose(p[0], 2 / 3)            # the reentrant corner
    # the two tip corners subtend arctan(h2) - arctan(h1) ~ 11.3 degrees
    gamma = np.arctan(1.5) - np.arctan(1.0)
    assert np.isclose(p[-1], np.pi / gamma)
    assert np.isclose(p[-2], np.pi / gamma)
    assert p[-1] > 15


@pytest.mark.parametrize('key', sorted(k for k, d in D.SUITE.items()
                                       if d.truth == 'analytic'))
def test_analytic_truth(key):
    d = D.SUITE[key]
    vals = np.asarray(d.truth_fn(d.n_eigs), dtype=float)
    assert vals.shape == (d.n_eigs,)
    assert np.all(np.isfinite(vals))
    assert vals[0] > 0
    assert np.all(np.diff(vals) >= 0)


def test_iso_tri_h1_is_the_right_isoceles_triangle():
    """Why iso_tri_h1 is allowed to claim analytic truth."""
    from lappy import geometry as G
    from lappy import reference as R
    assert np.isclose(G.iso_tri(1.0).area, G.iso_right_tri(np.sqrt(2)).area)
    assert np.allclose(D.SUITE['iso_tri_h1'].truth_fn(6),
                       R.iso_right_tri_eigs(6, np.sqrt(2)))


def test_production_view_shape():
    prod = D.for_reference_production()
    assert 'L_shape' in prod
    assert 'square' not in prod            # analytic, needs no production run
    assert 'chevron_1_125' not in prod     # status='open'
    for key, rec in prod.items():
        build, sym, n_basis, n_eigs = rec
        assert callable(build)
        assert sym is None or (isinstance(sym, tuple) and len(sym) == 2)
        assert n_basis > 0 and n_eigs > 0
