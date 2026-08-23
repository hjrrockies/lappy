"""`interior_l2`'s cubature cache must not return one domain's mesh for another.

The cache is keyed on `id(domain)`, which is a CPython object address and is reused after a
garbage collection. A sweep that builds one `Domain` per iteration and drops it -- which is what
every benchmark loop in this repo does -- can therefore be handed a *previous* domain's cubature
points, silently, because they are a perfectly valid `PointSet` either way. The corrupted `||u||`
then goes straight into the denominator of Moler--Payne's `eps`.

This was not hypothetical. 10 of 150 cells in `benchmarks/basis_lab/run/plan/s3.jsonl` were wrong
before the fix, by up to 6.7 digits: `chevron_1_2` reported 14.94 certified digits where its own
boundary residual said 8.23. Two headline findings were drawn from those cells and had to be
retracted. The errors ran in both directions, so they could not be excused as conservative.

The fix is that each cache entry holds a strong reference to its domain, so the address cannot be
recycled while the entry lives.
"""
import gc

import numpy as np
import pytest

import benchmarks.reference.certify as C
from lappy import geometry as geo


@pytest.fixture(autouse=True)
def _clean_cache():
    C.clear_mesh_cache()
    yield
    C.clear_mesh_cache()


def _l2_of_one(domain):
    """`||1||_L2(Omega) = sqrt(area)`, computed through the cached-mesh path."""
    norm, method = C.interior_l2(domain, lambda pts: np.ones(len(pts.pts)), deg=4)
    return norm, method


def test_the_norm_is_right_after_many_domains_have_been_built_and_dropped():
    """The regression proper. `||1|| = sqrt(area)` is domain-specific, so a recycled cache entry
    shows up immediately as the wrong area."""
    for i in range(120):
        d = geo.rect(1.0 + 0.01*i, 1.0)
        _l2_of_one(d)
        del d
        gc.collect()

    target = geo.rect(3.0, 2.0)
    norm, method = _l2_of_one(target)
    assert method.startswith('mesh'), method
    assert norm == pytest.approx(np.sqrt(6.0), rel=1e-10), (norm, np.sqrt(6.0))


def test_each_distinct_domain_gets_its_own_entry():
    doms = [geo.rect(1.0 + 0.01*i, 1.0) for i in range(50)]
    for d in doms:
        _l2_of_one(d)
    assert len(C._MESH_CACHE) == len(doms)


def test_cache_entries_pin_their_domain():
    """The invariant that makes `id()` keying safe."""
    d = geo.rect(2.0, 1.0)
    _l2_of_one(d)
    (entry,) = list(C._MESH_CACHE.values())
    assert entry[0] is d, 'the cache must hold the domain alive'


def test_two_domains_with_identical_area_and_perimeter_do_not_collide():
    """Why a geometric fingerprint is NOT an acceptable alternative key: the GWW pair is
    isospectral, so equal area, equal perimeter and the same segment count -- and different
    domains. Any fingerprint built from those would collide."""
    a, b = geo.GWW1(), geo.GWW2()
    assert a.area == pytest.approx(b.area)
    assert a.perimeter == pytest.approx(b.perimeter)
    na, _ = _l2_of_one(a)
    nb, _ = _l2_of_one(b)
    assert len(C._MESH_CACHE) == 2
    assert na == pytest.approx(np.sqrt(a.area), rel=1e-6)
    assert nb == pytest.approx(np.sqrt(b.area), rel=1e-6)


def test_clear_releases_the_entries():
    _l2_of_one(geo.rect(1.0, 1.0))
    assert C.clear_mesh_cache() == 1
    assert not C._MESH_CACHE
