# Refactor: `FourierBesselBasis` cleanup and partial alignment with `FundamentalBasis`

## Background

`bases.py` contains two basis classes for the Helmholtz eigensolver:

- **`FourierBesselBasis`** — places basis functions at domain corners, using local polar coordinates with per-corner branch cuts and corner exponents `alpha = π / ω` where `ω` is the wedge opening angle.
- **`FundamentalBasis`** — places Bessel-Y multipole basis functions at source points outside the domain, using a precomputed column index map (`_src_idx`, `_m`, `_is_sin`) for fully vectorized evaluation.

`FundamentalBasis` has a cleaner internal design. This task brings `FourierBesselBasis` into alignment where it makes sense, while preserving the parts of its design that are genuinely superior.

## Problems to fix

### 1. Raw angle round-trip in `_set_alphak`

`_set_alphak` converts input angles to complex units (`_ray0`, `_ray1`), then immediately extracts raw angles back out via `np.angle` to canonicalize them into `[0, 2π)`. This round-trip is unnecessary and uses `% (2*np.pi)` semantics implicitly, which gives the wrong half-open interval.

The wedge angle `phi` and branch-cut offset `phi_hat` must lie in **(0, 2π]** (2π included, 0 excluded), because a value of exactly 0 would mean the branch cut or zero-ray coincides with `ray0` — a degenerate case that must be represented as `2π`, not `0`. The `%` operator maps `2π → 0` and so gives the wrong interval.

**Fix:** drop the raw angle storage entirely. Work exclusively with `_ray0`, `_ray1`, and `branch_rays` as complex units. Precompute `_phi_hat` (branch cut offset relative to `ray0`) once at construction using `+= 2*pi` on `<= 0`:

```python
def _set_alphak(self):
    self._ray0 = np.exp(1j * self._phi0)
    self._ray1 = np.exp(1j * self._phi1)

    phi = np.angle(self._ray1 / self._ray0)
    phi[phi <= 0] += 2 * np.pi                  # wedge angle in (0, 2π]

    alpha = np.pi / phi
    self.alphak = [alphai * np.arange(1, ki + 1)
                   for alphai, ki in zip(alpha, self.orders)]
    self.alphak_vec = np.concatenate(self.alphak)[np.newaxis]

    self._phi_hat = np.angle(self.branch_rays / self._ray0)
    self._phi_hat[self._phi_hat <= 0] += 2 * np.pi  # branch cut offset in (0, 2π]

    del self._phi0, self._phi1
```

### 2. `_phi_hat` recomputed on every `_theta` call

The branch-cut offset `phi_hat = angle(branch_rays / ray0)` depends only on the geometry, not on the evaluation points. Currently it is recomputed inside `_theta` on every call.

**Fix:** precompute `self._phi_hat` in `_set_alphak` (see above) and use it directly in `_theta`.

### 3. `% (2*np.pi)` in `on_boundary`

```python
phi0 = np.angle(tangents) % (2*np.pi)
phi1 = (phi0 + np.pi) % (2*np.pi)
```

`phi1 = (phi0 + np.pi) % (2*np.pi)` maps `phi0 = π` to `phi1 = 0`, losing the `2π` case.

**Fix:**
```python
phi0 = np.angle(tangents)
phi0[phi0 <= 0] += 2 * np.pi

phi1 = phi0 + np.pi
phi1[phi1 > 2 * np.pi] -= 2 * np.pi
```

### 4. `cumk` loops in `_sin` and `_cos`

```python
cumk = np.concatenate(([0], np.cumsum(self.orders)))
for i in range(self.n_sources):
    sin[:, cumk[i]:cumk[i+1]] = np.sin(np.outer(theta[:, i], self.alphak[i]))
```

This is a Python loop over sources that prevents full vectorization. The `ExPrecFBBasis` subclass compounds this with an additional loop over points, giving O(n_pts × n_basis) Python iterations.

**Fix:** build a column index map analogous to `FundamentalBasis._build_index_maps`:

```python
def _build_index_maps(self):
    src_indices, alphak_vals = [], []
    for i, (aks, order) in enumerate(zip(self.alphak, self.orders)):
        for ak in aks:
            src_indices.append(i)
            alphak_vals.append(ak)
    self._src_idx = np.array(src_indices, dtype=int)
    self._alphak_col = np.array(alphak_vals)   # shape (n_basis,)
```

Then `_sin` and `_cos` become single vectorized expressions:

```python
@instance_cache
def _sin(self, pts):
    theta = self._theta(pts)                          # (n_pts, n_sources)
    theta_cols = theta[:, self._src_idx]              # (n_pts, n_basis)
    return np.sin(theta_cols * self._alphak_col)

@instance_cache
def _cos(self, pts):
    theta = self._theta(pts)
    theta_cols = theta[:, self._src_idx]
    return np.cos(theta_cols * self._alphak_col)
```

`_r_rep` simplifies similarly:

```python
@instance_cache
def _r_rep(self, pts):
    return self._r(pts)[:, self._src_idx]
```

### 5. Merge `_sin`/`_cos` into `_angular`/`_angular_deriv` (optional)

Following `FundamentalBasis`, a single `_angular` method dispatching via `np.where` is cleaner than maintaining parallel `_sin` and `_cos` caches. This is lower priority than items 1–4 since the cached separation of sin/cos is harmless and makes `_grad_pointset` slightly more readable.

## What to preserve

- **Complex-form gradient** (`_dr_dz`, `_dtheta_dz`). The `FourierBesselBasis` gradient uses cached complex-valued partial derivatives and combines them as `dA_dr * dr_dz + dA_dtheta * dtheta_dz`. This is more elegant than `FundamentalBasis`'s explicit real `dx`/`dy` unpacking and should be kept (or backported to `FundamentalBasis`).
- **Separate `FourierBesselBasis` and `FundamentalBasis` classes.** They use different Bessel functions (`jv` vs `yv`), different angular structure (per-corner `alpha` exponents vs integer multipole orders), and different branch-cut logic. A unified base class would obscure more than it clarifies.
- **`ExPrecFBBasis` subclass structure.** The extended-precision path is correct in design; it just needs the `cumk` loops replaced with the index-map vectorization.

## Summary of changes

| Location | Change | Priority |
|---|---|---|
| `_set_alphak` | Remove raw angle round-trip; precompute `_phi_hat` | High |
| `_theta` | Use precomputed `_phi_hat`; fix `< 0` → `<= 0` | High |
| `on_boundary` | Replace `%` with explicit interval arithmetic | High |
| `_build_index_maps` | Add to `FourierBesselBasis` | Medium |
| `_sin`, `_cos`, `_r_rep` | Vectorize using `_src_idx` / `_alphak_col` | Medium |
| `ExPrecFBBasis._sin`, `_cos` | Replace Python loops with index-map ops | Medium |
| `_angular` / `_angular_deriv` | Optionally merge `_sin`/`_cos` | Low |
