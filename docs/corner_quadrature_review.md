# Review: "Corner-Adapted Quadrature for Rellich Normalization"

**Verdict:** Holds water. The central idea — substituting $t=r^\nu$ with $\nu=\pi/\alpha$ taken *exactly* from geometry, rather than a generic flattening map — is a genuinely sharper tool than the Kress sigmoidal transformation discussed previously, because the corner exponent is known in closed form rather than merely bounded. One formula in Section 3.1 has an algebra slip (numbers unaffected), and the abstract overclaims relative to the document's own tables. Details below.

---

## Confirmed correct

- **Corner expansion and $\partial_n u \sim r^{\nu-1}F(r)$ (Eqs. 1–6).** Matches the Dirichlet corner structure derived independently elsewhere: their $\nu_k = k\pi/\alpha$ is the same as $p_m=m\pi/\gamma$, their $A_\mu(z^2)$ is the same entire-function device as $E_p$ (the $J_p(z)=(z/2)^p\times\text{entire}(z^2)$ split), just different notation for the same fact.
- **Proposition 1** ($x_0$ at the corner $\Rightarrow r\cdot\mathbf N\equiv0$ on both adjacent edges). Correct, and a clean special case of a more general fact: $r\cdot\mathbf N$ is constant along any straight edge, equal to zero exactly when the origin lies on that edge's line. Putting $x_0$ at the corner puts it on *both* adjacent lines at once.
- **Lemma 1's mass-concentration table.** Recomputed independently from $(\varepsilon/R)^{2\nu-1}$; every entry matches (e.g. $\alpha=1.5\pi \Rightarrow 2\nu-1=1/3 \Rightarrow (10^{-9})^{1/3}=10^{-3}$, as stated).
- **Proposition 2's substitution algebra.** Rederived from scratch: $r^{2\nu-2}\,dr \to \tfrac1\nu t^{1-1/\nu}\,dt$ is exactly right.
- **Section 4's sensitivity analysis.** Mechanism is correct and worth flagging as important: a mismatched $\nu$ leaves a residual $r^{-2\delta}$ factor that destroys analyticity outright. Unlike a generic grading order (which just needs to be "large enough"), this exponent must be *exact* — no margin-padding is safe.
- **The $3\pi/2$ cancellation trap.** Verified directly: $r\cdot\mathbf N$ on the two edges of a $3\pi/2$ corner is $\mathrm{Im}(x_0)$ and $-\mathrm{Re}(x_0)$. So it isn't that each edge vanishes on the diagonal $\mathrm{Re}(x_0)=\mathrm{Im}(x_0)$ — it's that the *sum* vanishes identically whenever the two edges' $(\partial_n u)^2$ contributions happen to match, which is exactly what a symmetric synthetic test function tends to produce. Correctly reasoned, easy trap to fall into.

## Error to fix: the $2/\nu \Leftrightarrow \alpha$ relationship

The claimed equivalence

> "$2/\nu\in\mathbb Z$, equivalently $\alpha = 2\pi/m$"

is wrong. Since $2/\nu = 2\alpha/\pi$, the correct statement is
$$\alpha = \frac{m\pi}{2} \qquad\Longleftrightarrow\qquad m = \frac{2\alpha}{\pi} = \frac{2}{\nu}.$$

Checking their own worked example: $\alpha=3\pi/2$ gives $2/\nu=3$ (matches the text) — but $\alpha=2\pi/m$ would require $m=2\pi/(3\pi/2)=4/3$, not an integer. The corrected formula gives $m=3$ immediately, consistent with the text.

**The numbers are unaffected** — every row of the Section 5(c) table ($1.6\pi\to3.200$, $1.7\pi\to3.400$, $1.85\pi\to3.700$, $1.95\pi\to3.900$) matches $m=2\alpha/\pi$ exactly, so this looks like a pure transcription slip in the prose rather than an error in the underlying code/numerics. Still worth fixing before anyone uses the stated formula to precompute which corners are "special" in a multi-corner domain — as written, it would misidentify every case.

**Suggested fix:**
```diff
- The substitution rationalizes the corner family completely, but maps the Bessel
- family r^{2q} to t^{2q/ν}, which is integral only when 2/ν ∈ ℤ — equivalently
- α = 2π/m for an integer m.
+ The substitution rationalizes the corner family completely, but maps the Bessel
+ family r^{2q} to t^{2q/ν}, which is integral only when 2/ν ∈ ℤ — equivalently
+ α = mπ/2 for an integer m (since 2/ν = 2α/π).
```

## Overclaim in the abstract

The abstract states the rule "integrates it to machine precision with $O(10)$ nodes." This is only true in the special $2/\nu\in\mathbb Z$ case. The body text is careful and correctly hedged — Section 3.1 and Table 5(c) explicitly show generic angles reaching only $\sim10^{-8}$–$10^{-9}$ at $n=8$, converging at "roughly two digits per node doubling," and hitting machine precision only around $n=32$–$64$. The abstract should reflect that hedge rather than state the special case as the general result.

**Suggested fix:**
```diff
- (iii) that a corner-local Gauss–Jacobi rule in the variable t = r^ν integrates
- it to machine precision with O(10) nodes, removing the constraint on x0 altogether.
+ (iii) that a corner-local Gauss–Jacobi rule in the variable t = r^ν integrates it
+ to machine precision with O(10) nodes when 2/ν ∈ ℤ, and otherwise at high-order
+ algebraic (≈2 digits per node doubling, reaching 1e-12 or better by n≈32),
+ removing the constraint on x0 altogether in either case.
```

## Scope notes (not errors — worth flagging for follow-on work)

- **Norm-only, so far.** The document treats only $u=v$ with weight $r\cdot\mathbf N$ at a Dirichlet corner. The same $t=r^\nu$ substitution should extend to the bilinear Gram-matrix kernels ($K^{NN},K^{TT},K^{\text{cr}}$) and to Hadamard's $V$-weighted integrals — the leading exponent is purely geometric and shared by any two eigenfunctions at that corner, and any polygon/spline weight has known local Taylor behavior to fold in the same way. This isn't demonstrated here, only plausible by analogy.
- **Doesn't address Cauchy-data evaluation stability.** The "black box" $\partial_n u$ near the corner is trusted as given. If the underlying evaluator has the same naive-Bessel-division cancellation risk flagged elsewhere (evaluating $J_p(z)$ directly and dividing by $(z/2)^p$ rather than using the entire-function representation directly), this quadrature fix and that evaluation fix are complementary, not substitutes.
- **Implementation detail to double check:** Section 6's recipe integrates $\int r^{2\nu-2}G(r)\,dr$, but the actual edge contribution is $\rho=r_N$ times that (Eq. 7). The multiplication by $r_N$ is stated earlier but not restated in the final three-step recipe — worth confirming it isn't dropped in translation to code.
- **Numbers not independently reproduced:** the specific `mpmath.quad` failure magnitudes ($-4.7\times10^{-5}$, $-4.3\times10^{-2}$) and the `KRESS_TAU_FLOOR = 1e-9` implementation detail are mechanistically plausible and consistent with everything else in the document, but weren't run independently here.

## Bottom line

Mathematically sound, more precise than the generic Kress-map approach for this specific problem (because $\nu$ is known exactly), and honestly self-critical in its body text about where it does and doesn't reach machine precision. Fix the $2/\nu\leftrightarrow\alpha$ formula and tighten the abstract's claim, and this is ready to fold into the larger framework as the preferred corner treatment whenever the leading exponent is known exactly — which, for this MPS setting, is always.
