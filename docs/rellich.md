# Overview
This describes the approach which will replace lappy's use of interior cubature formulas for normalizing and
orthogonalizing eigenfunctions. It is based on the Rellich identity for $L^2(\Omega)$ inner products and norms
of eigenfunctions, which uses boundary integrals only.

# Master identity
Consider a planar region $\Omega$ with boundary $\partial \Omega$. Suppose $-\Delta u = \lambda u$ and 
$-\Delta v = \lambda v$ on $\Omega$. Let $x_0 \in \mathbb{R}^2$ be any point, and let $r(x) = x - x_0$. Let $\mathbf{N}$ and $\mathbf{T}$ denote
the unit outward normal and tangent vectors to the boundary (oriented counter-clockwise). Let $\partial_\mathbf{N} w$ and
$\partial_\mathbf{T} w$ denote the derivatives of $w$ along $\partial \Omega$ in the normal and tangent directions, respectively.
The master identity is
$$\int_{\Omega} u\,v \;dA = \frac{1}{2\lambda}\oint_{\partial \Omega}(r \cdot \mathbf{N})\left[\partial_\mathbf{N} u\,\partial_\mathbf{N}v - \partial_\mathbf{T} u\,\partial_\mathbf{T} v + \lambda uv\right]\;ds + \frac{1}{2\lambda}\oint_{\partial \Omega}(r\cdot \mathbf{T})\left[\partial_\mathbf{T} u\,\partial_\mathbf{N} v + \partial_\mathbf{N} u\,\partial_\mathbf{T} v\right]\;ds.$$

That is, the $L^2(\Omega)$ inner product of $u,v$ can be expressed purely in terms of boundary integrals involving $u$,
$v$, and their first-order derivatives (i.e. Cauchy data).

# Applying boundary conditions
Suppose we enforce a Zaremba boundary condition, with a Dirichlet condition on $\Gamma_D$ and a Neumann condition on
$\Gamma_N$, such that $\Gamma_D \cup \Gamma_N = \partial \Omega$ and $\Gamma_D \cap \Gamma_N = \emptyset$. In this case,
the identity above simplifies to
$$\int_{\Omega} u\,v \;dA = \frac{1}{2\lambda}\oint_{\Gamma_D}(r \cdot \mathbf{N})\partial_\mathbf{N} u\,\partial_\mathbf{N}v \; ds + \frac{1}{2\lambda}\oint_{\Gamma_N}(r \cdot \mathbf{N})\left[\lambda u v - \partial_\mathbf{T} u\,\partial_\mathbf{T}v\right]\;ds.$$

The case of pure Dirichlet or pure Neumann boundary conditions each follow simply from this. It's useful to rewrite the
second integral above into two, giving the form
$$\begin{align*}
    \int_{\Omega} u\,v \;dA &= \frac{1}{2\lambda}\left(I_1 - I_2\right) + \frac{1}{2}I_3 \\
    I_1 &= \oint_{\Gamma_D}(r \cdot \mathbf{N})\partial_\mathbf{N} u\,\partial_\mathbf{N}v \; ds \\
    I_2 &= \oint_{\Gamma_N}(r \cdot \mathbf{N})\partial_\mathbf{T} u\,\partial_\mathbf{T}v\;ds \\
    I_3 &= \oint_{\Gamma_N}(r \cdot \mathbf{N})u v\;ds.
\end{align*}$$
This proves useful in evaluating these integrals using quadrature rules, as the integrands of $I_1$ and $I_2$ have
the same corner singularity behavior, of degree two less than the integrand of $I_3$. On the $jth$ boundary segment,
adjacent to corners with interior angles $\phi_j = \pi/\alpha_j$ and $\phi_{j+1} = \pi/\alpha_{j+1}$, $I_3$ can be evaluated using modified
Gauss-Jacobi rules with singular exponents $2\alpha,2\beta$. $I_1$ and $I_2$ then correspond to a rule with exponents
$2\alpha-2, 2\beta-2$.

# Goals for lappy
The goal here is a modification to lappy, focused around MPSEigensolver, which uses Gauss-Jacobi numerical integration
to compute the coefficient vectors for an orthonormal eigenbasis corresponding to the specified eigenvalue. Obviously,
in the case of a simple eigenvalue, one only needs to compute the normalizing constant from the eigenfunction 
$L^2(\Omega)$ norm. The multiple eigenvalue case can rely on building Gram matrices. For now, it is OK to leave the
interior cubature code in-place, but it will likely be removed in a future version.

This all suggests removing the support for Robin boundary conditions, which may be outside the scope of this work anyway.
For now, the eigensolver should opt-out of peforming orthonormalization for the Robin case, raising a warning in
relevant functions.