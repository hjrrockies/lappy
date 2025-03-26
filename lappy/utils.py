import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import patches, colors
from shapely.geometry import Polygon
from shapely import points

def complex_form(pts,y=None):
    """Converts points in the plane to complex form"""
    pts = np.asarray(pts)
    if y is not None:
        return pts + 1j*y
    elif pts.ndim == 1:
        raise ValueError('pts must be 2-dimensional')
    elif pts.shape[1] == 2:
        return pts[:,0] + 1j*pts[:,1]
    else:
        return pts[0] + 1j*pts[1]

def real_form(pts):
    return np.array([pts.real,pts.imag]).T

def polygon_edges(vertices,y=None):
    """Computes the edges of a polygon with vertices (assumed in complex form), ordered counter-clockwise"""
    if y is not None:
        vertices = complex_form(vertices,y)
    return np.roll(vertices,-1)-vertices

def interior_angles(vertices,y=None):
    """Computes the interior angles of a polygon with given vertices (assumed in complex form x + 1j*y), ordered
    counter-clockwise"""
    if y is not None:
        vertices = complex_form(vertices,y)
    e = polygon_edges(vertices)
    phis = np.roll(np.angle(-e),1)-np.angle(e)
    phis[phis<0] += 2*np.pi
    return phis

def edge_lengths(vertices,y=None):
    """Computes the edge lengths of a polygon with vertices in complex form, ordered
    counter-clockwise"""
    if y is not None:
        vertices = complex_form(vertices,y)
    return np.abs(polygon_edges(vertices))

def side_normals(vertices,y=None):
    """Computes the outward-pointing unit normals to the sides of a
    polygon with vertices in complex form, ordered counter-clockwise"""
    if y is not None:
        vertices = complex_form(vertices,y)
    e = polygon_edges(vertices)
    return (e.imag-1j*e.real)/np.abs(e)

def edge_angles(vertices,y=None):
    """Gets the angle of each side of a polygon with the given vertices, with respect to the real axis."""
    if y is not None:
        vertices = complex_form(vertices,y)
    e = polygon_edges(vertices)
    psis = np.angle(e)
    return psis

def singular_corner_check(angles,tol=1e-15):
    """Checks to see if polygon corners are singular (not an integer fraction of pi)"""
    alpha = np.pi/angles
    maxalph = np.ceil(alpha.max())
    diffs = np.abs(np.subtract.outer(alpha,np.arange(1,maxalph+1)))
    return (diffs>tol).min(axis=1)

def edge_midpoints(vertices,y=None):
    if y is not None:
        vertices = complex_form(vertices,y)
    return 0.5*(np.roll(vertices,-1)+vertices)

def boundary_pts_polygon(vertices,n_pts=20,rule='legendre',skip=None):
    from .quad import boundary_nodes_polygon
    return boundary_nodes_polygon(vertices,n_pts,rule,skip)

def rand_interior_points(vertices,m,y=None,oversamp=10):
    """Computes random interior points for a polygon with vertices in complex form,
    ordered counter-clockwise."""
    if y is not None:
        vertices = complex_form(vertices,y)
    x_min, x_max = np.min(vertices.real), np.max(vertices.real)
    y_min, y_max = np.min(vertices.imag), np.max(vertices.imag)
    x_i = (x_max-x_min)*np.random.rand(oversamp*m)+x_min
    y_i = (y_max-y_min)*np.random.rand(oversamp*m)+y_min

    poly = Polygon(np.array([vertices.real,vertices.imag]).T)
    pts = points(np.array([x_i,y_i]).T)
    mask = poly.contains(pts)
    if mask.sum() < m:
        return rand_interior_points(m,vertices,oversamp=2*oversamp)
    return x_i[mask][:m] + 1j*y_i[mask][:m]

def plot_polygon(vertices,y=None,ax=None,**plotkwargs):
    """Plot a polygon with vertices in complex form, which are assumed to be in
    counter-clockwise order."""
    if ax is None: makeax = True
    else: makeax = False
    if y is not None:
        vertices = complex_form(vertices,y)
    v_ = np.append(vertices,vertices[0])
    if makeax:
        fig = plt.figure()
        ax = plt.gca()
    ax.plot(v_.real,v_.imag,**plotkwargs)
    if makeax:
        ax.set_aspect('equal')
        plt.show()

def plot_angles(vertices,y=None,ax=None):
    """Plots the angle arcs to confirm accurate calculation of angles. Also looks
    nice."""
    if y is not None:
        vertices = complex_form(vertices,y)
    phis = np.rad2deg(interior_angles(vertices))
    d = .5*np.minimum(*edge_lengths(vertices))
    start = np.rad2deg(edge_angles(vertices))
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    for i in range(len(phis)):
        arc = patches.Arc((vertices.real[i],vertices.imag[i]),d[i],d[i],angle=start[i],theta2=phis[i])
        ax.add_patch(arc)

def polygon_area(vertices,y=None):
    """Computes the area of a polygon with vertices in complex form,
    ordered counter-clockwise. Uses the Shoelace formula."""
    if y is not None:
        vertices = complex_form(vertices,y)
    x,y = vertices.real, vertices.imag
    return 0.5*np.sum((x-np.roll(x,-1))*(y+np.roll(y,-1)))

def polygon_perimeter(vertices,y=None,jac=False):
    if y is not None:
        vertices = complex_form(vertices,y)
    return np.sum(edge_lengths(vertices))

def reg_polygon(n,r):
    """Generates a regular polygon with n vertices and radius r."""
    theta = np.linspace(0,2*np.pi,n+1)
    return r*np.exp(1j*theta)[:-1]

def random_polygon(n,r_avg,eps,sigma):
    """Generate the vertices of a random polygon, with vertices ordered
    counter-clockwise."""
    u = np.random.rand(n)
    dtheta = (2*np.pi/n+eps)*u + (2*np.pi/n-eps)*(1-u)
    k = dtheta.sum()/(2*np.pi)
    theta = np.zeros(n)
    theta[0] = dtheta[0]/k
    for i in range(1,n):
        theta[i] = theta[i-1] + dtheta[i]/k
    r = sigma*np.random.randn(n)+r_avg
    r[r<0] = 0
    r[r>2*r_avg] = 2*r_avg
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x+1j*y

### Miscellaneous utils
def invert_permutation(p):
    """Invert a pertmutation vector. For use with column-pivoted QR solves"""
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s

### Eigenvalues of rectangles (for validation purposes)
def rect_eig(m,n,L,H):
    """Computes the (m,n) Dirichlet eigenvalue of an L-by-H rectangle"""
    return m**2*np.pi**2/L**2 + n**2*np.pi**2/H**2

def rect_eigs_k(k,L,H,ret_mn=False):
    """Gets first k Dirichlet eigenvalues for an LxH rectangle.
    Vectorized to handle arrays for L and H of various shapes, returning
    an array such that rect_eigs_k(L,H,k).flatten()[idx] is the first
    k eigenvalues of an L.flatten()[idx]xH.flatten()[idx] rectangle.

    Optionally returns the indices of the corresponding eigenvalues."""
    L,H = np.asarray(L), np.asarray(H)
    mn = np.arange(1,k+1)
    M,N = np.meshgrid(mn,mn,indexing='ij')
    eigs = rect_eig(M.flatten()[np.newaxis],
                    N.flatten()[np.newaxis],
                    L.flatten()[:,np.newaxis],
                    H.flatten()[:,np.newaxis])

    idx = np.argsort(eigs,axis=-1)
    eigs = np.take_along_axis(eigs,idx,axis=-1)[:,:k]
    eigs = eigs.reshape((*L.shape,k))
    if ret_mn:
        m = np.take_along_axis(M.flatten()[np.newaxis],idx,axis=-1)[:,:k]
        n = np.take_along_axis(N.flatten()[np.newaxis],idx,axis=-1)[:,:k]
        return eigs,m.reshape((*L.shape,k)),n.reshape((*L.shape,k))
    else:
        return eigs

def rect_eig_grad(m,n,L,H):
    """Derivatives of rectangular eigenvalues wrt length and width"""
    m,n = np.asarray(m), np.asarray(n)
    L,H = np.asarray(L), np.asarray(H)
    return (-2*(np.pi*m.T)**2/L**3).T, (-2*(np.pi*n.T)**2/H**3).T

def rect_eig_bound_idx(bound,L,H):
    """Identifies the indices of Dirichlet eigenvalues foran L-by-H rectangles
    which less than a given upper bound"""
    m_max = 1
    while True:
        eig = rect_eig(m_max,1,L,H)
        if eig > bound: break
        m_max += 1

    n_max = 1
    while True:
        eig = rect_eig(1,n_max,L,H)
        if eig > bound: break
        n_max += 1

    M = np.arange(1,m_max+1)[:,np.newaxis]
    N = np.arange(1,n_max+1)[np.newaxis]
    Lambda = rect_eig(M,N,L,H)
    return np.argwhere(Lambda <= bound)+1

def rect_eig_mult(lambda_,L,H,maxind=1000):
    """Compute the indices of rectangle Dirichlet eigenvalues which are
    duplicates of lambda_. For use in testing multiplicity."""
    Lam = rect_eig(np.arange(1,maxind)[np.newaxis],np.arange(1,maxind)[:,np.newaxis],L,H)
    diff = np.abs(lambda_-Lam)
    tot = (diff<1e-12).sum()
    ind = np.unravel_index(np.argsort(diff, axis=None), diff.shape)
    return (ind[0]+1)[:tot],(ind[1]+1)[:tot]

def rect_eig_mult_mn(m,n,L,H):
    """Compute the indices of rectangle Dirichlet eigenvalues which are
    duplicates of the (m,n) eigenvalue. For use in testing multiplicity."""
    return rect_eig_mult(rect_eig(m,n,L,H),L,H,maxind=10*max(m,n))

def loss_plot(L,H,loss,log=True,ax=None):
    n_levels = 50
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    if log:
        low = np.floor(np.log10(loss.min()+1e-16))
        high = np.ceil(np.log10(loss.max()))
        levels = 10.0**np.linspace(low,high,n_levels)
        norm = colors.LogNorm(vmin=loss.min()+1e-16, vmax=loss.max())
    else:
        levels = n_levels
        norm = 'linear'
    cs = ax.contourf(L,H,loss+1e-16,levels,norm=norm)
    cbar = ax.get_figure().colorbar(cs,ax=ax)
    if log: cbar.set_ticks(10.0**np.arange(low,(high+1)))
    ax.set_xlabel('L')
    ax.set_ylabel('H')
    return ax,cs

def eig_ratios(eigs):
    """Computes the ratios lambda_j/lambda_1, assuming the eigenvalues are
    indexed along the last dimension of the array"""
    return (eigs.T[1:]/eigs.T[0]).T

def rect_loss_std(L,H,eigs_true,weights=1,jac=False):
    """Computes the standard loss function for a rectangle against
    target eigenvalues. Can use weights."""
    weights = np.asarray(weights).flatten()
    k = len(eigs_true)
    eigs,m,n = rect_eigs_k(L,H,k,ret_mn=True)
    pweights = weights**(0.5)
    out = la.norm(pweights*(eigs-eigs_true),axis=-1)**2
    if jac:
        grad = weights*2*(eigs-eigs_true)
        deig_dL, deig_dH = rect_eig_grad(m,n,L,H)
        grad = np.array([(grad*deig_dL).sum(axis=-1), (grad*deig_dH).sum(axis=-1)])
        return out,grad
    else:
        return out

def rect_loss_reciprocal(L,H,eigs_true,weights=1):
    """Computes the standard loss function for a rectangle against
    target eigenvalues in reciprocal form. Can use weights."""
    weights = np.asarray(weights).flatten()
    k = len(eigs_true)
    eigs,m,n = rect_eigs_k(L,H,k,ret_mn=True)
    pweights = weights**(0.5)
    out = la.norm(pweights*(1/eigs-1/eigs_true),axis=-1)**2
    jac = False
    if jac:
        grad = weights*2*(eigs-eigs_true)
        deig_dL, deig_dH = rect_eig_grad(m,n,L,H)
        grad = np.array([(grad*deig_dL).sum(axis=-1), (grad*deig_dH).sum(axis=-1)])
        return out,grad
    else:
        return out

def rect_loss_outerlog(L,H,eigs_true,weights=1,eps=1e-200,jac=False):
    """Computes the 'outer-log loss' for rectangular eigenvalues"""
    L,H = np.asarray(L), np.asarray(H)
    weights = np.asarray(weights).flatten()
    k = len(eigs_true)
    if len(weights) == 1:
        weights = weights*np.ones(k)
    eigs,m,n = rect_eigs_k(L,H,k,ret_mn=True)
    eigsT = eigs.T
    eigs_trueT = eigs_true.reshape((-1,*np.ones(eigsT.ndim-1,dtype='int')))
    weightsT = weights.reshape((-1,*np.ones(eigsT.ndim-1,dtype='int')))
    out = weightsT[0]*np.log((eigsT[0]-eigs_trueT[0])**2+eps)
    eigdiff = eigsT[1:][np.newaxis]-eigs_trueT[1:][:,np.newaxis]
    out += (weightsT[1:]*np.log(eigdiff**2 + eps).sum(axis=0)).sum(axis=0)
    if jac:
        deig_dL, deig_dH = rect_eig_grad(m,n,L,H)
        C = np.empty((k,*L.shape))
        C[0] = (eigsT[0]-eigs_trueT[0])/((eigsT[0]-eigs_trueT[0])**2+eps)
        eigdiff_sum = eigdiff.sum(axis=0)
        C[1:] = eigdiff_sum/(eigdiff_sum**2+eps)
        C *= 2*weightsT
        grad = np.array([(C.T*deig_dL).sum(axis=-1),(C.T*deig_dH).sum(axis=-1)])
        return out,grad
    else:
        return out

def logify(obj):
    def log_obj(*args,**kwargs):
        if 'jac' in kwargs.keys():
            if kwargs['jac'] == True:
                jac = True
            else: jac = False
        else: jac = False
        if jac:
            out,grad = obj(*args,**kwargs)
            return np.log(out+1e-200), grad/(out+1e-200)
        else:
            return np.log(obj(*args,**kwargs)+1e-200)
    return log_obj

