import numpy as np
import scipy.linalg as la

# convert between real and complex representations of points in the plane
def complex_form(pts, y=None):
    """Converts points in the plane to complex form"""
    pts = np.asarray(pts)
    if y is not None and np.isrealobj(pts):
        y = np.asarray(y)
        if y.dtype != 'float64':
            raise ValueError('y must be a real-valued array')
        return pts + 1j*y
    elif np.iscomplexobj(pts):
        return pts
    elif pts.ndim <= 1:
        return pts.astype('complex128')
    elif pts.shape[-1] == 2 and np.isrealobj(pts):
        return np.transpose(pts.T[0] + 1j*pts.T[1])
    else:
        raise ValueError('pts must be a real array of shape (...,2)')

def real_form(pts):
    pts = np.asarray(pts)
    if np.iscomplexobj(pts):
        return np.array([pts.real,pts.imag]).T
    elif pts.shape[-1] == 2:
        return pts
    else:
        raise ValueError('pts must be a complex array, or already in real form with shape (...,2)')

# polygon common functions
def polygon_edges(vertices):
    """Computes the edges of a polygon with vertices, ordered counter-clockwise"""
    vertices = complex_form(vertices)
    return np.roll(vertices,-1)-vertices

def interior_angles(vertices):
    """Computes the interior angles of a polygon with given vertices, ordered
    counter-clockwise"""
    e = polygon_edges(vertices)
    phis = np.angle(np.roll(-e,1)/e)
    phis[phis<0] += 2*np.pi
    return phis

def edge_lengths(vertices):
    """Computes the edge lengths of a polygon with vertices in complex form, ordered
    counter-clockwise"""
    vertices = complex_form(vertices)
    return np.abs(polygon_edges(vertices))

def side_normals(vertices):
    """Computes the outward-pointing unit normals to the sides of a
    polygon with vertices in complex form, ordered counter-clockwise"""
    vertices = complex_form(vertices)
    e = polygon_edges(vertices)
    return (e.imag-1j*e.real)/np.abs(e)

def edge_angles(vertices):
    """Gets the angle of each side of a polygon with the given vertices, with respect to the real axis."""
    vertices = complex_form(vertices)
    e = polygon_edges(vertices)
    psis = np.angle(e)
    return psis

def singular_corner_check(angles,tol=1e-15):
    """Checks to see if polygon corners are singular (not an integer fraction of pi)"""
    alpha = np.pi/angles
    maxalph = np.ceil(alpha.max())
    diffs = np.abs(np.subtract.outer(alpha,np.arange(1,maxalph+1)))
    return (diffs>tol).min(axis=1)

def edge_midpoints(vertices):
    vertices = complex_form(vertices)
    return 0.5*(np.roll(vertices,-1)+vertices)

def segment_intersection(a0, a1, b0, b1):
    """Computes the intersection of two line segments, if it exists."""
    # segment vectors
    d1 = a1 - a0
    d2 = b1 - b0
    
    # Cross product in 2D: d1 × d2
    cross = d1.real * d2.imag - d1.imag * d2.real
    
    # Solve for parameters s and t
    delta = b0 - a0
    if cross == 0:
        s,t = np.inf, np.inf
    else:
        s = (delta.real * d2.imag - delta.imag * d2.real) / cross
        t = (delta.real * d1.imag - delta.imag * d1.real) / cross
    
    # Check if intersection is within both segments
    if 0 <= s <= 1 and 0 <= t <= 1:
        return a0 + s*d1
    
    else: return None

def boundary_pts_polygon(vertices,n_pts=20,rule='legendre',skip=None):
    from .quad import boundary_nodes_polygon
    return boundary_nodes_polygon(vertices,n_pts,rule,skip)

def rand_interior_points(vertices, m, oversamp=2):
    """Computes random interior points for a polygon with vertices in complex form,
    ordered counter-clockwise."""
    from shapely.geometry import Polygon
    from shapely import points
    vertices = complex_form(vertices)
    poly_area = polygon_area(vertices)
    x_min, x_max = np.min(vertices.real), np.max(vertices.real)
    y_min, y_max = np.min(vertices.imag), np.max(vertices.imag)
    box_area = (x_max-x_min)*(y_max-y_min)
    npts = int(np.ceil(m*oversamp*box_area/poly_area))
    x_i = (x_max-x_min)*np.random.rand(npts)+x_min
    y_i = (y_max-y_min)*np.random.rand(npts)+y_min

    poly = Polygon(np.array([vertices.real,vertices.imag]).T)
    pts = points(np.array([x_i,y_i]).T)
    mask = poly.contains(pts)
    if mask.sum() < m:
        return rand_interior_points(vertices, m, oversamp=2*oversamp)
    return x_i[mask][:m] + 1j*y_i[mask][:m]

def plot_polygon(vertices,ax=None,**plotkwargs):
    """Plot a polygon with vertices in complex form, which are assumed to be in
    counter-clockwise order."""
    import matplotlib.pyplot as plt
    if ax is None: makeax = True
    else: makeax = False
    vertices = complex_form(vertices)
    v_ = np.append(vertices,vertices[0])
    if makeax:
        fig = plt.figure()
        ax = plt.gca()
    ax.plot(v_.real,v_.imag,**plotkwargs)
    if makeax:
        ax.set_aspect('equal')
        plt.show()

def plot_angles(vertices,ax=None):
    """Plots the angle arcs to confirm accurate calculation of angles. Also looks
    nice."""
    import matplotlib.pyplot as plt
    from matplotlib import patches
    vertices = complex_form(vertices)
    phis = np.rad2deg(interior_angles(vertices))
    l = edge_lengths(vertices)
    d = .5*np.minimum(l,np.roll(l,-1))
    start = np.rad2deg(edge_angles(vertices))
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    for i in range(len(phis)):
        arc = patches.Arc((vertices.real[i],vertices.imag[i]),d[i],d[i],angle=start[i],theta2=phis[i])
        ax.add_patch(arc)

def polygon_area(vertices):
    """Computes the area of a polygon with vertices in complex form,
    ordered counter-clockwise. Uses the Shoelace formula."""
    vertices = complex_form(vertices)
    x,y = vertices.real, vertices.imag
    return 0.5*np.sum((x-np.roll(x,-1))*(y+np.roll(y,-1)))

def polygon_perimeter(vertices):
    vertices = complex_form(vertices)
    return np.sum(edge_lengths(vertices))

def polygon_diameter(vertices):
    vertices = complex_form(vertices)
    return np.max(np.abs(np.subtract.outer(vertices,vertices)))

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

from .exact import rect_eig, rect_eigs, rect_eig_grad

def loss_plot(L,H,loss,log=True,ax=None):
    import matplotlib.pyplot as plt
    from matplotlib import colors
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
    eigs,m,n = rect_eigs(k,L,H,ret_mn=True)
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
    eigs,m,n = rect_eigs(k,L,H,ret_mn=True)
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
    eigs,m,n = rect_eigs(k,L,H,ret_mn=True)
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

