import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import patches, cm, ticker, colors
from shapely.geometry import Polygon
from shapely import points

def complex_form(x,y):
    """Converts points on the plane to complex form"""
    return x + 1j*y

def interior_angles(vertices,y=None):
    """Computes the interior angles of a polygon with given vertices (assumed in complex form x + 1j*y), ordered
    counter-clockwise"""
    if y is not None:
        vertices = complex_form(vertices,y)
    psis = edge_angles(vertices)
    phis = psis-np.roll(psis,1)
    phis[phis<0] += 2*np.pi
    return phis

def polygon_edges(vertices,y=None):
    """Computes the edges of a polygon with vertices (assumed in complex form), ordered counter-clockwise"""
    if y is not None:
        vertices = complex_form(vertices,y)
    return np.roll(vertices,-1)-vertices

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

def edge_midpoints(vertices,y=None):
    if y is not None:
        vertices = complex_form(vertices,y)
    return 0.5*(np.roll(vertices,-1)+vertices)

def boundary_points(x,y,m,method='even',skip=None):
    """Generates points along the boundary of a polygon with vertices x and y,
    ordered counter-clockwise. Makes m points for each side."""
    raise NotImplementedError('Needs to be updated for complex arithmetic')
    mask = np.ones(len(x),dtype=bool)
    if skip is not None:
        mask[skip] = 0
    if method == 'even':
        x_b = np.linspace(x,np.roll(x,-1),m+2)[1:-1,mask].flatten(order='F')
        y_b = np.linspace(y,np.roll(y,-1),m+2)[1:-1,mask].flatten(order='F')
    elif method == 'legguass':
        raise(NotImplementedError)
    return x_b,y_b

def interior_points(m,vertices,y=None,oversamp=10):
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
        return interior_points(m,vertices,oversamp=2*oversamp)
    return x_i[mask][:m] + 1j*y_i[mask][:m]

def plot_polygon(vertices,y=None,ax=None,**plotkwargs):
    """Plot a polygon with vertices in complex form, which are assumed to be in
    counter-clockwise order."""
    if y is not None:
        vertices = complex_form(vertices,y)
    v_ = np.append(vertices,vertices[0])
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    ax.plot(v_.real,v_.imag,**plotkwargs)
    if ax is None:
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

def polygon_perimiter(vertices,y=None):
    if y is not None:
        vertices = complex_form(vertices,y)
    return np.sum(edge_lengths(vertices))

def reg_polygon(r,n):
    """Generates a regular polygon with n vertices and radius r."""
    theta = np.linspace(0,2*np.pi,n+1)
    return r*np.exp(1j*theta)[:-1]

def edge_indices(points,vertices):
    """Takes an array of points and identifies which polygon edge the points lie on,
    or if they are not on an edge."""
    raise NotImplementedError('Needs to be updated for complex arithmetic')
    x,y = points[:,0],points[:,1]
    n = len(vertices)
    arr = np.full(len(points),n,dtype='int')
    for i in range(n):
        j = i+1
        if j==n: j=0
        u,v = vertices[i],vertices[j]
        mask = np.isclose(np.sqrt((x-u[0])**2+(y-u[1])**2)+np.sqrt((x-v[0])**2+(y-v[1])**2),
                        np.sqrt((u[0]-v[0])**2+(u[1]-v[1])**2))
        arr[mask] = i
    return arr

def rect_eig(m,n,L,H):
    """Computes the (m,n) Dirichlet eigenvalue of an L-by-H rectangle"""
    return m**2*np.pi**2/L**2 + n**2*np.pi**2/H**2

def rect_eigs_k(L,H,k,ret_mn=False):
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

def rect_eig_grad(m,n,L,H):
    """Gradients of *simple* rectangular eigenvalues with respect to rectangle
    vertices. Used to test derivative estimation code."""
    m2L3 = m**2/L**3
    n2H3 = n**2/H**3
    return (np.pi**2)*np.array([m2L3,-m2L3,-m2L3,m2L3,n2H3,n2H3,-n2H3,-n2H3])

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

def invert_permutation(p):
    """Invert a pertmutation vector. For use with column-pivoted QR solves"""
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s

rho = (3-5**0.5)/2
def golden_search(f,a,b,tol=1e-14,maxiter=100):
    """Golden ratio minimization search"""
    h = b-a
    u, v = a+rho*h, b-rho*h
    fu, fv = f(u), f(v)
    i = 0
    while (b-a>=tol)&(i<=maxiter):
        i += 1
        if fu < fv:
            b = v
            h = b-a
            v = u
            u = a+rho*h
            fv = fu
            fu = f(u)
        else:
            a = u
            h = b-a
            u = v
            v = b-rho*h
            fu = fv
            fv = f(v)
    if f(a)<f(b): return a,i
    else: return b,i

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

def discrete_local_min_idx(x,y):
    x = x.flatten()
    sort_idx = x.argsort()
    x = x[sort_idx]
    y = y[sort_idx]

    min_idx = np.argwhere((y[1:-1]<y[:-2])*(y[1:-1]<y[2:]))
    return min_idx+1

def parabola_vertex(x,y):
    """Finds the vertex of a parabola passing through the points
    {(x[0],y[0]),(x[1],y[1]),(x[2],y[2])}"""
    dx0,dx1,dx2 = x[1]-x[0],x[2]-x[1],x[2]-x[0]
    dy0,dy1,dy2 = y[1]-y[0],y[2]-y[1],y[2]-y[0]

    C = (dx0*dy1 - dx1*dy0)/(dx0*dx1*dx2)
    if C<=0 or np.isnan(C): vertex = x[np.argmin(y)]
    else: vertex = (x[1]+x[2]-dy1/(dx1*C))/2
    return vertex

def parabolic_iter_min(f,x,y,xtol=1e-8,maxiter=10,maxresc=2,resc_param=0.1,verbose=False):
    """Finds a local minimum of f in the interval [x[0],x[2]] by repeated parabolic interpolation."""
    if x[1]<=x[0] or x[2]<=x[1]:
        raise ValueError('x not increasing!')

    # shift left endpoint to zero for maximum precisioon
    z = x-x[0]

    # shifted interval bounds
    zlo, zhi = z[0],z[2]

    # function evaluation & rescue counters
    fevals = 0
    rescues = 0

    # previous vertex
    vold = np.nan

    if verbose: print(f'parabolic_iter_min on ({x[0]:.3e},{x[1]:.3e},{x[2]:.3e})')
    # iterative parabolic interpolation
    for i in range(maxiter):
        v = parabola_vertex(z,y)

        # if the vertex hasn't changed much, conclude iteration
        if np.abs(v-vold)<xtol:
            break

        # vertex falls off left, rescue if possible
        elif v<zlo:
            if verbose: print('fell off left')
            if rescues < maxresc:
                rescues += 1
                v = (1-resc_param)*zlo + resc_param*z[1]
            else:
                if verbose: print('too many rescues')
                return None, fevals
        # vertex falls off right, rescue if possible
        elif v>zhi:
            if verbose: print('fell off right')
            if rescues < maxresc:
                rescues += 1
                v = (1-resc_param)*zhi + resc_param*z[1]
            else:
                if verbose: print('too many rescues')
                return None, fevals
        # if vertex too close to old points, wiggle it
        if np.abs(z-v).min() < xtol/2: v += xtol

        # interpolate using new vertex and adjacent points in z
        vold = v
        yv = f(v+x[0])
        fevals += 1
        if v<z[1]:
            z = np.array([z[0],v,z[1]])
            y = np.array([y[0],yv,y[1]])
        elif z[1]<v:
            z = np.array([z[1],v,z[2]])
            y = np.array([y[1],yv,y[2]])
        if verbose: print(f'z={np.array_str(z,precision=3)}')

    # return final vertex, shifted back to original interval
    if verbose: print(f'parabolic_iter_min concluded, x_min={v+x[0]}')
    return v+x[0], fevals

def minsearch(f,start,end,h,xtol=1e-8,verbose=False,maxdepth=10):
    """Finds all minima in the interval [start,end]"""
    spaces = (10-maxdepth)*('  ')
    x0,x1 = start, start+h
    y0,y1 = f(x0), f(x1)
    fevals = 2
    n_intervals = int(np.ceil((end-start)/h))
    minima = []
    if h<= xtol:
        maxdepth = 0
    if verbose:
        print(spaces+f'minsearch on [{start:.3e},{end:.3e}]')
        print(spaces+f'h={h}, n_intervals={n_intervals}')
    for i in range(n_intervals-2):
        x2 = x1 + h
        y2 = f(x2)
        fevals += 1
        x = np.array([x0,x1,x2])
        y = np.vstack((y0,y1,y2)).T
        if verbose:
            print(spaces+f'x={np.array_str(x,precision=3)}')
            print(spaces+f'sigma1={np.array_str(y[0],precision=3)}')
            print(spaces+f'sigma2={np.array_str(y[1],precision=3)}')

        # catch discrete local min
        if (y1[0]<y0[0] and y1[0]<y2[0]):
            if verbose: print(spaces+f'discrete local min at x={x1:.3e}')

            # catch small second subspace angle, recurse on finer grid
            if np.min(y[1]) < 1.1*h and maxdepth > 0:
                if verbose:
                    print(spaces+f'sigma2 = {np.min(y[1]):.3e}<1.1*h={1.1*h:.3e}')
                    print('recursing on finer grid...')
                mins, fe = minsearch(f,x0-h/2,x2+h/2,h/2,xtol=xtol,verbose=verbose,maxdepth=maxdepth-1)
                minima += [x for x in mins if (x>=x0) and (x<=x2)]
                fevals += fe
            else:
                min, fe = parabolic_iter_min(lambda x: f(x)[0]**2,x,y[0]**2,xtol=xtol,verbose=verbose)
                if min is not None: minima.append(min)
                fevals += fe

        x0,x1 = x1,x2
        y0,y1 = y1,y2
    if verbose:
        print(spaces+f"minsearch on [{start:.3e},{end:.3e}] concluded")
        print(spaces+f"found minima: {minima}")
    return minima, fevals

### Rewrite of utils for complex arithmetic ###
if __name__ == "__main__":
    def complex_form(x,y):
        """Converts points on the plane to complex form"""
        return x + 1j*y

    def interior_angles(vertices,y=None):
        """Computes the interior angles of a polygon with given vertices (assumed in complex form x + 1j*y), ordered
        counter-clockwise"""
        if y is not None:
            vertices = complex_form(vertices,y)
        psis = edge_angles(vertices)
        phis = edge_angles-np.roll(edge_angles,1)
        phis[phis<0] += 2*np.pi
        return phis
    
    def polygon_edges(vertices,y=None):
        """Computes the edges of a polygon with vertices (assumed in complex form), ordered counter-clockwise"""
        if y is not None:
            vertices = complex_form(vertices,y)
        return np.roll(vertices,-1)-vertices

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
    
    def edge_midpoints(vertices,y=None):
        if y is not None:
            vertices = complex_form(vertices,y)
        return 0.5*(np.roll(vertices,-1)+vertices)

    def boundary_points(x,y,m,method='even',skip=None):
        """Generates points along the boundary of a polygon with vertices x and y,
        ordered counter-clockwise. Makes m points for each side."""
        raise NotImplementedError('Needs to be updated for complex arithmetic')
        mask = np.ones(len(x),dtype=bool)
        if skip is not None:
            mask[skip] = 0
        if method == 'even':
            x_b = np.linspace(x,np.roll(x,-1),m+2)[1:-1,mask].flatten(order='F')
            y_b = np.linspace(y,np.roll(y,-1),m+2)[1:-1,mask].flatten(order='F')
        elif method == 'legguass':
            raise(NotImplementedError)
        return x_b,y_b

    def interior_points(m,vertices,y=None,oversamp=10):
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
            return interior_points(m,vertices,oversamp=2*oversamp)
        return x_i[mask][:m] + 1j*y_i[mask][:m]

    def plot_polygon(vertices,y=None,ax=None,**plotkwargs):
        """Plot a polygon with vertices in complex form, which are assumed to be in
        counter-clockwise order."""
        if y is not None:
            vertices = complex_form(vertices,y)
        v_ = np.append(vertices,vertices[0])
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        ax.plot(v_.real,v_.imag,**plotkwargs)
        if ax is None:
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

    def polygon_perimiter(vertices,y=None):
        if y is not None:
            vertices = complex_form(vertices,y)
        return np.sum(edge_lengths(vertices))

    def reg_polygon(r,n):
        """Generates a regular polygon with n vertices and radius r."""
        theta = np.linspace(0,2*np.pi,n+1)
        return r*np.exp(1j*theta)[:-1]

    def edge_indices(points,vertices):
        """Takes an array of points and identifies which polygon edge the points lie on,
        or if they are not on an edge."""
        raise NotImplementedError('Needs to be updated for complex arithmetic')
        x,y = points[:,0],points[:,1]
        n = len(vertices)
        arr = np.full(len(points),n,dtype='int')
        for i in range(n):
            j = i+1
            if j==n: j=0
            u,v = vertices[i],vertices[j]
            mask = np.isclose(np.sqrt((x-u[0])**2+(y-u[1])**2)+np.sqrt((x-v[0])**2+(y-v[1])**2),
                            np.sqrt((u[0]-v[0])**2+(u[1]-v[1])**2))
            arr[mask] = i
        return arr