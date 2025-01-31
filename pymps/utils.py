import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import patches, cm, ticker, colors
from shapely.geometry import Polygon
from shapely import points
from scipy.spatial.distance import cdist

def radii(x,y,x_v,y_v):
    """Computes the radial distance from each point in x,y to each polygon vertex
    in x_v, y_v. For use in evaluating Fourier-Bessel functions in the Method of
    Particular Solutions.
    """
    return cdist(np.array([x,y]).T,np.array([x_v,y_v]).T)

def thetas(x,y,x_v,y_v):
    """Computes the angles between given points and the polygon edges which are
    counter-clockwise from the vertices (x_i,v_i). For use in evaluating Fourier-Bessel
    functions in the Method of Particular Solutions."""
    dx_v = np.roll(x_v,-1)-x_v
    dy_v = np.roll(y_v,-1)-y_v
    int_angles = calc_angles(x_v,y_v)

    out = np.empty((len(x_v),len(x)))
    for i in range(len(x_v)):
        dx,dy = x-x_v[i],y-y_v[i]
        out[i] = np.arctan2(dx_v[i]*dy-dx*dy_v[i],dx_v[i]*dx+dy_v[i]*dy)
        out[i][out[i]<0] += 2*np.pi
        out[i][out[i]>(int_angles[i]/2+np.pi)] -= 2*np.pi

    return out.T

def calc_angles(x,y):
    """Computes the interior angles of a polygon with vertices x and y, ordered
    counter-clockwise"""
    dx_p, dx_m = np.roll(x,-1)-x, np.roll(x,1)-x
    dy_p, dy_m = np.roll(y,-1)-y, np.roll(y,1)-y

    theta = np.arctan2(dx_p*dy_m-dx_m*dy_p,dx_p*dx_m+dy_p*dy_m)
    theta[theta<0] += 2*np.pi
    return theta

def calc_dists(x,y):
    """Computes the side lengths of a polygon with vertices x and y, ordered
    counter-clockwise"""
    dx_p, dx_m = np.roll(x,1)-x, np.roll(x,-1)-x
    dy_p, dy_m = np.roll(y,1)-y, np.roll(y,-1)-y

    return (dx_p**2+dy_p**2)**0.5, (dx_m**2+dy_m**2)**0.5

def calc_normals(x,y):
    """Computes the outward-pointing unit normal vectors to the sides of a
    polygon with vertices x and y, ordered counter-clockwise"""
    dx, dy = np.roll(x,-1)-x, np.roll(y,-1)-y
    d = (dx**2+dy**2)**0.5
    n = np.vstack((dy,-dx))/d
    return n

def seg_angles(x,y):
    """Gets the angle of each side of the polygon compared to the x-axis. Purely
    a helper function"""
    dx_m = np.roll(x,-1)-x
    dy_m = np.roll(y,-1)-y
    return np.arctan2(dy_m,dx_m)

def boundary_points(x,y,m,method='even',skip=None):
    """Generates points along the boundary of a polygon with vertices x and y,
    ordered counter-clockwise. Makes m points for each side."""
    mask = np.ones(len(x),dtype=bool)
    if skip is not None:
        mask[skip] = 0
    if method == 'even':
        x_b = np.linspace(x,np.roll(x,-1),m+2)[1:-1,mask].flatten(order='F')
        y_b = np.linspace(y,np.roll(y,-1),m+2)[1:-1,mask].flatten(order='F')
    elif method == 'legguass':
        raise(NotImplementedError)
    return x_b,y_b

def interior_points(x,y,m,oversamp=10):
    """Computes random interior points for a polygon with vertices x and y,
    ordered counter-clockwise."""
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    x_i = (x_max-x_min)*np.random.rand(oversamp*m)+x_min
    y_i = (y_max-y_min)*np.random.rand(oversamp*m)+y_min

    poly = Polygon(np.array([x,y]).T)
    pts = points(np.array([x_i,y_i]).T)
    mask = poly.contains(pts)
    if mask.sum() < m:
        return interior_points(x,y,oversamp=2*oversamp)
    return x_i[mask][:m], y_i[mask][:m]

def eps_boundary_points(x,y,m,eps,method='even',skip=None):
    """Computetes points near the boundary. Probably not needed anymore with
    the most recent developments in boundary derivative calculation."""
    mask = np.ones(len(x),dtype=bool)
    n = calc_normals(x,y)
    if skip is not None:
        mask[skip] = 0
    if method == 'even':
        x_b_eps = (np.linspace(x,np.roll(x,-1),m+2)[1:-1,mask]-eps*n[0][mask]).flatten(order='F')
        y_b_eps = (np.linspace(y,np.roll(y,-1),m+2)[1:-1,mask]-eps*n[1][mask]).flatten(order='F')
    elif method == 'chebyshev':
        pass
    return x_b_eps,y_b_eps

def plot_polygon(x,y,ax=None,**plotkwargs):
    """Plot a polygon with vertices x and y, which are assumed to be in
    counter-clockwise order."""
    x_ = np.append(x,x[0])
    y_ = np.append(y,y[0])
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    ax.plot(x_,y_,**plotkwargs)
    if ax is None:
        plt.show()

def plot_angles(x,y,ax=None):
    """Plots the angle arcs to confirm accurate calculation of angles. Also looks
    nice."""
    theta = np.rad2deg(calc_angles(x,y))
    d = .5*np.minimum(*calc_dists(x,y))
    start = np.rad2deg(seg_angles(x,y))
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    for i in range(len(theta)):
        arc = patches.Arc((x[i],y[i]),d[i],d[i],angle=start[i],theta2=theta[i])
        ax.add_patch(arc)

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
    return x,y

def rect_lambda(m,n,L,H):
    """Computes the (m,n) Dirichlet eigenvalue of an L-by-H rectangle"""
    return m**2*np.pi**2/L**2 + n**2*np.pi**2/H**2

rect_eig = rect_lambda

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
        eig = rect_lambda(m_max,1,L,H)
        if eig > bound: break
        m_max += 1

    n_max = 1
    while True:
        eig = rect_lambda(1,n_max,L,H)
        if eig > bound: break
        n_max += 1

    M = np.arange(1,m_max+1)[:,np.newaxis]
    N = np.arange(1,n_max+1)[np.newaxis]
    Lambda = rect_lambda(M,N,L,H)
    return np.argwhere(Lambda <= bound)+1

def polygon_area(x,y):
    """Computes the area of a polygon with vertices x and y,
    ordered counter-clockwise. Uses the Shoelace formula."""
    return 0.5*np.sum((x-np.roll(x,-1))*(y+np.roll(y,-1)))

def polygon_perimiter(x,y):
    return np.sqrt((np.roll(x,-1)-x)**2+(np.roll(y,-1)-y)**2).sum()

def poly_eig_lower_bound(k,x,y):
    """Very very weak lower bound for planar Dirichlet eigenvalues."""
    A = polygon_area(x,y)
    return 2/A*k

def rect_lambda_grad(m,n,L,H):
    """Gradients of *simple* rectangular eigenvalues with respect to rectangle
    vertices. Used to test derivative estimation code."""
    m2L3 = m**2/L**3
    n2H3 = n**2/H**3
    return (np.pi**2)*np.array([m2L3,-m2L3,-m2L3,m2L3,n2H3,n2H3,-n2H3,-n2H3])

def rect_eig_mult(lambda_,L,H,maxind=1000):
    """Compute the indices of rectangle Dirichlet eigenvalues which are
    duplicates of lambda_. For use in testing multiplicity."""
    Lam = rect_lambda(np.arange(1,maxind)[np.newaxis],np.arange(1,maxind)[:,np.newaxis],L,H)
    diff = np.abs(lambda_-Lam)
    tot = (diff<1e-12).sum()
    ind = np.unravel_index(np.argsort(diff, axis=None), diff.shape)
    return (ind[0]+1)[:tot],(ind[1]+1)[:tot]

def rect_eig_mult_mn(m,n,L,H):
    """Compute the indices of rectangle Dirichlet eigenvalues which are
    duplicates of the (m,n) eigenvalue. For use in testing multiplicity."""
    return rect_eig_mult(rect_lambda(m,n,L,H),L,H,maxind=10*max(m,n))

def reg_polygon(r,n):
    """Generates a regular polygon with n vertices and radius r."""
    theta = np.linspace(0,2*np.pi,n+1)
    z = r*np.exp(1j*theta)[:-1]
    return list(z.real),list(z.imag)

def edge_indices(points,vertices):
    """Takes an array of points and identifies which polygon edge the points lie on,
    or if they are not on an edge."""
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

# def gp_minsearch(f,a,b,length_scale=2.5,nu=2.5,scale=0.3**2,n=30,alpha=0.3,beta=0.0,gamma=0.1,tol=1e-5):
#     kernel = scale*gp.kernels.Matern(nu=nu,length_scale=length_scale)
#     gpr = gp.GaussianProcessRegressor(kernel=kernel,n_targets=2,optimizer=None)
#
#     x_train = np.linspace(a,b,n)
#     y_train = np.array([f(x) for x in x_train])
#     intervals = np.array([x_train[:-1],x_train[1:]]).T
#     x_train = x_train.reshape(-1,1)
#     gpr.fit(x_train,y_train)
#
#     def obj(x,alpha=alpha,beta=beta,gamma=gamma):
#         if x.ndim==0: x = x.reshape(1,-1)
#         mu,sigma = gpr.predict(x,return_std=True)
#         return (mu[:,0] - alpha*(sigma[:,0])/(mu[:,1]-beta*sigma[:,1]+gamma))
#
#     feval = len(x_train)
#     while len(intervals) > 0:
#         new_intervals = []
#         for interval in intervals:
#             lcb_min = golden_search(obj,interval[0],interval[1],tol=tol)
#             if (lcb_min == interval[0]) or (lcb_min == interval[1]):
#                 pass
#             else:
#                 x_new = lcb_min
#                 y_new = f(x_new)
#                 feval += 1
#                 x_train = np.append(x_train,[[x_new]],axis=0)
#                 y_train = np.append(y_train,[y_new],axis=0)
#                 gpr.fit(x_train,y_train)
#                 if x_new-interval[0]>tol:
#                     new_intervals.append((interval[0],x_new))
#                 if interval[1]-x_new>tol:
#                     new_intervals.append((x_new,interval[1]))
#         intervals = new_intervals
#
#     x_train = x_train.flatten()
#     sort_idx = x_train.argsort()
#     x_train = x_train[sort_idx]
#     y_train = y_train[sort_idx]
#     return gpr,x_train,y_train

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
