import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from shapely.geometry import Polygon
from shapely import points
from scipy.spatial.distance import cdist
import sklearn.gaussian_process as gp

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
    theta = np.zeros((len(x_v),len(x)))
    dx_p = np.roll(x_v,-1)-x_v
    dy_p = np.roll(y_v,-1)-y_v
    for i in range(len(x_v)):
        dx, dy = x-x_v[i], y-y_v[i]
        theta[i] = np.arccos(1-cdist([[dx_p[i],dy_p[i]]],np.array([dx,dy]).T,'cosine'))
        reentrant = (dx_p[i]*dy - dy_p[i]*dx)<0
        theta[i][reentrant] = 2*np.pi - theta[i][reentrant]
    return theta.T

def calc_angles(x,y):
    """Computes the interior angles of a polygon with vertices x and y, ordered
    counter-clockwise"""
    dx_p, dx_m = np.roll(x,-1)-x, np.roll(x,1)-x
    dy_p, dy_m = np.roll(y,-1)-y, np.roll(y,1)-y

    theta = np.arccos((dx_p*dx_m+dy_p*dy_m)/np.sqrt((dx_p**2+dy_p**2)*(dx_m**2+dy_m**2)))
    reentrant = (-dy_m*(dx_p-dx_m)+dx_m*(dy_p-dy_m))>0
    theta[reentrant] = 2*np.pi - theta[reentrant]
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
    a help"""
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
    elif method == 'chebyshev':
        pass
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

def plot_polygon(x,y,ax=None):
    """Plot a polygon with vertices x and y, which are assumed to be in
    counter-clockwise order."""
    x_ = np.append(x,x[0])
    y_ = np.append(y,y[0])
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    ax.plot(x_,y_)
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
    if f(a)<f(b): return a
    else: return b

def gp_minsearch(f,a,b,length_scale=2.5,nu=2.5,scale=0.3**2,n=30,alpha=0.3,beta=0.0,gamma=0.1,tol=1e-5):
    kernel = scale*gp.kernels.Matern(nu=nu,length_scale=length_scale)
    gpr = gp.GaussianProcessRegressor(kernel=kernel,n_targets=2,optimizer=None)

    x_train = np.linspace(a,b,n)
    y_train = np.array([f(x) for x in x_train])
    intervals = np.array([x_train[:-1],x_train[1:]]).T
    x_train = x_train.reshape(-1,1)
    gpr.fit(x_train,y_train)

    def obj(x,alpha=alpha,beta=beta,gamma=gamma):
        if x.ndim==0: x = x.reshape(1,-1)
        mu,sigma = gpr.predict(x,return_std=True)
        return (mu[:,0] - alpha*(sigma[:,0])/(mu[:,1]-beta*sigma[:,1]+gamma))

    feval = len(x_train)
    while len(intervals) > 0:
        new_intervals = []
        for interval in intervals:
            lcb_min = golden_search(obj,interval[0],interval[1],tol=tol)
            if (lcb_min == interval[0]) or (lcb_min == interval[1]):
                pass
            else:
                x_new = lcb_min
                y_new = f(x_new)
                feval += 1
                x_train = np.append(x_train,[[x_new]],axis=0)
                y_train = np.append(y_train,[y_new],axis=0)
                gpr.fit(x_train,y_train)
                if x_new-interval[0]>tol:
                    new_intervals.append((interval[0],x_new))
                if interval[1]-x_new>tol:
                    new_intervals.append((x_new,interval[1]))
        intervals = new_intervals

    x_train = x_train.flatten()
    sort_idx = x_train.argsort()
    x_train = x_train[sort_idx]
    y_train = y_train[sort_idx]
    return gpr,x_train,y_train

def discrete_local_min_idx(x,y):
    x = x.flatten()
    sort_idx = x.argsort()
    x = x[sort_idx]
    y = y[sort_idx]

    min_idx = np.argwhere((y[1:-1]<y[:-2])*(y[1:-1]<y[2:]))
    return min_idx+1
