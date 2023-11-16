import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from shapely.geometry import Polygon
from shapely import points
from scipy.spatial.distance import cdist

def radii(x,y,x_v,y_v):
    """Computes the radial distance from each point in x,y to each polygon vertex in x_v, y_v.
    For use in evaluating Fourier-Bessel functions in the Method of Particular Solutions.
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
    dx_p, dx_m = np.roll(x,-1)-x, np.roll(x,1)-x
    dy_p, dy_m = np.roll(y,-1)-y, np.roll(y,1)-y

    theta = np.arccos((dx_p*dx_m+dy_p*dy_m)/np.sqrt((dx_p**2+dy_p**2)*(dx_m**2+dy_m**2)))
    reentrant = (-dy_m*(dx_p-dx_m)+dx_m*(dy_p-dy_m))>0
    theta[reentrant] = 2*np.pi - theta[reentrant]
    return theta

def calc_dists(x,y):
    dx_p, dx_m = np.roll(x,1)-x, np.roll(x,-1)-x
    dy_p, dy_m = np.roll(y,1)-y, np.roll(y,-1)-y

    return (dx_p**2+dy_p**2)**0.5, (dx_m**2+dy_m**2)**0.5

def calc_normals(x,y):
    dx, dy = np.roll(x,-1)-x, np.roll(y,-1)-y
    d = (dx**2+dy**2)**0.5
    n = np.vstack((dy,-dx))/d
    return n

def seg_angles(x,y):
    dx_m = np.roll(x,-1)-x
    dy_m = np.roll(y,-1)-y
    return np.arctan2(dy_m,dx_m)

def boundary_points(x,y,m,method='even',skip=None):
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
    """Plot a polygon with vertices from x and y, which are assumed to be in order"""
    x_ = np.append(x,x[0])
    y_ = np.append(y,y[0])
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    ax.plot(x_,y_)
    if ax is None:
        plt.show()

def plot_angles(x,y,ax=None):
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
    return m**2*np.pi**2/L**2 + n**2*np.pi**2/H**2

def rect_eig_bound_idx(bound,L,H):
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
    return 0.5*np.sum((x-np.roll(x,-1))*(y+np.roll(y,-1)))

def poly_eig_lower_bound(k,x,y):
    A = polygon_area(x,y)
    return 2/A*k

def rect_lambda_grad(m,n,L,H):
    m2L3 = m**2/L**3
    n2H3 = n**2/H**3
    return (np.pi**2)*np.array([m2L3,-m2L3,-m2L3,m2L3,n2H3,n2H3,-n2H3,-n2H3])

def rect_eig_mult(lambda_,L,H,maxind=1000):
    Lam = rect_lambda(np.arange(1,maxind)[np.newaxis],np.arange(1,maxind)[:,np.newaxis],1,1)
    diff = np.abs(lambda_-Lam)
    tot = (diff<1e-12).sum()
    ind = np.unravel_index(np.argsort(diff, axis=None), diff.shape)
    return (ind[0]+1)[:tot],(ind[1]+1)[:tot]

def rect_eig_mult_mn(m,n,L,H):
    return rect_eig_mult(rect_lambda(m,n,L,H),L,H,maxind=10*max(m,n))

def reg_polygon(r,n):
    theta = np.linspace(0,2*np.pi,n+1)
    z = r*np.exp(1j*theta)[:-1]
    return list(z.real),list(z.imag)

def edge_indices(nodes,vertices):
    x,y = nodes[:,0],nodes[:,1]
    n = len(vertices)
    arr = np.full(len(nodes),n,dtype='int')
    for i in range(n):
        j = i+1
        if j==n: j=0
        u,v = vertices[i],vertices[j]
        mask = np.isclose(np.sqrt((x-u[0])**2+(y-u[1])**2)+np.sqrt((x-v[0])**2+(y-v[1])**2),
                          np.sqrt((u[0]-v[0])**2+(u[1]-v[1])**2))
        arr[mask] = i
    return arr

def invert_permutation(p):
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s
