import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from shapely.geometry import Polygon
from shapely import points
from scipy.spatial.distance import cdist

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

def seg_angles(x,y):
    dx_m = np.roll(x,-1)-x
    dy_m = np.roll(y,-1)-y
    return np.arctan2(dy_m,dx_m)

def boundary_points(x,y,m,method='even',skip=None):
    mask = np.ones(len(x),dtype=bool)
    if skip is not None:
        mask[skip] = False
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

def interior_points_test(x,y,m,oversamp=10):
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    x_i = (x_max-x_min)*np.random.rand(oversamp*m)+x_min
    y_i = (y_max-y_min)*np.random.rand(oversamp*m)+y_min

    poly = Polygon(np.array([x,y]).T)
    pts = points(np.array([x_i,y_i]).T)
    mask = poly.contains(pts)
    return mask.sum()/(oversamp*m), m-mask.sum()

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
