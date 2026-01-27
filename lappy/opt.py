import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.optimize import brentq
from .utils import polygon_area
from .param import polygon_vertices, poly_perim

def parabola_vertex(x,y):
    """Finds the vertex of a parabola passing through the points
    {(x[0],y[0]),(x[1],y[1]),(x[2],y[2])}"""
    dx0,dx1,dx2 = x[1]-x[0],x[2]-x[1],x[2]-x[0]
    dy0,dy1,dy2 = y[1]-y[0],y[2]-y[1],y[2]-y[0]

    C = (dx0*dy1 - dx1*dy0)/(dx0*dx1*dx2)
    if C<=0 or np.isnan(C): vertex = x[np.argmin(y)]
    else: vertex = (x[1]+x[2]-dy1/(dx1*C))/2
    return vertex

def parabolic_iter_min(f, x, y, xtol=1e-12, maxiter=10, maxresc=2, resc_param=0.1, nrecurse=0, verbose=0):
    """Finds a local minimum of f in the interval [x[0],x[2]] by repeated parabolic interpolation."""
    tabs = min(nrecurse,10)*"\t"
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

    if verbose > 0: print(tabs+f"parabolic_iter_min on ({x[0]:.3e},{x[1]:.3e},{x[2]:.3e})")
    # iterative parabolic interpolation
    for i in range(maxiter):
        v = parabola_vertex(z,y)

        # if the vertex hasn't changed much, conclude iteration
        if np.abs(v-vold)<xtol:
            break

        # vertex falls off left, rescue if possible
        elif v<zlo-2*xtol:
            if verbose > 1: print(tabs+'fell off left')
            if rescues < maxresc:
                rescues += 1
                v = (1-resc_param)*zlo + resc_param*z[1]
            else:
                if verbose > 1: print(tabs+'too many rescues')
                return None, fevals
        # vertex falls off right, rescue if possible
        elif v>zhi+2*xtol:
            if verbose > 1: print(tabs+'fell off right')
            if rescues < maxresc:
                rescues += 1
                v = (1-resc_param)*zhi + resc_param*z[1]
            else:
                if verbose > 1: print(tabs+'too many rescues')
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
        if verbose > 1: print(tabs+f"z={np.array_str(z,precision=3)}")

    # return final vertex, shifted back to original interval
    if verbose > 0: print(tabs+f'parabolic_iter_min concluded, x_min={v+x[0]}')
    return float(v+x[0]), fevals

def get_refinement_intervals(recurse_flag):
    padded_flags = np.zeros(len(recurse_flag)+2,dtype=int)
    padded_flags[1:-1] = recurse_flag.astype(int)
    diffs = np.diff(padded_flags)
    starts = np.nonzero(diffs==1)[0]-1
    ends = np.nonzero(diffs==-1)[0]
    return np.array([starts,ends]).T

def discrete_loc_min_idx(y):
    """Computes the indicies of y that are discrete local minima. Ignores endpoints (assumes use of ghost points)."""
    return np.nonzero((y[1:-1] < y[:-2])&(y[1:-1] < y[2:]))[0]+1

def fill_refinement(f, x, y, start, end, shrink):
    length = end-start # number of intervals in this run
    # x and y grids for recursive call
    x_tmp = np.concatenate(([x[start]],
                            *np.linspace(x[np.arange(start,end)],x[np.arange(start,end)+1],shrink+1)[1:].T))
    y_tmp = np.empty((2,len(x_tmp)))

    # fill in known values of y=f(x)
    y_tmp[:,::shrink] = y[:,start:end+1]

    # evaluate f(x) on new grid points
    fevals = 0
    for i in range(length):
        y_tmp[:,1+i*shrink:(i+1)*shrink] = np.array([f(x_) for x_ in x_tmp[1+i*shrink:(i+1)*shrink]]).T
        fevals += shrink-1
    
    # add ghost points to runs that don't already have them
    if start > 0:
        # runs that don't have the leading ghost point
        # prepend x,y from previous interval endpoint
        x_tmp2 = np.concatenate(([x[start-1]],x_tmp))
        y_tmp2 = np.empty((2,len(x_tmp)+1))
        y_tmp2[:,1:] = y_tmp
        y_tmp2[:,0] = y[:,start-1]
        x_tmp,y_tmp = x_tmp2,y_tmp2
    if end < len(x)-1:
        # runs that don't have the trailing ghost point
        # append x,y from following interval endpoint
        x_tmp2 = np.concatenate((x_tmp,[x[end+1]]))
        y_tmp2 = np.empty((2,len(x_tmp)+1))
        y_tmp2[:,:-1] = y_tmp
        y_tmp2[:,-1] = y[:,end+1]
        x_tmp,y_tmp = x_tmp2,y_tmp2
    return x_tmp, y_tmp, fevals

def parabolic_gridmin(f, x, y, xtol=1e-12, shrink=2, nrecurse=0, para_kwargs={}, verbose=0):
    """Minimizes the first component of the length 2 vector-valued function f using a gridsearch and parabolic fitting.
    Refines the grid based on proximity of local minima of the second component of f"""
    
    tabs = min(nrecurse,10)*"\t" # tab spacing for verbose mode
    if verbose > 0: 
        print(tabs+f"searching on [{x[0]:.3e},{x[-1]:.3e}] (len={x[-1]-x[0]:.2e})")
        print(tabs+f"with {len(x)-1} subintervals")
        print(tabs+f"recursive level = {nrecurse}")
    
    # recursion flag
    recurse_flag = np.zeros(len(x)-1, dtype=bool)

    # get discrete local min indices for y1 and y2
    y0_min_idx = discrete_loc_min_idx(y[0])
    y1_min_idx = discrete_loc_min_idx(y[1])

    # local mins of y2 relative to mins of y1
    y1_min_idx_rel = y1_min_idx - y0_min_idx[:,np.newaxis]

    # flag for refinement & recursion
    same = np.any(y1_min_idx_rel == 0, axis=1)
    right = np.any(y1_min_idx_rel == 1, axis=1)
    left = np.any(y1_min_idx_rel == -1, axis=1)
    recurse_flag[y0_min_idx] = same
    if np.any(right):
        recurse_flag[y0_min_idx] += right
        idx = y0_min_idx[y0_min_idx < len(recurse_flag)-1]+1
        recurse_flag[idx] += right
    if np.any(left):
        recurse_flag[y0_min_idx] += left
        idx = y0_min_idx[y0_min_idx > 0]-1
        recurse_flag[idx] += left

    minima = []
    fevals = 0
    # f^2 for iterative parabolic minimzation
    f2 = lambda x: f(x)[0]**2
    for idx in y0_min_idx:
        if not recurse_flag[idx]:
            min_, fe = parabolic_iter_min(f2, x[idx-1:idx+2], y[0,idx-1:idx+2]**2, xtol=xtol, 
                                          nrecurse=nrecurse, verbose=verbose-1, **para_kwargs)
            fevals += fe
            if min_ is not None: minima.append(min_)

    # recurse if needed to refine the grid
    if np.any(recurse_flag):
        if verbose > 1: 
            print(tabs+"setting up recursion...")
        
        recurse_interval_idx = get_refinement_intervals(recurse_flag)

        # loop over runs of flagged intervals
        for start, end in recurse_interval_idx:
            if x[end]-x[start] < xtol:
                if verbose: print(tabs+f"grid spacing below xtol, using discrete local min")
                min_idx = np.argmin(y[0,start:end+1])
                minima.append(x[start:end+1][min_idx])
            else:
                x_tmp, y_tmp, fe = fill_refinement(f, x, y, start, end, shrink)
                fevals += fe
                # check for flat objective at this scale
                band = y_tmp[0].max()-y_tmp[0].min()
                if band < 2*xtol:
                    if verbose > 0: 
                        print(tabs+f"f is flat at this scale, using discrete local min")
                        print(tabs+f"band = {band:.3e}")
                    idx = y_tmp[0].argmin()
                    minima.append(x_tmp[idx])

                # recurse, extending the list of minima and incrementing the function evaluations
                else:
                    if verbose > 0: print(tabs+f"recursing on [{x[start]:.3e},{x[end]:.3e}]\n")
                    mins, fe = parabolic_gridmin(f, x_tmp, y_tmp, xtol=xtol, shrink=shrink, 
                                                 nrecurse=nrecurse+1, para_kwargs=para_kwargs, verbose=verbose)
                    minima += mins
                    fevals += fe
                    
    # only keep minima that are inside the "non-ghost" interval, with some extra tolerance on each end
    minima = [min_ for min_ in minima if (x[1]-xtol <= min_ <= x[-2]+xtol)]
    if verbose > 1: 
        print(tabs+f"found minima: {np.array_str(np.array(minima),precision=3)}")
        print(tabs+f"fevals={fevals}")
    if nrecurse == 0:
        return np.array(minima), fevals
    else:
        return minima, fevals