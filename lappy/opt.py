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

# def parabola_vertex(x,y):
#     """Finds the vertex of a parabola passing through the points
#     {(x[0],y[0]),(x[1],y[1]),(x[2],y[2])}"""
#     dx0,dx1,dx2 = x[1]-x[0],x[2]-x[1],x[2]-x[0]
#     dy0,dy1,dy2 = y[1]-y[0],y[2]-y[1],y[2]-y[0]

#     C = (dx0*dy1 - dx1*dy0)/(dx0*dx1*dx2)
#     if C<=0 or np.isnan(C): vertex = x[np.argmin(y)]
#     else: vertex = (x[1]+x[2]-dy1/(dx1*C))/2
#     return vertex

# def parabolic_iter_min(f,x,y,xtol=1e-12,maxiter=10,maxresc=2,resc_param=0.1,nrecurse=0,verbose=0):
#     """Finds a local minimum of f in the interval [x[0],x[2]] by repeated parabolic interpolation."""
#     tabs = min(nrecurse,10)*"\t"
#     if x[1]<=x[0] or x[2]<=x[1]:
#         raise ValueError('x not increasing!')

#     # shift left endpoint to zero for maximum precisioon
#     z = x-x[0]

#     # shifted interval bounds
#     zlo, zhi = z[0],z[2]

#     # function evaluation & rescue counters
#     fevals = 0
#     rescues = 0

#     # previous vertex
#     vold = np.nan

#     if verbose > 0: print(tabs+f"parabolic_iter_min on ({x[0]:.3e},{x[1]:.3e},{x[2]:.3e})")
#     # iterative parabolic interpolation
#     for i in range(maxiter):
#         v = parabola_vertex(z,y)

#         # if the vertex hasn't changed much, conclude iteration
#         if np.abs(v-vold)<xtol:
#             break

#         # vertex falls off left, rescue if possible
#         elif v<zlo-2*xtol:
#             if verbose > 1: print(tabs+'fell off left')
#             if rescues < maxresc:
#                 rescues += 1
#                 v = (1-resc_param)*zlo + resc_param*z[1]
#             else:
#                 if verbose > 1: print(tabs+'too many rescues')
#                 return None, fevals
#         # vertex falls off right, rescue if possible
#         elif v>zhi+2*xtol:
#             if verbose > 1: print(tabs+'fell off right')
#             if rescues < maxresc:
#                 rescues += 1
#                 v = (1-resc_param)*zhi + resc_param*z[1]
#             else:
#                 if verbose > 1: print(tabs+'too many rescues')
#                 return None, fevals
#         # if vertex too close to old points, wiggle it
#         if np.abs(z-v).min() < xtol/2: v += xtol

#         # interpolate using new vertex and adjacent points in z
#         vold = v
#         yv = f(v+x[0])
#         fevals += 1
#         if v<z[1]:
#             z = np.array([z[0],v,z[1]])
#             y = np.array([y[0],yv,y[1]])
#         elif z[1]<v:
#             z = np.array([z[1],v,z[2]])
#             y = np.array([y[1],yv,y[2]])
#         if verbose > 1: print(tabs+f"z={np.array_str(z,precision=3)}")

#     # return final vertex, shifted back to original interval
#     if verbose > 0: print(tabs+f'parabolic_iter_min concluded, x_min={v+x[0]}')
#     return float(v+x[0]), fevals

rho = (3-5**0.5)/2
def golden_search(f,a,b,tol=1e-15,maxiter=100):
    """Golden ratio minimization search"""
    h = b-a
    u, v = a+rho*h, b-rho*h
    fu, fv = f(u), f(v)
    fevals = 2
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
        fevals += 1
    if f(a)<f(b): return a,fevals
    else: return b,fevals

def find_all_roots(f, a, b, n):
    x = np.linspace(a, b, n)
    y = f(x)

    roots = []
    for i in range(n-1):
        if y[i]*y[i+1] < 0:
            roots.append(brentq(f, x[i], x[i+1]))
    return roots

# def secant_root(x1,y1,x2,y2):
#     """Returns the root of the secant line passing through (x1,y1) and (x2,y2).
#     Automatically handles array broadcasting for multiple sets of lines."""
#     return x1-(y1*(x2-x1))/(y2-y1)

# def interval_check(x,xgrid):
#     """Identifies which subinterval a value x lies within, compared to xgrid. Points 'near' the ends of the grid
#     are flagged as being in the first or last subinterval."""
#     if (2*xgrid[0]-xgrid[1] <= x < xgrid[0]): return 0
#     elif (xgrid[-1] <= x <= 2*xgrid[-1]-xgrid[-2]): return len(xgrid)-2
#     elif (xgrid[0] <= x <= xgrid[-1]): return np.nonzero(x>=xgrid)[0][-1]
#     else: return np.inf

# def check_flag_recurse(idx, x, y, recurse_flag, verbose, tabs):
#     # estimate y2 zeros from segments
#     x1,x2 = x[idx-1:idx+1],x[idx:idx+2]
#     y1,y2 = y[1,idx-1:idx+1],y[1,idx:idx+2]
#     z1,z2 = secant_root(x1,y1,x2,y2)

#     # find intervals where estimated zeros live
#     idx1,idx2 = interval_check(z1,x),interval_check(z2,x)

#     # no nearby predicted zeros, don't recurse
#     if ((idx1 < idx-3) or (idx1 > idx+2)) and ((idx2 < idx-3) or (idx2 > idx+2)):
#         return False
    
#     # predicted zeros nearby
#     else:
#         # first predicted zero within three intervals, flag relevant stretch
#         if (idx-4 <= idx1 <= idx+3):
#             if verbose > 2: 
#                 print(tabs+f"z1 = {z1:.3e} in interval {idx1}=[{x[idx1]:.3e},{x[idx1+1]:.3e}]")
#                 print(tabs+f"flagging intervals {min(idx1,idx-1)} to {max(idx1,idx)} for recursion")
#             recurse_flag[min(idx1,idx-1):max(idx1,idx)+1] = True

#         # second predicted zero within three intervals, flag relevant stretch
#         if (idx-4 <= idx2 <= idx+3):
#             if verbose > 2:
#                 print(tabs+f"z2 = {z2:.3e} in interval {idx2}=[{x[idx2]:.3e},{x[idx2+1]:.3e}]")
#                 print(tabs+f"flagging intervals {min(idx2,idx-1)} to {max(idx2,idx)} for recursion")
#             recurse_flag[min(idx2,idx-1):max(idx2,idx)+1] = True
#         return True

# def gridmin(f,x,y,xtol=1e-12,shrink=2,nrecurse=0,verbose=0):
#     """Finds all minima on a grid. Grid should have ghost points at the ends."""
#     tabs = min(nrecurse,10)*"\t" # tab spacing for verbose mode
#     if verbose > 0: 
#         print(tabs+f"searching on [{x[0]:.3e},{x[-1]:.3e}] (len={x[-1]-x[0]}) with {len(x)-1} subintervals")
#         print(tabs+f"recursive level = {nrecurse}")
    
#     # recursion flag
#     recurse_flag = np.zeros(len(x)-1,dtype=bool)

#     # parabolic min flag
#     para_min = []

#     # mark intervals as increasing/decreasing/constant
#     delta = np.sign(np.diff(y[0]))

#     # get discrete local min indices
#     locmin = np.nonzero((delta[:-1]==-1)*(delta[1:]==1))[0]+1
#     if verbose > 0: print(tabs+f"num. local min = {len(locmin)}\n")

#     # list for minima, and counter for function evaluation
#     minima = []
#     fevals = 0

#     # loop over discrete local mins
#     for idx in locmin:
#         if verbose > 0:
#             print(tabs+f"locmin at x={x[idx]:.3e}")
#         if verbose > 1:
#             print(tabs+f"intervals {idx-1}=[{x[idx-1]:.3e},{x[idx]:.3e}] and {idx}=[{x[idx]:.3e},{x[idx+1]:.3e}]")

#         if recurse_flag[idx-1] or recurse_flag[idx]:
#             if verbose > 0: print(tabs+"interval already flagged for recursion")
#             recurse_flag[idx-1:idx+1] = True

#         # check to see if subinterval is too finely spaced for recursion
#         elif x[idx+1]-x[idx-1] < xtol:
#             if verbose > 0: print(tabs+f"grid spacing at xtol, flagging for parabolic minimzation")
#             para_min.append(idx)
        
#         # check to see if the interval (or others) needs to be flagged for recursion and grid refinement
#         elif not check_flag_recurse(idx, x, y, recurse_flag, verbose, tabs):
#             if verbose > 0: print(tabs+"no nearby predicted zeros, flagging for parabolic minimization")
#             para_min.append(idx)
#         if verbose: print("")

#     # run parabolic minimization on locmin intervals that were not flagged for recursion
#     if len(para_min) >= 1:
#         for idx in para_min:
#             if recurse_flag[idx-1] or recurse_flag[idx]:
#                 recurse_flag[idx-1:idx+1] = True
#             else:
#                 min_,fe = parabolic_iter_min(lambda x: f(x)[0]**2,x[idx-1:idx+2],
#                                          y[0,idx-1:idx+2]**2,xtol=xtol,
#                                          nrecurse=nrecurse,verbose=verbose-1)
#                 fevals += fe
#                 if min_ is not None:
#                     minima.append(min_)

#     # recurse if needed to refine the grid
#     if np.any(recurse_flag):
#         if verbose > 1: 
#             print(tabs+"setting up recursion...")
#             print(tabs+f"pre-recursion found minima: {np.array_str(np.array(minima),precision=3)}")
#             print(tabs+f"pre-recursion fevals={fevals}")
        
#         # recurse on flagged intervals
#         # flags are padded to easily identify consecutive runs of flagged intervals
#         padded_flags = np.zeros(len(recurse_flag)+2,dtype=int)
#         padded_flags[1:-1] = recurse_flag.astype(int)
#         diffs = np.diff(padded_flags)
#         starts = np.nonzero(diffs==1)[0]
#         ends = np.nonzero(diffs==-1)[0]

#         # loop over runs of flagged intervals
#         for start,end in zip(starts,ends):
#             length = end-start # number of intervals in this run
#             # x and y grids for recursive call
#             x_tmp = np.concatenate(([x[start]],
#                                     *np.linspace(x[np.arange(start,end)],x[np.arange(start,end)+1],shrink+1)[1:].T))
#             y_tmp = np.empty((2,len(x_tmp)))

#             # fill in known values of y=f(x)
#             y_tmp[:,::shrink] = y[:,start:end+1]

#             # evaluate f(x) on new grid points
#             for i in range(length):
#                 y_tmp[:,1+i*shrink:(i+1)*shrink] = np.array([f(x_) for x_ in x_tmp[1+i*shrink:(i+1)*shrink]]).T
#                 fevals += shrink-1
            

#             # add ghost points to runs that don't already have them
#             if start > 0:
#                 # runs that don't have the leading ghost point
#                 # prepend x,y from previous interval endpoint
#                 x_tmp2 = np.concatenate(([x[start-1]],x_tmp))
#                 y_tmp2 = np.empty((2,len(x_tmp)+1))
#                 y_tmp2[:,1:] = y_tmp
#                 y_tmp2[:,0] = y[:,start-1]
#                 x_tmp,y_tmp = x_tmp2,y_tmp2
#             if end < len(x)-1:
#                 # runs that don't have the trailing ghost point
#                 # append x,y from following interval endpoint
#                 x_tmp2 = np.concatenate((x_tmp,[x[end+1]]))
#                 y_tmp2 = np.empty((2,len(x_tmp)+1))
#                 y_tmp2[:,:-1] = y_tmp
#                 y_tmp2[:,-1] = y[:,end+1]
#                 x_tmp,y_tmp = x_tmp2,y_tmp2

#             # check for flat objective at this scale
#             band = y_tmp[0].max()-y_tmp[0].min()
#             if band < 2*xtol:
#                 if verbose > 0: print(tabs+f"f is flat at this scale, using midpoint")
#                 minima += [(x[end]+x[start])/2]

#             # recurse, extending the list of minima and incrementing the function evaluations
#             else:
#                 if verbose > 0: print(tabs+f"recursing on [{x[start]:.3e},{x[end]:.3e}]\n")
#                 mins,fe = gridmin(f,x_tmp,y_tmp,xtol=xtol,shrink=shrink,nrecurse=nrecurse+1,verbose=verbose)
#                 minima += mins
#                 fevals += fe

#     # only keep minima that are inside the "non-ghost" interval, with some extra tolerance on each end
#     minima = [min_ for min_ in minima if (x[1]-xtol <= min_ <= x[-2]+xtol)]
#     if verbose > 1: 
#         print(tabs+f"found minima: {np.array_str(np.array(minima),precision=3)}")
#         print(tabs+f"fevals={fevals}")
#     return minima,fevals

# def discrete_loc_min_idx(y):
#     """Computes the indicies of y that are discrete local minima. Ignores endpoints (assumes use of ghost points)."""
#     return np.nonzero((y[1:-1] < y[:-2])&(y[1:-1] < y[2:]))[0]+1

# def get_refinement_brackets(recurse_flag):
#     padded_flags = np.zeros(len(recurse_flag)+2,dtype=int)
#     padded_flags[1:-1] = recurse_flag.astype(int)
#     diffs = np.diff(padded_flags)
#     starts = np.nonzero(diffs==1)[0]
#     ends = np.nonzero(diffs==-1)[0]
#     return np.array([starts,ends]).T

# def fill_refinement(f, x, y, start, end, shrink):
#     length = end-start # number of intervals in this run
#     # x and y grids for recursive call
#     x_tmp = np.concatenate(([x[start]],
#                             *np.linspace(x[np.arange(start,end)],x[np.arange(start,end)+1],shrink+1)[1:].T))
#     y_tmp = np.empty((2,len(x_tmp)))

#     # fill in known values of y=f(x)
#     y_tmp[:,::shrink] = y[:,start:end+1]

#     # evaluate f(x) on new grid points
#     fevals = 0
#     for i in range(length):
#         y_tmp[:,1+i*shrink:(i+1)*shrink] = np.array([f(x_) for x_ in x_tmp[1+i*shrink:(i+1)*shrink]]).T
#         fevals += shrink-1
    
#     # add ghost points to runs that don't already have them
#     if start > 0:
#         # runs that don't have the leading ghost point
#         # prepend x,y from previous interval endpoint
#         x_tmp2 = np.concatenate(([x[start-1]],x_tmp))
#         y_tmp2 = np.empty((2,len(x_tmp)+1))
#         y_tmp2[:,1:] = y_tmp
#         y_tmp2[:,0] = y[:,start-1]
#         x_tmp,y_tmp = x_tmp2,y_tmp2
#     if end < len(x)-1:
#         # runs that don't have the trailing ghost point
#         # append x,y from following interval endpoint
#         x_tmp2 = np.concatenate((x_tmp,[x[end+1]]))
#         y_tmp2 = np.empty((2,len(x_tmp)+1))
#         y_tmp2[:,:-1] = y_tmp
#         y_tmp2[:,-1] = y[:,end+1]
#         x_tmp,y_tmp = x_tmp2,y_tmp2
#     return x_tmp, y_tmp, fevals

# def bracket_mins(f, x, y, xtol=1e-12, shrink=2, nrecurse=0, verbose=0):
#     """Finds all minima on a grid. Grid should have ghost points at the ends."""
#     tabs = min(nrecurse,10)*"\t" # tab spacing for verbose mode
#     if verbose > 0: 
#         print(tabs+f"searching on [{x[0]:.3e},{x[-1]:.3e}] (len={x[-1]-x[0]}) with {len(x)-1} subintervals")
#         print(tabs+f"recursive level = {nrecurse}")
    
#     # recursion flag
#     recurse_flag = np.zeros(len(x)-1, dtype=bool)

#     # get discrete local min indices for y1 and y2
#     idx1 = discrete_loc_min_idx(y[0])
#     idx2 = discrete_loc_min_idx(y[1])

#     # local mins of y2 relative to mins of y1
#     idx1_2 = idx2 - idx1[:,np.newaxis]

#     # flag for refinement & recursion
#     same = np.any(idx1_2 == 0, axis=1)
#     right = np.any(idx1_2 == 1, axis=1)
#     left = np.any(idx1_2 == -1, axis=1)
#     recurse_flag[idx1] = same
#     recurse_flag[idx1] += right
#     recurse_flag[idx1] += left
#     recurse_flag[idx1+1] += right
#     recurse_flag[idx1-1] += left

#     brackets = []
#     for idx in idx1:
#         if not recurse_flag[idx]:
#             brackets.append((x[idx-1], x[idx+1]))

#     # recurse if needed to refine the grid
#     fevals = 0
#     if np.any(recurse_flag):
#         if verbose > 1: 
#             print(tabs+"setting up recursion...")
        
#         recurse_bracket_idx = get_refinement_brackets(recurse_flag)

#         # loop over runs of flagged intervals
#         for start, end in recurse_bracket_idx:
#             if x[end]-x[start] < xtol:
#                 if verbose: print(tabs+f"grid spacing below xtol, using bracket")
#                 brackets.append((x[start], x[end]))

#             else:
#                 x_tmp, y_tmp, fe = fill_refinement(f, x, y, start, end, shrink)
#                 fevals += fe
#                 # check for flat objective at this scale
#                 band = y_tmp[0].max()-y_tmp[0].min()
#                 if band < 2*xtol:
#                     if verbose > 0: print(tabs+f"f is flat at this scale, using smallest for bracket")
#                     idx = y_tmp[0].argmin()
#                     brackets.append((x_tmp[idx-1], x_tmp[idx+1]))

#                 # recurse, extending the list of minima and incrementing the function evaluations
#                 else:
#                     if verbose > 0: print(tabs+f"recursing on [{x[start]:.3e},{x[end]:.3e}]\n")
#                     bra, fe = gridmin(f, x_tmp, y_tmp, xtol=xtol, shrink=shrink, nrecurse=nrecurse+1, verbose=verbose)
#                     brackets += bra
#                     fevals += fe

#     return brackets

# def eig_obj(p,eigs_target,perim_tol=1e-15,evp_kwargs={'order':20},mps_kwargs={},eig_list=None,log=False,verbose=False):
#     from .evp import PolygonEVP
#     # check number of eigenvalues, convert targets to normalized reciprocals
#     K = len(eigs_target)
#     targets = normalized_reciprocals(eigs_target)
    
#     # get vertices from parameter vector
#     vertices = polygon_vertices(p)
#     N = len(vertices)
#     if verbose: print(f"Evaluating at p={np.array_str(p,precision=3)}")

#     # get perimeter and perimeter gradient
#     perim, perim_grad = poly_perim(vertices,jac=True)
#     if np.abs(perim-1) > perim_tol:
#         raise ValueError('perimeter is too high')
#     implicit_grad = -np.concatenate((perim_grad[1:-1].real,perim_grad[1:-1].imag))/(perim_grad[0].real)
#     vertices_jac = np.zeros((2*N,len(p)))
#     vertices_jac[0] = implicit_grad
#     vertices_jac[1:N-1,:len(p)//2] = np.eye(len(p)//2)
#     vertices_jac[N+1:-1,len(p)//2:] = np.eye(len(p)//2)

#     # rescale vertices so the first eigenvalue is O(1)
#     A,P = polygon_area(vertices),1
#     weyl_1 = ((P+np.sqrt(P**2+16*np.pi*A))/(2*A))**2

#     # get eigenvalues and eigenderivatives w.r.t vertices
#     evp = PolygonEVP(np.sqrt(weyl_1)*vertices,**evp_kwargs)
#     eigs = evp.solve_eigs_ordered(K+1,ppl=20,mps_kwargs=mps_kwargs)[0][:K]
#     if eig_list is not None:
#         eig_list.append(weyl_1*eigs)
#     eigs_jac = np.zeros((K,2*N))
#     for i in range(K):
#         dz = evp.eig_grad(eigs[i])
#         eigs_jac[i,:N] = dz.real
#         eigs_jac[i,N:] = dz.imag

#     # get normalized reciprocals
#     nus, nus_jac = normalized_reciprocals(eigs,jac=True)

#     # evaluate loss function
#     loss, loss_grad = l2_loss(nus,targets,log,jac=True)

#     # compose derivatives with chain rule
#     # obj_grad = (loss_grad.reshape(1,-1)@nus_jac)@(np.outer(eigs_jac[:,0],implicit_grad) + np.delete(eigs_jac,[0,N,N+1,2*N-1],axis=1))
#     grad = (((loss_grad.reshape(1,-1)@nus_jac)@eigs_jac)@vertices_jac)[0]

#     return loss, grad

# def normalized_reciprocals(x,jac=False):
#     if jac:
#         col = (1/x[1:]).reshape(-1,1)
#         diag = -x[0]/(x[1:]**2)
#         jacobian = np.hstack((col,np.diag(diag)))
#         return x[0]/x[1:], jacobian
#     return x[0]/x[1:]

# def l2_loss(x,y,log=False,jac=False):
#     diff = x-y
#     out = la.norm(diff)**2
#     if jac:
#         grad = 2*diff
#         if log: return np.log(out),grad/out
#         else: return out, grad
#     if log: return np.log(out)
#     else: return out