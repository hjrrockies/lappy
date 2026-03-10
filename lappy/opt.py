import numpy as np
from scipy.optimize import brentq, minimize_scalar
from tqdm import tqdm

def parabola_vertex(x,y):
    """Finds the vertex of a parabola passing through the points
    {(x[0],y[0]),(x[1],y[1]),(x[2],y[2])}"""
    dx0,dx1,dx2 = x[1]-x[0],x[2]-x[1],x[2]-x[0]
    dy0,dy1,dy2 = y[1]-y[0],y[2]-y[1],y[2]-y[0]

    C = (dx0*dy1 - dx1*dy0)/(dx0*dx1*dx2)
    if C<=0 or np.isnan(C): vertex = x[np.argmin(y)]
    else: vertex = (x[1]+x[2]-dy1/(dx1*C))/2
    return vertex

def parabolic_iter_min(f, x, y, xtol=1e-12, maxiter=10, maxresc=2, resc_param=0.1, verbose=0):
    """Finds a local minimum of f in the interval [x[0],x[2]] by repeated parabolic interpolation."""
    if x[1]<=x[0] or x[2]<=x[1]:
        raise ValueError('x not increasing!')

    # shift left endpoint to zero for maximum precision
    z = x-x[0]

    # shifted interval bounds
    zlo, zhi = z[0],z[2]

    # function evaluation & rescue counters
    fevals = 0
    rescues = 0

    # previous vertex
    vold = np.nan

    if verbose > 0: print(f"parabolic_iter_min on ({x[0]:.2e},{x[1]:.2e},{x[2]:.2e})")
    # iterative parabolic interpolation
    for i in range(maxiter):
        v = parabola_vertex(z,y)
        if verbose > 1: print(f"v={v:.2e}")

        # vertex falls off left, rescue if possible
        if v<zlo-xtol:
            if verbose > 1: print('fell off left')
            if rescues < maxresc:
                rescues += 1
                v = (1-resc_param)*zlo + resc_param*z[1]
            else:
                if verbose > 1: print('too many rescues')
                return None, fevals
        # vertex falls off right, rescue if possible
        elif v>zhi+xtol:
            if verbose > 1: print('fell off right')
            if rescues < maxresc:
                rescues += 1
                v = (1-resc_param)*zhi + resc_param*z[1]
            else:
                if verbose > 1: print('too many rescues')
                return None, fevals
        # if the vertex hasn't changed much, conclude iteration
        elif np.abs(v-vold)<xtol:
            break
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
        if verbose > 1: print(f"z={np.array_str(z,precision=3)}")

    # return final vertex, shifted back to original interval
    if verbose > 0: print(f'converged, x_min={v+x[0]}')
    return float(v+x[0]), fevals

def discrete_locmin_idx(y):
    """Computes the indicies of y that are discrete local minima. Ignores endpoints (assumes use of ghost points)."""
    return np.nonzero((y[1:-1] < y[:-2])&(y[1:-1] < y[2:]))[0]+1

def flag_refinement_intervals(n_intervals, y0_min_idx, y1_min_idx):
    """Flags intervals for refinement based on concidence/adjacency of discrete local minima for y0 & y1"""
    refine_flag = np.zeros(n_intervals, dtype=bool)

    # local mins of y1 relative to local mins of y0
    min_idx_rel = y1_min_idx - y0_min_idx[:,np.newaxis]

    # flag coincident minima
    coincident = np.any(min_idx_rel == 0, axis=1)
    refine_flag[y0_min_idx] = coincident

    # flag if minima to the left (mark both subintervals)
    on_left = np.any(min_idx_rel == -1, axis=1)
    has_left = (y0_min_idx > 0) # leftmost interval has no subinterval to the left
    refine_flag[y0_min_idx[has_left]] += on_left[has_left]
    refine_flag[y0_min_idx[has_left]-1] += on_left[has_left]

    # flag if minima to the right (mark both subintervals)
    on_right = np.any(min_idx_rel == 1, axis=1)
    has_right = (y0_min_idx < len(refine_flag)-1) # rightmost interval has no subinterval to the right
    refine_flag[y0_min_idx[has_right]] += on_right[has_right]
    refine_flag[y0_min_idx[has_right]+1] += on_right[has_right]
    
    return refine_flag

def merge_refinement_intervals(refine_flag):
    """Gets indices of intervals marked for refinement, merging adjacent marked intervals."""
    padded_flags = np.zeros(len(refine_flag)+2,dtype=int)
    padded_flags[1:-1] = refine_flag.astype(int)
    diffs = np.diff(padded_flags)
    starts = np.nonzero(diffs==1)[0]-1
    ends = np.nonzero(diffs==-1)[0]
    return np.array([starts,ends]).T

def fill_refinement(f, x, y, start, end, shrink, verbose=0):
    """Refine the search grid by evaluating f at new points in the interval (x[start], x[end]), filling in
    new arrays of input-output pairs."""
    length = end-start # number of intervals in this run
    # x and y grids for recursive call
    x_tmp = np.concatenate(([x[start]],
                            *np.linspace(x[np.arange(start,end)],x[np.arange(start,end)+1],shrink+1)[1:].T))
    y_tmp = np.empty((3,len(x_tmp)))

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
        y_tmp2 = np.empty((3,len(x_tmp)+1))
        y_tmp2[:,1:] = y_tmp
        y_tmp2[:,0] = y[:,start-1]
        x_tmp, y_tmp = x_tmp2, y_tmp2
    if end < len(x)-1:
        # runs that don't have the trailing ghost point
        # append x,y from following interval endpoint
        x_tmp2 = np.concatenate((x_tmp,[x[end+1]]))
        y_tmp2 = np.empty((3,len(x_tmp)+1))
        y_tmp2[:,:-1] = y_tmp
        y_tmp2[:,-1] = y[:,end+1]
        x_tmp, y_tmp = x_tmp2, y_tmp2
    return x_tmp, y_tmp, fevals

def bracket_mins(f, x, y, xtol=1e-8, shrink=2, nrecurse=0, verbose=0):
    """Bracket the minima of f(x)[0] using a gridsearch. Returns a list of brackets which (hopefully!) each 
    contain a single local minimum of the first component of f. Uses the local minima 
    of f(x)[1] and f(x)[2] to guide refinement."""
    tabs = min(nrecurse,5)*"\t" # tab spacing for verbose mode
    if verbose > 0:
        print(tabs+f"bracket_mins on [{x[1]:.2e},{x[-2]:.2e}] (len={x[-2]-x[1]:.2e}, npts={len(x)})")

    # get discrete local min indices for y1 and y2
    y0_min_idx = discrete_locmin_idx(y[0])
    y1_min_idx = discrete_locmin_idx(y[1])
    y2_min_idx = discrete_locmin_idx(y[2])

    # get refinement flags (based on proximity of y1_mins/y2_mins relative to y0_mins)
    refine_flag1 = flag_refinement_intervals(len(x)-1, y0_min_idx, y1_min_idx)
    refine_flag2 = flag_refinement_intervals(len(x)-1, y0_min_idx, y2_min_idx)
    refine_flag = refine_flag1|refine_flag2

    # brackets that don't need refinement
    brackets = []
    for idx in y0_min_idx:
        if not refine_flag[idx]:
            brackets.append((x[idx-1:idx+2],y[0,idx-1:idx+2]))
            if verbose > 1:
                print(tabs+f"+[{x[idx-1]:.2e},{x[idx+1]:.2e}]")
    if verbose > 0 and len(y0_min_idx) == 0:
        print(f"No minima found on {x[1]:.16e} to {x[-2]:.16e}")
        with np.printoptions(precision=16):
            print(repr(x))
            print(repr(y))

    fevals = 0
    # recurse if needed to refine the grid
    if np.any(refine_flag):
        refine_interval_idx = merge_refinement_intervals(refine_flag)
        # loop over runs of flagged intervals
        for start, end in refine_interval_idx:
            tol = xtol*x[start] # relative tolerance
            # don't refine if run has length below tolerance
            if x[end]-x[start] < tol:
                min_idx = y[0,start:end+1].argmin() + start
                brackets.append((x[[start, min_idx, end]], y[0,[start, min_idx, end]]))
                if verbose > 1:
                    print(tabs+f"+[{x[start]:.2e},{x[end]:.2e}] (below xtol)")
            else:
                x_tmp, y_tmp, fe = fill_refinement(f, x, y, start, end, shrink, verbose)
                fevals += fe
                if verbose > 1:
                    print(tabs+f"refine on [{x[start]:.2e},{x[end]:.2e}], shrink={shrink}")
                # check for flat objective at this scale
                mindiff = np.abs(np.diff(y_tmp[0])).min()
                if mindiff == 0:
                    half_idx = int((end+start)/2)
                    brackets.append((x[[start,half_idx,end]],y[0,[start,half_idx,end]]))
                    if verbose > 1:
                        print(tabs+f"+[{x[start]:.2e},{x[end]:.2e}] (flat objective)")

                # recurse, extending the list of brackets and incrementing the function evaluations
                else:
                    if verbose > 0:
                        print(tabs+"recursing...")
                    bracks, fe = bracket_mins(f, x_tmp, y_tmp, xtol, shrink, nrecurse+1, verbose)
                    brackets += bracks
                    fevals += fe
    if verbose > 0 and nrecurse == 0:
        print(f"found {len(brackets)} brackets, fevals={fevals}")
    return brackets, fevals

def minimize_on_bracket(f, bracket, xtol, minsolver='parabolic', verbose=0):
    x, y = bracket # unpack bracket
    if verbose > 0:
        print(f"minimizing on [{x[0]:.2e},{x[2]:.2e}]")
    tol = xtol*x[0] # get absolute tolerance from relative

    # only minimize further if bracket is at least width tol
    if x[2]-x[0] > tol:
        if minsolver == 'parabolic':
            minimizer, fevals = parabolic_iter_min(lambda x: f(x)**2, x, y, tol, verbose=verbose-1)
        elif minsolver == 'brent':
            brent_verb = (3 if verbose > 2 else max(verbose-1,0))
            res = minimize_scalar(f, x, tol, options={'disp':brent_verb})
            minimizer, fevals = res.x, res.nfev
        elif minsolver == 'golden':
            minimizer, fevals = golden_search(f, x[0], x[2], tol, verbose=verbose-1)
        # use golden search as backup if 'parabolic' or 'brent' fails to converge within bracket
        if minimizer is None or minimizer <= x[0] or minimizer >= x[2]:
            minimizer, fe = golden_search(f, x[0], x[2], tol, verbose=verbose-1)
            fevals += fe
    else:
        minimizer = x[1]
        fevals = 0
    if verbose > 0:
        print(f"min={minimizer:.2e}, fevals={fevals}")
    return minimizer, fevals

rho = (3-5**0.5)/2
def golden_search(f, a, b, tol=1e-15, maxiter=100, verbose=0):
    """Golden ratio minimization search"""
    if verbose > 0:
        print(f"golden search on [{a:.2e},{b:.2e}]")
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
            if verbose > 1:
                print(f"on left, [a,b] = [{a:.2e},{b:.2e}]")
        else:
            a = u
            h = b-a
            u = v
            v = b-rho*h
            fu = fv
            fv = f(v)
            if verbose > 1:
                print(f"on right, [a,b] = [{a:.2e},{b:.2e}]")
        fevals += 1
    if verbose > 0: print("converged")
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