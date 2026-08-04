import numpy as np
from scipy.optimize import brentq, minimize_scalar
from .core import EigensolverFailure

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
        x_tmp, y_tmp = x_tmp2, y_tmp2
    if end < len(x)-1:
        # runs that don't have the trailing ghost point
        # append x,y from following interval endpoint
        x_tmp2 = np.concatenate((x_tmp,[x[end+1]]))
        y_tmp2 = np.empty((2,len(x_tmp)+1))
        y_tmp2[:,:-1] = y_tmp
        y_tmp2[:,-1] = y[:,end+1]
        x_tmp, y_tmp = x_tmp2, y_tmp2
    return x_tmp, y_tmp, fevals

def bracket_mins(f, x, y, xtol=1e-8, shrink=2, nrecurse=0, verbose=0,
                 max_recurse=30, noise_factor=4.0, max_minima=None):
    """Bracket the minima of f(x)[0] using a gridsearch. Returns a list of brackets which (hopefully!) each 
    contain a single local minimum of the first component of f. Uses the local minima 
    of f(x)[1] to guide refinement.

    Refinement stops on two conditions, and the distinction matters:

    ``noise_factor`` is the real control. A run whose objective varies by no
    more than ``noise_factor`` times its own minimum has nothing left to
    resolve -- that is what a tension curve looks like once it is down to
    roundoff wiggle, where every "minimum" is spurious and subdividing only
    manufactures more of them. A genuine well varies by orders of magnitude
    above its floor and keeps refining.

    ``max_recurse`` is only a backstop against true non-termination, and is
    deliberately generous. A tight depth cap is the wrong instrument: it
    throttles well-resolved problems to protect against noisy ones, and it has
    a bad regime in the middle. On ``rect(1, 1.00001)``, whose eigenvalues come
    in pairs 1.2e-5 apart, capping at 8 leaves the pairs unresolved (4.8 true
    digits), 12 resolves them *partially* and misaligns the list (0.2 digits),
    and 16 or unbounded gives 5.9. Depth is a proxy for "is refinement still
    informative"; the noise test asks that question directly, per run rather
    than globally.
    """
    tabs = min(nrecurse,5)*"\t" # tab spacing for verbose mode
    if verbose > 0:
        print(tabs+f"bracket_mins on [{x[1]:.5e},{x[-2]:.5e}] (len={x[-2]-x[1]:.2e}, npts={len(x)})")

    # get discrete local min indices for y1 and y2
    y0_min_idx = discrete_locmin_idx(y[0])
    y1_min_idx = discrete_locmin_idx(y[1])

    if verbose > 0:
        print("number of local minima:",len(y0_min_idx))

    # Abort on an objective that is numerical noise rather than structure.
    #
    # This is the *detector* for an ill-posed instance. Depth is not: an
    # ill-posed problem does not recurse deeper than a well-posed one with
    # repeated or near-repeated eigenvalues -- it branches wider, from many
    # spurious minima. So count minima, and count them at every level, since
    # noise usually appears only in part of the window and only once the grid
    # is fine enough to resolve the wiggle.
    #
    # `max_minima` is the calibrated form: the caller knows how many minima to
    # expect (for MPS, the eigenvalue count in the window, from asymp.weyl_est)
    # and passes a multiple of it. With `None` we fall back to the original
    # grid-fraction heuristic, which scales with sampling density rather than
    # with the problem -- kept only so existing callers are unaffected.
    if max_minima is not None:
        if len(y0_min_idx) > max_minima:
            raise EigensolverFailure(
                f"f has too many local minima ({len(y0_min_idx)} > {max_minima}) "
                f"on [{x[1]:.6g}, {x[-2]:.6g}] at recursion depth {nrecurse}: "
                f"the objective is noise, not structure")
    elif nrecurse == 0 and len(y0_min_idx) > len(x)/3:
        raise EigensolverFailure("f has too many local minima")

    # Optional depth cap. Unbounded recursion here is a real hazard: the
    # "too many local minima" check above fires only at nrecurse==0, so once
    # sigma is numerical noise over part of the window, every level finds
    # spurious minima, flags them all for refinement, and each flagged run
    # spawns another level whose grid is `shrink` times finer. The cost
    # compounds across levels *and* across flagged runs. Observed on
    # iso_right_tri: 11 live levels of recursion, driving a 16GB machine to a
    # 59.8GB footprint and 40GB of swap.
    #
    # Default is 8, i.e. ~256x the initial grid spacing -- far finer than any
    # sensible xtol needs, and empirically free: eq_tri gives identical
    # eigenvalues (13.6 certified / 14.5 true) with the cap and without. Pass
    # max_recurse=None to restore unbounded recursion. At the cap we keep the
    # intervals we have as brackets rather than discarding them, so the caller
    # still gets candidates to polish.
    if max_recurse is not None and nrecurse >= max_recurse:
        out = [(x[i-1:i+2], y[0, i-1:i+2]) for i in y0_min_idx
               if 0 < i < len(x) - 1]
        if verbose > 0:
            print(tabs + f"depth cap {max_recurse} reached, "
                         f"returning {len(out)} unrefined brackets")
        return out, 0

    # get refinement flags (based on proximity of y1_mins/y2_mins relative to y0_mins)
    refine_flag = flag_refinement_intervals(len(x)-1, y0_min_idx, y1_min_idx)

    # brackets that don't need refinement
    brackets = []
    for idx in y0_min_idx:
        if not refine_flag[idx]:
            brackets.append((x[idx-1:idx+2],y[0,idx-1:idx+2]))
            if verbose > 1:
                print(tabs+f"+[{x[idx-1]:.5e},{x[idx+1]:.5e}]")
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
                    print(tabs+f"+[{x[start]:.5e},{x[end]:.5e}] (below xtol)")
            else:
                x_tmp, y_tmp, fe = fill_refinement(f, x, y, start, end, shrink, verbose)
                fevals += fe
                if verbose > 1:
                    print(tabs+f"refine on [{x[start]:.5e},{x[end]:.5e}], shrink={shrink}")
                # check for flat objective at this scale
                mindiff = np.abs(np.diff(y_tmp[0])).min()

                # Noise-limited? A run whose spread is no more than
                # `noise_factor` times its own floor carries no structure left
                # to resolve: this is roundoff wiggle, where every apparent
                # minimum is spurious and subdividing manufactures more of
                # them. A genuine well sits orders of magnitude below its own
                # spread and fails this test, so it keeps refining. Asked per
                # run, so a domain can be noise-limited in one part of the
                # window and still resolve cleanly elsewhere.
                # Spread relative to the floor. A genuine well spans orders of
                # magnitude between its surroundings and its minimum, so
                # ptp >> floor and it keeps refining -- this holds even when
                # the run brackets *two* close minima, because the tension
                # still rises well above the floor between them. Roundoff
                # wiggle has ptp comparable to the floor itself.
                #
                # Comparing the floor to the run's *median* instead was tried
                # and is too aggressive: once a run narrows onto a tight
                # cluster most of its values are small, so the median collapses
                # toward the floor and real pairs get declared noise
                # (rect_near_deg_1e5 fell from 11 resolved eigenvalues to 7).
                y_run = y_tmp[0]
                y_floor = y_run.min()
                noisy = (noise_factor is not None
                         and np.ptp(y_run) <= noise_factor * y_floor)

                if mindiff == 0 or noisy:
                    half_idx = int((end+start)/2)
                    brackets.append((x[[start,half_idx,end]],y[0,[start,half_idx,end]]))
                    if verbose > 1:
                        why = 'flat objective' if mindiff == 0 else 'noise-limited'
                        print(tabs+f"+[{x[start]:.5e},{x[end]:.5e}] ({why})")

                # recurse, extending the list of brackets and incrementing the function evaluations
                else:
                    if verbose > 0:
                        print(tabs+"recursing...")
                    bracks, fe = bracket_mins(f, x_tmp, y_tmp, xtol, shrink,
                                              nrecurse+1, verbose,
                                              max_recurse=max_recurse,
                                              noise_factor=noise_factor,
                                              max_minima=max_minima)
                    brackets += bracks
                    fevals += fe
    if verbose > 0 and nrecurse == 0:
        print(f"found {len(brackets)} brackets, fevals={fevals}")
    return brackets, fevals

def minimize_on_bracket(f, bracket, xtol, minsolver='parabolic', verbose=0):
    x, y = bracket # unpack bracket
    if verbose > 0:
        print(f"minimizing on [{x[0]:.5e},{x[2]:.5e}]")
    tol = xtol*x[0] # get absolute tolerance from relative

    # only minimize further if bracket is at least width tol
    if x[2]-x[0] > tol:
        # A degenerate bracket (interior point coincident with an endpoint) can
        # come out of bracket_mins where sigma is flat to machine precision over
        # a range -- which is exactly what happens at a high-multiplicity
        # eigenvalue. parabolic_iter_min requires x strictly increasing and
        # raises otherwise, so route these straight to the golden search that is
        # already this function's fallback: it needs only the two endpoints.
        if not (x[0] < x[1] < x[2]):
            minimizer, fevals = golden_search(f, x[0], x[2], tol,
                                              verbose=verbose-1)
        elif minsolver == 'parabolic':
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
        print(f"min={minimizer:.5e}, fevals={fevals}")
    return minimizer, fevals

rho = (3-5**0.5)/2
def golden_search(f, a, b, tol=1e-15, maxiter=100, verbose=0):
    """Golden ratio minimization search"""
    if verbose > 0:
        print(f"golden search on [{a:.5e},{b:.5e}]")
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