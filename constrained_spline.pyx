from cvxopt import lapack, solvers, matrix, mul
from cvxopt.modeling import op, variable, max, sum
import numpy as np

def const_spline(t,y,lbound=-.4,rbound=.4,nknots=5,clamps=True,order=4,gamma=0,force_convex=True):
    """ Adapted from example on CVXOPT site -- Changing as little as possible 

        t - X values of function to fit spline to
        y - Y values of function
        lbound,rbound - Boundary X-values where you will interpolate
        nknots - Number of knot points
        clamps - Spline is clamped into linear function at end of boundaries
        order - Order of Spline (4 = cubic, 3 = quadratic, else NYI)
        gamma - Penalty smoothing term (0 is no penalty)
        force_convex - Forces entire surface to be convex (Dangerous perhaps)

        Returns list of coefficients, knot points, and lastly a function
        pointer which you should use to spline new X points.(Or the current ones!)
    """

    m = len(t)
    if clamps: #give em the clamps!
        tweak = (rbound-lbound)/5./4.
        knots = np.linspace(lbound+tweak,rbound-tweak,nknots+2,endpoint=True)
        nknots+=2
    else:
        knots = np.linspace(lbound,rbound,nknots+1,endpoint=False)[1:]
    n = order * (nknots+1)


    #get array indices by knot locations
    Is = []
    Is.append( [ k for k in range(m) if lbound  <= t[k] < knots[0] ])
    prev_knot = knots[0]
    for i,knot in enumerate(knots[1:]):
        Is.append( [ k for k in range(m) if prev_knot  <= t[k] < knot ])
        prev_knot = knot
    Is.append( [ k for k in range(m) if prev_knot  <= t[k] <= rbound ])
    
    # build system matrix
    A = matrix(0.0, (m,n))
    for k in range(order):
        offset = 0
        for I in Is:
            A[I,k+offset] = t[I]**k
            offset+=order
    
    G2 = matrix(0.0, (nknots,n))
    
    if order == 4: #aka cubic spline (4 terms brah)
        
        num_constraints = 3 * nknots
        G = matrix(0.0, (num_constraints,n))
        
        #1st order
        for i,knot in enumerate(knots):
            G[i,list(range(order*i,order*i+8))] = 1.0,knot,knot**2,knot**3,-1.0,-knot,-knot**2,-knot**3

        #2nd order
        for i,knot in enumerate(knots):
            G[i+nknots,list(range(order*i,order*i+8))] = 0, 1.0,2*knot,3*knot**2, 0, -1.0, -2*knot,-3*knot**2

        #3rd order
        for i,knot in enumerate(knots):
            G[i+nknots*2,list(range(order*i,order*i+8))] = 0, 0, 2, 6*knot,0,0,-2,-6*knot

        # inequality constraints
        for i,knot in enumerate(knots):
            G2[i,list(range(order*i,order*i+8))] = 0, 0, 2, 6*knot,0,0,0,0
            
    elif order == 3: #local quadtraic polynomials (3 terms)
        
        num_constraints = 2 * nknots
        G = matrix(0.0, (num_constraints,n))
        
        #1st order
        for i,knot in enumerate(knots):
            G[i,list(range(order*i,order*i+6))] = 1.0,knot,knot**2,-1.0,-knot,-knot**2

        #2nd order
        for i,knot in enumerate(knots):
            G[i+nknots,list(range(order*i,order*i+6))] = 0, 1.0,2*knot, 0, -1.0, -2*knot

        for i,knot in enumerate(knots):
            G2[i,list(range(order*i,order*i+6))] = 0, 0, 2,0,0,0
    else:
        raise 'Not Yet Implemented'


    #set up constraints
    constraints = []
    xcheb = variable(n)
    constraints.append( G*xcheb == 0)
    
    if force_convex:
        constraints.append( G2*xcheb >= 0)
    if clamps and order == 4:
        constraints +=  [xcheb[2] == 0,xcheb[3] == 0,xcheb[-2] == 0,xcheb[-1] == 0]
    elif clamps and order == 3:
        constraints +=  [xcheb[2] == 0,xcheb[-1] == 0]
    if gamma>0:
        op( sum(abs(A*xcheb - y)) + gamma*sum(abs(xcheb[::order])), constraints).solve()
    else:
        op( sum(abs(A*xcheb - y)) , constraints).solve()
    
    #function pointer that gets returned for spline evaluation (yes this code is awful but should be cythoned anyways)
    def f(ts,knots=knots,xcheb=xcheb.value,order=order):
        ts = np.atleast_1d(ts)
        nopts,lbound,rbound = len(ts),np.min(ts),np.max(ts)
        Is = []
        Is.append( [ k for k in range(nopts) if lbound  <= ts[k] < knots[0] ])
        prev_knot = knots[0]
        for i,knot in enumerate(knots[1:]):
            Is.append( [ k for k in range(nopts) if prev_knot  <= ts[k] < knot ])
            prev_knot = knot
        Is.append( [ k for k in range(nopts) if prev_knot  <= ts[k] <= rbound ])
        ycheb = matrix(0.0, (nopts,1))
        offset = 0
        for I in Is:
            ycheb[I] = sum( xcheb[k+offset]*ts[I]**k for k in range(order) )
            offset+=order
        return np.asarray(ycheb)
    
    #return coefficients, knot locations, and pointer to f()
    return xcheb.value,knots,f