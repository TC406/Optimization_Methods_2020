import  numpy as np
from numpy import asfarray

def wrap_function(function,args):
    ncalls = [0]
    if function is None:
        return ncalls,None


def _minimize_neldermead(func,x0,args=(),callback=None,
                         maxiter=400,
                         xatol=1e-4,fatol=1e-4):
    alpha = 1
    chi = 2
    beta = 0.5
    sigma = 0.5

    nonzdelt = 0.05
    zdelt = 0.00025

    x0 = asfarray(x0).flatten()

    # Create simplex
    N = len(x0)

    sim = np.zeros((N + 1,N),dtype=x0.dtype)
    sim[0] = x0
    # Generating simplex points
    # Generate them outside?
    for k in range(N):
        y = np.array(x0,copy=True)
        if y[k] != 0:
            y[k] = (1 + nonzdelt) * y[k]
        else:
            y[k] = zdelt
        sim[k + 1] = y

    allvecs = [sim,]

    one2np1 = list(range(1,N + 1))
    fsim = np.zeros((N + 1,),float)

    for k in range(N + 1):
        fsim[k] = func(sim[k])

    ind = np.argsort(fsim)
    fsim = np.take(fsim,ind,0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim,ind,0)

    iterations = 1

    while (iterations < maxiter):
        # If distance between coordinates less than xatol
        # and difference between function values is less than fatol
        if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
                np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
            break
        # xbar - center of gravity without worst point
        x_c = np.add.reduce(sim[:-1],0) / N
        # xr - Reflection.
        # sim[-1] - highest value
        x_r = (1 + alpha) * x_c - alpha * sim[-1]
        fxr = func(x_r)
        doshrink = 0
        # If new point xr is better than lowest value
        if fxr < fsim[0]:
            # Expansion case 4a
            x_e = (1 + alpha * chi) * x_c - alpha * chi * sim[-1]
            fxe = func(x_e)

            if fxe < fxr:
                sim[-1] = x_e
                fsim[-1] = fxe
            else:
                sim[-1] = x_r
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr 4b
            if fxr < fsim[-2]:
                sim[-1] = x_r
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2] 4c
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + beta * alpha) * x_c - beta * alpha * sim[-1]
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:  # 4d
                    # Perform an inside contraction
                    xcc = (1 - beta) * x_c + beta * sim[-1]
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:  # 6
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:  # 5
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = func(sim[j])

        ind = np.argsort(fsim)
        sim = np.take(sim,ind,0)
        fsim = np.take(fsim,ind,0)
        iterations += 1
        allvecs.append(sim)

    x = sim[0]
    fval = np.min(fsim)
    warnflag = 0

    # result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
    #                         status=warnflag, success=(warnflag == 0),
    #                         x=x, final_simplex=(sim, fsim))
    # if retall:
    #     result['allvecs'] = allvecs
    # return result
    return x, allvecs