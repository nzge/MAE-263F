import numpy as np
import importlib

import objective as obj
import contraction as contract
import parameters as param

importlib.reload(param)
importlib.reload(obj)
importlib.reload(contract)

   
def solver(worm, dt, maxTime, tol, maximum_iter):
    worm.q = worm.q0
    worm.u = worm.u0

    frames = []   # to store x configurations
    times = []    # corresponding times

    t = np.arange(0, maxTime + dt, dt)
    
    for k in range(len(t)-1):
        t_val = t[k]
        t_new = t[k+1]
        
        F_contract = contract.getContract_single_segment(worm, t_new)
        
        if k % param.plotStep == 0:
            worm.plot(t_val)
        if k == len(t) - 2:
            worm.plot(t_new)

        q_new, u_new, flag, f = obj.objfun(worm, dt, tol, maximum_iter, F_contract)

        if flag == -1:
            print("Maximum number of iterations reached.")
            break

        frames.append(q_new.copy())
        times.append(t_new)
        
        worm.q = q_new
        worm.u = u_new

    return frames, times

def solver_with_predictor_corrector(worm, dt, maxTime, tol, maximum_iter, contractType = 'single_segment'):

    worm.q = worm.q0
    worm.u = worm.u0

    frames = []   # to store x configurations
    times = []    # corresponding times

    t = np.arange(0, maxTime + dt, dt)
    
    for k in range(len(t)-1):
        t_val = t[k]
        t_new = t[k+1]
        
        if contractType == 'single_segment':
            F_contract = contract.getContract_single_segment(worm, t_new)
        elif contractType == 'multiple_segments':
            F_contract = contract.getContract(worm, t_new)
        
        if k % param.plotStep == 0:
            worm.plot(t_val)
            # print('Force contract: ', contract.reshape(worm.nv, worm.dim))
        if k == len(t) - 2:
            worm.plot(t_new)

        q_new, u_new, flag, f = obj.objfun(worm, dt, tol, maximum_iter, F_contract)
        print('force residuals: ', f)

        if flag == -1:
            print("Maximum number of iterations reached.")
            break

        # Check if corrector is needed
        needCorrector = False
        for k in range(worm.nv-1):
            # Condition 1: if free node, check if it fell under ground?
            if worm.isFixed[k] == 0 and q_new[worm.freeIndex[2*k+1]] < worm.groundPosition:
                needCorrector = True
                worm.isFixed[k] = 1
                worm.q[2*k + 1] = worm.groundPosition
            # Condition 2: if k-th node is fixed but has "wrong" (inward normal) reaction force
            elif worm.isFixed[k] == 1 and f[2*k+1] < 0:
                needCorrector = True
                worm.isFixed[k] = 0

        # Corrector Step
        if needCorrector:
            q_new, u_new, error, reactionForce = obj.objfun(worm, dt, tol, maximum_iter, F_contract)
            print('Entering corrector step')
            print('error: ', error)
            print('force residuals: ', reactionForce) 

				# compiling animation frames
        frames.append(q_new.copy())
        times.append(t_new)
				# updating worm state
        worm.q = q_new
        worm.u = u_new

    return frames, times

def getFreeIndex(worm):
  # isFixed is a 0 or 1 vector of size nv
  # free_index is the output of size ndof = 2 * nv
  worm.nv = len(worm.isFixed) # Number of vertices
  all_DOFs = np.zeros(worm.ndof) # Set of all DOFs -- all DOFs are free

  # Hard code the clamp condition
  all_DOFs[0:4] = 1 # Fix the x-coordinate (left wall)

  for k in range(worm.nv):
    if worm.isFixed[k] == 1:
      # all_DOFs[2*k] = 1
      all_DOFs[worm.dim*k+1] = 1 # Fix the y-coordinate (ground)
  free_index = np.where(all_DOFs == 0)[0]
  return free_index