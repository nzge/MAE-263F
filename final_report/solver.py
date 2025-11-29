import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import objective as obj

from contraction import getFexternal

dt = 0.1 # Time step size
maxTime = 10   # total time of simulation

def solver(worm, q_old, u_old, dt, maxTime, tol, maximum_iter):

    # free indices
    fixed_DOF = np.array([ 1, 5])  # Fix x and y of node 0 and node 4
    free_DOF = np.array([i for i in range(worm.ndof) if i not in fixed_DOF])

    plot_indices = np.array([0, 0.1, 1, 2,3,4,5,6,7,8,9, 10])

    frames = []   # to store x configurations
    times = []    # corresponding times

    t = np.arange(0, maxTime + dt, dt)
    
    for k in range(len(t)-1):
        t_val = t[k]
        t_new = t[k+1]
        
        contract = getFexternal(t_new, worm.ndof)  
        q_new, u_new, flag, f = obj.objfun(worm, q_old, u_old, dt, tol, maximum_iter, contract)


        if k == len(t) - 2:
            worm.plot()
        
        worm.update_internal_state(q_new)
        frames.append(q_new.copy())
        times.append(t_new)
        
        q_old = q_new
        u_old = u_new

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
      all_DOFs[2*k+1] = 1 # Fix the y-coordinate (ground)
  free_index = np.where(all_DOFs == 0)[0]
  return free_index