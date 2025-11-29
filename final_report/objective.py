import numpy as np
import forces as force

def objfun(worm, q_old, u_old, dt, tol, maximum_iter, contract):

  free_index = worm.freeIndex
  q_new = q_old.copy() # Guess solution

  # Newton Raphson
  iter_count = 0 # number of iterations
  error = tol * 10 # error
  flag = 1 # if flag = 1, it is a good solution

  while error > tol:
    
    # Compute force and Jacobian
    f,J = force.getForceJacobian(q_new, q_old, u_old, dt, worm, contract)

    f_free = f[free_index]
    J_free = J[np.ix_(free_index, free_index)]

    # Newton's update (all DOFs are FREE)
    dq_free = np.linalg.solve(J_free, f_free)
    q_new[free_index] = q_new[free_index] - dq_free

    # Get the error
    error = np.linalg.norm(f_free)

    # Update the iteration number
    iter_count += 1
    if iter_count > maximum_iter:
      flag = -1 # Return with an error signal
      print("Maximum number of iterations reached.")
      return q_new, flag

    u_new = (q_new - q_old) / dt # Velocity
  return q_new, u_new, flag, f


