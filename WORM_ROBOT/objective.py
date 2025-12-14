import numpy as np
import pandas as pd
from IPython.display import display
import forces as force

def objfun(worm, dt, tol, maximum_iter, contract):

  free_index = worm.freeIndex

  q_old = worm.q.flatten()
  u_old = worm.u.flatten()
  q_new = q_old.copy()
  #print(q_old)

  # Newton Raphson
  iter_count = 0
  error = tol * 10
  flag = 1

  while error > tol:
    
    # Compute force and Jacobian
    f, J = force.getForceJacobian(q_new, q_old, u_old, dt, worm, contract)

    f_free = f[free_index]
    J_free = J[np.ix_(free_index, free_index)]

    # Newton's update
    dq_free = np.linalg.solve(J_free, f_free)
    
    # === PRINT DQ TABLE ===
    dq_full = np.zeros(worm.ndof)
    dq_full[free_index] = dq_free
    print_dq(worm, dq_full, iter_count, verbose=False)
    # ======================
    
    q_new[free_index] = q_new[free_index] - dq_free
    u_new = (q_new - q_old) / dt

    error = np.linalg.norm(f_free)
    iter_count += 1
    
    if iter_count > maximum_iter:
      flag = -1
      print("Maximum number of iterations reached.")
      return q_new.reshape(worm.nv, worm.dim), u_new.reshape(worm.nv, worm.dim), flag, f

  return q_new.reshape(worm.nv, worm.dim), u_new.reshape(worm.nv, worm.dim), flag, f


def print_dq(worm, dq, iter_count, verbose=True):
    """Print dq as a pandas table with node X and Y."""
    if not verbose:
        return

    nv = worm.nv
    
    # Reshape dq into (nv, dim) for cleaner display
    dq_reshaped = dq.reshape(nv, worm.dim)
    
    # Node labels
    labels = []
    for i in range(nv):
        node_type = "main" if i % 3 == 0 else ("top" if i % 3 == 1 else "bot")
        labels.append(f"N{i} ({node_type})")
    
    df = pd.DataFrame(dq_reshaped, index=labels, columns=['dq_X', 'dq_Y'])
    
    # Filter to non-zero only
    mask = (df.abs() > 1e-12).any(axis=1)
    df_filtered = df[mask]
    
    pd.set_option('display.float_format', '{:.8f}'.format)
    
    print(f"\n--- dq (Newton step {iter_count}) ---")
    display(df_filtered)
    print(f"||dq|| = {np.linalg.norm(dq):.2e}\n")


