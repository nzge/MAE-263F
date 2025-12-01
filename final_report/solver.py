import numpy as np
import importlib

import objective as obj
import contraction as contract
import parameters as param

importlib.reload(param)
importlib.reload(obj)
importlib.reload(contract)

   
def solver(worm, dt, maxTime, tol, maximum_iter, contractType = 'single_segment'):
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


def solver_with_predictor_corrector(worm, dt, maxTime, tol, maximum_iter, contractType='single_segment'):

    worm.q = worm.q0.copy()
    worm.u = worm.u0.copy()
    
    # Track which nodes are in ground contact (by node index, not DOF)
    # Initialize: no nodes in contact
    in_contact = np.zeros(worm.nv, dtype=bool)

    frames = []
    times = []
    t = np.arange(0, maxTime + dt, dt)
    
    for step in range(len(t)-1):
        t_val = t[step]
        t_new = t[step+1]
        
        if step % param.plotStep == 0:
            worm.plot(t_val)
        if step == len(t) - 2:
            worm.plot(t_new)

        # Get contraction force
        if contractType == 'single_segment':
            F_contract = contract.getContract_single_segment(worm, t_new)
        elif contractType == 'multiple_segments':
            F_contract = contract.getContract(worm, t_new)

        # === PREDICTOR STEP ===
        # Update freeIndex based on current contact state
        update_free_index(worm, in_contact)
        
        q_new, u_new, flag, f = obj.objfun(worm, dt, tol, maximum_iter, F_contract)

        if flag == -1:
            print("Maximum number of iterations reached.")
            break

        # === CHECK FOR CONTACT CHANGES ===
        contact_changed = False
        
        for node in range(worm.nv):
            y_dof = 2 * node + 1  # Y DOF index in flat array
            y_pos = q_new[node, 1]  # Y position (q_new is (nv, dim) shaped)
            
            # Condition 1: Free node penetrates ground → activate contact
            if not in_contact[node] and y_pos < worm.groundPosition:
                print(f"Node {node} hit ground at y={y_pos:.4f}")
                in_contact[node] = True
                contact_changed = True
            
            # Condition 2: Contact node wants to lift off → release
            # Release when f_y > 0 (reaction would need to pull down, impossible)
            elif in_contact[node] and f[y_dof] > 0:
                print(f"Node {node} lifting off, f_y={f[y_dof]:.4f}")
                in_contact[node] = False
                contact_changed = True

        # === CORRECTOR STEP ===
        if contact_changed:
            print("Contact changed - running corrector...")
            
            # Enforce ground position for newly contacted nodes
            for node in range(worm.nv):
                if in_contact[node]:
                    worm.q[node, 1] = worm.groundPosition
            
            # Update constraints and re-solve
            update_free_index(worm, in_contact)
            q_new, u_new, flag, f = obj.objfun(worm, dt, tol, maximum_iter, F_contract)

        # Store results
        frames.append(q_new.copy())
        times.append(t_new)
        
        worm.q = q_new
        worm.u = u_new
        worm.residuals = f.copy()

    return frames, times


def update_free_index(worm, in_contact):
    """Update freeIndex based on initial constraints + ground contact."""
    # Start with user-specified fixed DOFs
    fixed_dofs = set(worm.fixedIndex)
    
    # Add Y DOFs for nodes currently in ground contact
    for node in range(worm.nv):
        y_dof = worm.dim * node + 1
        if in_contact[node]:
            fixed_dofs.add(y_dof)
    
    worm.freeIndex = np.setdiff1d(np.arange(worm.ndof), list(fixed_dofs))