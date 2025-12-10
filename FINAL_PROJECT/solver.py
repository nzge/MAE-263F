import numpy as np
import importlib

import objective as obj
import contraction as contract
import parameters as param

importlib.reload(param)
importlib.reload(obj)
importlib.reload(contract)

   
def solver(worm, dt, maxTime, tol, maximum_iter, contractType = 'single_segment'):
    
    t = np.arange(0, maxTime + dt, dt)
    worm.q = worm.q0
    worm.u = worm.u0

    frames = []   # to store x configurations
    times = []    # corresponding times
    
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
    
    frames = []
    times = []
    t = np.arange(0, maxTime + dt, dt)

    worm.q = worm.q0.copy()
    worm.u = worm.u0.copy()
    
    # Track which nodes are in ground contact (by node index, not DOF)
    in_contact = np.zeros(worm.nv, dtype=bool) # Initialize: no nodes in contact
    total_mass = worm.m.sum()  # or use worm.m.sum() for main nodes only

    for step in range(len(t)-1):
        t_val = t[step]
        t_new = t[step+1]
        
        if step % param.plotStep == 0:
            worm.plot(t_val)
        if step == len(t) - 2:
            worm.plot(t_new)

        F_contract = np.zeros(worm.ndof) # Get contraction force
        if contractType == 'single_segment':
            F_contract = contract.getContract_single_segment(worm, t_new)
        elif contractType == 'multiple_segments':
            F_contract = contract.getContract(worm, t_new)

        # === PREDICTOR STEP ===
        update_free_index(worm, in_contact) # Update freeIndex based on current contact state
        
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
                #print(f"Node {node} hit ground at y={y_pos:.4f}")
                in_contact[node] = True
                contact_changed = True
            # Condition 2: Contact node wants to lift off → release
            elif in_contact[node] and f[y_dof] > 0: # Release when f_y > 0 (reaction would need to pull down, impossible)
                #print(f"Node {node} lifting off, f_y={f[y_dof]:.4f}")
                in_contact[node] = False
                contact_changed = True

        # === CORRECTOR STEP ===
        if contact_changed:
            #print("Contact changed - running corrector...")
            for node in range(worm.nv): # Enforce ground position for newly contacted nodes
                if in_contact[node]:
                    worm.q[node, 1] = worm.groundPosition
            update_free_index(worm, in_contact) # Update constraints and re-solve
            q_new, u_new, flag, f = obj.objfun(worm, dt, tol, maximum_iter, F_contract)

        # Store results
        frames.append(q_new.copy())
        times.append(t_new)
        
        # Calculate COM displacement (forward progress)
        q_old_flat = worm.q.flatten()
        q_new_flat = q_new.flatten()
        # COM position (x-component) before and after
        com_old = np.mean([q_old_flat[3*i] for i in range(0, worm.n+1)])  # x-positions
        com_new = np.mean([q_new_flat[3*i] for i in range(0, worm.n+1)])
        delta_com = com_new - com_old
        # Option 1: Keep actuator work but normalize properly
        work_step = F_contract.dot(q_new_flat - q_old_flat)

        if abs(delta_com) > 1e-6:
            #cot_step = work_step / (total_mass * delta_com)  # J/(kg·m)
            cot_step = work_step / ( delta_com)  # J/(kg·m)
        else:
            cot_step = 0.0

        worm.COT.append(cot_step)
        worm.cumulative_work = getattr(worm, 'cumulative_work', 0) + abs(work_step) # Also track cumulative values for total COT
        worm.cumulative_distance = getattr(worm, 'cumulative_distance', 0) + delta_com
        worm.com_xpos.append(com_new)
        #print('delta_com:', delta_com,'com_old:', com_old, 'com_new:', com_new)

        worm.q = q_new
        worm.u = u_new
        worm.residuals = f.copy()
        worm.residual_history.append(worm.residuals.copy())
    
    worm.total_COT = worm.cumulative_work / (worm.cumulative_distance)
    #worm.total_COT = worm.cumulative_work / (total_mass * worm.cumulative_distance)
    return frames, times


def update_free_index(worm, in_contact):
    """Update freeIndex based on initial constraints + ground contact."""
    fixed_dofs = set(worm.fixedIndex) # Start with user-specified fixed DOFs
    
    # Add Y DOFs for nodes currently in ground contact
    for node in range(worm.nv):
        y_dof = worm.dim * node + 1
        if in_contact[node]:
            fixed_dofs.add(y_dof)
    
    worm.freeIndex = np.setdiff1d(np.arange(worm.ndof), list(fixed_dofs))