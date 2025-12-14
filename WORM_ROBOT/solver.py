import numpy as np
import importlib
import matplotlib.pyplot as plt

import objective as obj
import contraction as contract
import parameters as param

importlib.reload(param)
importlib.reload(obj)
importlib.reload(contract)

   
def solver(worm, dt, maxTime, tol, maximum_iter, contraction_params=None):
    
    t = np.arange(0, maxTime + dt, dt)
    worm.q = worm.q0
    worm.u = worm.u0

    frames = []   # to store x configurations
    times = []    # corresponding times
    
    for k in range(len(t)-1):
        t_val = t[k]
        t_new = t[k+1]
        
        # Default contraction parameters if not provided
        if contraction_params is None:
            contraction_params = {
                'contraction_type': 'single_segment',
                'wave_type': 'traveling',
                'T_wave': 2.0,
                'wavelength': 1.0
            }
        
        # Get contraction force based on parameters
        if contraction_params.get('contraction_type', 'single_segment') == 'single_segment':
            F_contract = contract.getContract_single_segment(worm, t_new)
        elif contraction_params.get('contraction_type') == 'multiple_segments':
            F_contract = contract.getContract(
                worm, 
                t=t_new,
                T_wave=contraction_params.get('T_wave', 2.0),
                wavelength=contraction_params.get('wavelength', 1.0),
                wave_type=contraction_params.get('wave_type', 'traveling')
            )
        
        # if k % param.plotStep == 0:
        #     worm.plot(t_val)
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


def solver_with_predictor_corrector(worm, dt, maxTime, tol, maximum_iter, contraction_params=None):
    
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
        
        # if step % param.plotStep == 0:
        #     worm.plot(t_val)
        if step == len(t) - 2:
            worm.plot(t_new)

        F_contract = np.zeros(worm.ndof) # Get contraction force
        
        # Default contraction parameters if not provided
        if contraction_params is None:
            contraction_params = {
                'contraction_type': 'multiple_segments',
                'wave_type': 'traveling',
                'T_wave': 2.0,
                'wavelength': 1.0
            }
        
        # Get contraction force based on parameters
        if contraction_params.get('contraction_type', 'single_segment') == 'single_segment':
            F_contract = contract.getContract_single_segment(worm, t_new)
        elif contraction_params.get('contraction_type') == 'multiple_segments':
            F_contract = contract.getContract(
                worm, 
                t=t_new,
                T_wave=contraction_params.get('T_wave', 2.0),
                wavelength=contraction_params.get('wavelength', 1.0),
                wave_type=contraction_params.get('wave_type', 'traveling')
            )

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
    
    cot_cumsum = np.cumsum(np.abs(worm.COT))

    metrics = {
      "COT": worm.COT.copy(),
      "cot_cumsum": cot_cumsum.copy(),
      "total_COT": worm.total_COT.copy(),
      "com_xpos": worm.com_xpos.copy(),
      "cumulative_work": worm.cumulative_work.copy(),
      "cumulative_distance": worm.cumulative_distance.copy(),
    }

    print('solver_with_predictor_corrector completed')
    return frames, times, metrics

def plot_sim_results(
    metrics,
    frames,
    times,
    *,
    ax_first_node=None,
    ax_last_node=None,
    ax_com=None,
    ax_cot_inst=None,
    ax_cot_cum=None,
    label_suffix="",
    worm_name="worm",
    main_node_indices=None,
):
    """
    Plot simulation results using metrics returned by solver_with_predictor_corrector.

    Parameters
    ----------
    metrics : dict
        Expected keys: cot_steps, cot_cumsum (optional, else computed),
        com_xpos (optional), total_COT (optional), cumulative_work (optional),
        cumulative_distance (optional).
    frames : list of ndarray
        Shape (nv, dim) per frame.
    times : list/array
        Time stamps.
    Optional axes allow overlaying multiple runs on shared plots.
    """

    # Extract position data
    first_node_xpos = [frame[0, 0] if frame.ndim == 2 else frame[0][0] for frame in frames]

    last_node_xpos = [
        (frame[-1, 0] if frame.ndim == 2 else frame[-1][0]) - param.length for frame in frames
    ]

    # COM position: prefer provided metrics; else compute average x of provided main nodes or all nodes
    if metrics.get("com_xpos") is not None:
        com_xpos = metrics["com_xpos"]
    else:
        com_xpos = []
        for frame in frames:
            if main_node_indices is not None and len(main_node_indices) > 0:
                xs = [frame[idx, 0] if frame.ndim == 2 else frame[idx][0] for idx in main_node_indices]
            else:
                xs = frame[:, 0] if frame.ndim == 2 else [node[0] for node in frame]
            com_xpos.append(np.mean(xs))

    # COT data
    cot_steps = np.array(metrics.get("COT", []))
    cot_time = np.array(times)

    # Guard against mismatched lengths
    if len(cot_steps) != len(cot_time):
        min_len = min(len(cot_steps), len(cot_time))
        if min_len == 0:
            cot_steps = np.array([])
            cot_time = np.array([])
        else:
            print(
                f"[plot_sim_results] Warning: len(cot_steps)={len(cot_steps)} "
                f"!= len(times)={len(cot_time)}; truncating to {min_len}."
            )
            cot_steps = cot_steps[:min_len]
            cot_time = cot_time[:min_len]

    cot_cumsum = np.array(metrics.get("cot_cumsum")) if metrics.get("cot_cumsum") is not None else np.cumsum(
        np.abs(cot_steps)
    )

    # Helper to create/use axes
    def get_or_create_ax(ax, title, ylabel):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure
        if not ax.get_title():
            ax.set_title(title)
        if not ax.get_xlabel():
            ax.set_xlabel("Time (s)")
        if not ax.get_ylabel():
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return ax

    # First node
    ax_first = get_or_create_ax(ax_first_node, "First Node Position vs Time", "X Position (m)")
    ax_first.plot(times, first_node_xpos, label=f"{worm_name}: First Node{label_suffix}", linestyle="-", marker="s", markersize=2)
    ax_first.legend()

    # Last node
    ax_last = get_or_create_ax(ax_last_node, "Last Node Position vs Time", "X Position (m)")
    ax_last.plot(times, last_node_xpos, label=f"{worm_name}: Last Node - Worm Length{label_suffix}", linestyle="-", marker="o", markersize=2)
    ax_last.legend()

    # COM
    ax_com_plot = get_or_create_ax(ax_com, "Center of Mass Position vs Time", "X Position (m)")
    ax_com_plot.plot(times, com_xpos, label=f"{worm_name}: COM{label_suffix}", linestyle="-", marker="^", markersize=2)
    ax_com_plot.legend()

    # Instantaneous COT
    ax_cot_i = get_or_create_ax(ax_cot_inst, "Instantaneous COT vs Time", "COT")
    ax_cot_i.plot(cot_time, cot_steps, label=f"{worm_name}: Instantaneous COT{label_suffix}", alpha=0.7, linestyle="-", marker="o", markersize=2)
    ax_cot_i.legend()

    # Cumulative COT
    ax_cot_c = get_or_create_ax(ax_cot_cum, "Cumulative COT vs Time", "COT")
    ax_cot_c.plot(cot_time, cot_cumsum, label=f"{worm_name}: Cumulative COT (∑|COT|){label_suffix}", linestyle="-", linewidth=2)
    ax_cot_c.legend()

    return {
        "first_node": ax_first,
        "last_node": ax_last,
        "com": ax_com_plot,
        "cot_inst": ax_cot_i,
        "cot_cum": ax_cot_c,
    }


def update_free_index(worm, in_contact):
    """Update freeIndex based on initial constraints + ground contact."""
    fixed_dofs = set(worm.fixedIndex) # Start with user-specified fixed DOFs
    
    # Add Y DOFs for nodes currently in ground contact
    for node in range(worm.nv):
        y_dof = worm.dim * node + 1
        if in_contact[node]:
            fixed_dofs.add(y_dof)
    
    worm.freeIndex = np.setdiff1d(np.arange(worm.ndof), list(fixed_dofs))