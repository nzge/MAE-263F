import numpy as np
import pandas as pd
from IPython.display import display # Only for iPython
import importlib

import helper as helper
import parameters as param
importlib.reload(param)
importlib.reload(helper)

def getForceJacobian(q_new, q_old, u_old, dt, worm, contractionEngine):
  
  # Gravity
  #W = worm.W

  # Inertia
  F_inertia = worm.m/dt * ((q_new - q_old) / dt - u_old)
  J_inertia = worm.mMat / dt ** 2

  # Elastic forces: Stretching and Bending
  Fs, Js = getFs(worm, q_new)
  Fb, Jb = getFb(worm, q_new)
  Fb, Jb = 0, 0
  F_elastic = Fs + Fb
  J_elastic = Js + Jb

  # Viscous force
  #C = worm.c
  #Fv = - C @ ( q_new - q_old ) / dt
  #Jv = - C / dt
  Fv = 0
  Jv = 0

  # Ground contact and friction
  F_friction = getFriction(worm, q_old, dt)

  # Equations of motion
  f = F_inertia - F_elastic - Fv - F_friction - contractionEngine
  J = J_inertia - J_elastic - Jv  # Friction Jacobian often ignored

  # Pretty-print force summary (set verbose=False to disable, show_all=True to see all DOFs)
  print_force_summary(worm, q_new, F_inertia, Fs, Fb, contractionEngine, F_friction, f, verbose=False, show_all=False)
  return f, J

def print_force_summary(worm, q_new, F_inertia, Fs, Fb, F_contract, F_friction, f_residual, verbose=True, show_all=False):
    
    if not verbose:
        return
    
    nv = worm.nv
    labels = []   # Build row labels: Node 0 X, Node 0 Y, Node 1 X, ...
    for i in range(nv):
        node_type = "main" if i % 3 == 0 else ("top" if i % 3 == 1 else "bot")
        labels.append(f"N{i} ({node_type}) X")
        labels.append(f"N{i} ({node_type}) Y")
    
    # Mark fixed DOFs
    fixed_marker = np.array(['free'] * worm.ndof)
    fixed_marker[worm.fixedIndex] = 'FIXED'
    
    df = pd.DataFrame({
        'q_new': q_new,
        'F_inertia': F_inertia,
        'F_stretch': Fs,
        'F_bend': Fb,
        'F_contract': F_contract,
        'F_friction': F_friction,
        'f_residual': f_residual,
        'status': fixed_marker
    }, index=labels)
    
    # Only show rows with non-zero values (for cleaner output) unless show_all=True
    if not show_all:
        mask = (df[['F_inertia', 'F_stretch', 'F_bend', 'F_contract', 'F_friction', 'f_residual']].abs() > 1e-10).any(axis=1)
        df_filtered = df[mask]
    else:
        df_filtered = df
    
    pd.set_option('display.float_format', '{:.6f}'.format)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    
    print("\n" + "="*80)
    print("FORCE & POSITION SUMMARY" + (" (non-zero DOFs only)" if not show_all else " (all DOFs)"))
    print("="*80)
    display(df_filtered)
    print(f"\nResidual norm: {np.linalg.norm(f_residual):.2e}")
    print(f"Free DOF residual norm: {np.linalg.norm(f_residual[worm.freeIndex]):.2e}")
    print("="*80 + "\n")


#-----------------------------------------------------------------------#

def getFriction(worm, q_old, dt):
    """
    Anisotropic Coulomb friction for directional locomotion.
    
    For RIGHTWARD motion:
    - LOW friction when moving right (easy to slide forward)
    - HIGH friction when moving left (anchor/grip to push off)
    """
    F_friction = np.zeros(worm.ndof)
    
    # Anisotropic friction coefficients
    # Ratio > 1 means harder to move backward than forward
    mu_forward = param.mu_sliding * 0.3   # Moving RIGHT: low friction (slide easy)
    mu_backward = param.mu_sliding * 2.0  # Moving LEFT: high friction (grip/anchor)
    
    # Only apply friction to nodes in ground contact
    for node in range(worm.nv):
        y_dof = worm.dim * node + 1
        x_dof = worm.dim * node
        
        # Check if this node's Y is constrained (in contact)
        if y_dof in worm.fixedIndex:
            # Use stored reaction force from previous solve
            if hasattr(worm, 'reaction_forces') and y_dof < len(worm.reaction_forces):
                N = abs(worm.reaction_forces[y_dof])  # Normal force magnitude
            else:
                N = worm.m[y_dof] * 9.8  # Fallback: use weight as estimate
            
            # Velocity in X direction
            if hasattr(worm.u, 'shape') and len(worm.u.shape) > 1:
                v_x = worm.u[node, 0]
            else:
                v_x = worm.u[x_dof] if x_dof < len(worm.u) else 0
            
            if N > 0:
                if v_x > 1e-6:  # Moving RIGHT (forward) - low friction
                    F_friction[x_dof] = -mu_forward * N * np.sign(v_x)
                elif v_x < -1e-6:  # Moving LEFT (backward) - high friction
                    F_friction[x_dof] = -mu_backward * N * np.sign(v_x)
                # else: static friction (v_x ≈ 0) - could add static friction here
    return F_friction

#-----------------------------------------------------------------------#
# Spring
def getFs(worm, q):
  """Compute spring forces at configuration q (flattened DOF vector)"""
  f_spring = np.zeros(worm.ndof)
  J_spring = np.zeros((worm.ndof, worm.ndof))
  
  for i in range(worm.ne):
    ind = worm.springs[i]
    # Read from q parameter, not worm.q
    xi = q[2*ind[0]]
    yi = q[2*ind[0]+1]
    xj = q[2*ind[1]]
    yj = q[2*ind[1]+1]
    indices = [2*ind[0], 2*ind[0]+1, 2*ind[1], 2*ind[1]+1]
    stiffness = worm.spring_k[i]

    f_spring[indices] += gradEs(xi, yi, xj, yj, worm.spring_l0[i], stiffness)
    J_spring[np.ix_(indices,indices)] += hessEs(xi, yi, xj, yj, worm.spring_l0[i], stiffness)
  return f_spring, J_spring  # OUTSIDE the loop!

def gradEs(xk, yk, xkp1, ykp1, l_k, k):
    """
    Calculate the gradient of the stretching energy with respect to the coordinates.

    Args:
    - xk (float): x coordinate of the current point
    - yk (float): y coordinate of the current point
    - xkp1 (float): x coordinate of the next point
    - ykp1 (float): y coordinate of the next point
    - l_k (float): reference length
    - EA (float): elastic modulus

    Returns:
    - F (np.array): Gradient array
    """
    # print("Calculating gradEs...")
    # print(xk, yk, xkp1, ykp1, l_k, k)
    
    F = np.zeros(4)
    F[0] = -(1.0 - np.sqrt((xkp1 - xk)**2.0 + (ykp1 - yk)**2.0) / l_k) * ((xkp1 - xk)**2.0 + (ykp1 - yk)**2.0)**(-0.5) / l_k * (-2.0 * xkp1 + 2.0 * xk)
    F[1] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (-0.2e1 * ykp1 + 0.2e1 * yk)
    F[2] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (0.2e1 * xkp1 - 0.2e1 * xk)
    F[3] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (0.2e1 * ykp1 - 0.2e1 * yk)

    F = 0.5 * k * l_k * F  # Scale by k and l_k

    return F

def hessEs(xk, yk, xkp1, ykp1, l_k, k):
    """
    This function returns the 4x4 Hessian of the stretching energy E_k^s with
    respect to x_k, y_k, x_{k+1}, and y_{k+1}.
    """
    J = np.zeros((4, 4))  # Initialize the Hessian matrix
    J11 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J12 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (-2 * ykp1 + 2 * yk) / 0.2e1
    J13 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (2 * xkp1 - 2 * xk) / 0.2e1 + 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J14 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (2 * ykp1 - 2 * yk) / 0.2e1
    J22 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J23 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (2 * xkp1 - 2 * xk) / 0.2e1
    J24 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (2 * ykp1 - 2 * yk) / 0.2e1 + 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J33 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((2 * xkp1 - 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J34 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (2 * xkp1 - 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (2 * xkp1 - 2 * xk) * (2 * ykp1 - 2 * yk) / 0.2e1
    J44 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((2 * ykp1 - 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J = np.array([[J11, J12, J13, J14],
                   [J12, J22, J23, J24],
                   [J13, J23, J33, J34],
                   [J14, J24, J34, J44]])
    J *= 0.5 * k * l_k
    return J

#-----------------------------------------------------------------------#
# Torsional spring
def getFb(worm, q):
    """
    Torsional spring using cross-product formulation.
    Penalizes deviation from collinearity: E = (k/2) * (t1 × t2)²
    When collinear (t1 || t2): cross = 0, E = 0
    Includes FULL Jacobian for proper Newton convergence.
    """
    Fb = np.zeros(worm.ndof)
    Jb = np.zeros((worm.ndof, worm.ndof))
    
    k_torsion = param.k_torsion / 2  # k/2 per diagonal
    
    for i in range(1, worm.n):
        main_curr = 3 * i
        
        # Diagonal 1: top_{i-1} → main → bot_i
        top_prev = 3 * i - 2
        bot_curr = 3 * i + 2
        
        # Diagonal 2: bot_{i-1} → main → top_i  
        bot_prev = 3 * i - 1
        top_curr = 3 * i + 1
        
        for (node_a, node_b, node_c) in [(top_prev, main_curr, bot_curr),
                                          (bot_prev, main_curr, top_curr)]:
            # DOF indices
            dofs_a = [2*node_a, 2*node_a + 1]
            dofs_b = [2*node_b, 2*node_b + 1]
            dofs_c = [2*node_c, 2*node_c + 1]
            all_dofs = dofs_a + dofs_b + dofs_c  # 6 DOFs total
            
            # Positions
            xa, ya = q[dofs_a[0]], q[dofs_a[1]]
            xb, yb = q[dofs_b[0]], q[dofs_b[1]]
            xc, yc = q[dofs_c[0]], q[dofs_c[1]]
            
            # Edge vectors
            e1 = np.array([xb - xa, yb - ya])
            e2 = np.array([xc - xb, yc - yb])
            
            len1 = np.linalg.norm(e1)
            len2 = np.linalg.norm(e2)
            
            if len1 < 1e-10 or len2 < 1e-10:
                continue
            
            # Unit vectors
            t1 = e1 / len1
            t2 = e2 / len2
            
            # 2D cross product: t1 × t2 = t1[0]*t2[1] - t1[1]*t2[0]
            cross = t1[0] * t2[1] - t1[1] * t2[0]
            
            # Get natural cross product from initial configuration
            q0 = worm.q0.flatten()
            xa0, ya0 = q0[dofs_a[0]], q0[dofs_a[1]]
            xb0, yb0 = q0[dofs_b[0]], q0[dofs_b[1]]
            xc0, yc0 = q0[dofs_c[0]], q0[dofs_c[1]]
            
            e1_0 = np.array([xb0 - xa0, yb0 - ya0])
            e2_0 = np.array([xc0 - xb0, yc0 - yb0])
            len1_0 = np.linalg.norm(e1_0)
            len2_0 = np.linalg.norm(e2_0)
            
            if len1_0 > 1e-10 and len2_0 > 1e-10:
                t1_0 = e1_0 / len1_0
                t2_0 = e2_0 / len2_0
                cross0 = t1_0[0] * t2_0[1] - t1_0[1] * t2_0[0]
            else:
                cross0 = 0.0
            
            # Penalize deviation from natural cross product
            delta_cross = cross - cross0
            
            # === GRADIENT COMPUTATION ===
            # dcross/dt1 and dcross/dt2
            dcross_dt1 = np.array([t2[1], -t2[0]])
            dcross_dt2 = np.array([-t1[1], t1[0]])
            
            # dt/de = (I - t⊗t) / len
            I2 = np.eye(2)
            dt1_de1 = (I2 - np.outer(t1, t1)) / len1
            dt2_de2 = (I2 - np.outer(t2, t2)) / len2
            
            dcross_de1 = dt1_de1 @ dcross_dt1
            dcross_de2 = dt2_de2 @ dcross_dt2
            
            # de/dq: de1 = pb - pa, de2 = pc - pb
            # de1/dpa = -I, de1/dpb = +I, de1/dpc = 0
            # de2/dpa = 0, de2/dpb = -I, de2/dpc = +I
            
            # dcross/dq (6-vector: [dpa_x, dpa_y, dpb_x, dpb_y, dpc_x, dpc_y])
            dcross_dq = np.zeros(6)
            dcross_dq[0:2] = -dcross_de1           # dpa
            dcross_dq[2:4] = dcross_de1 - dcross_de2  # dpb
            dcross_dq[4:6] = dcross_de2            # dpc
            
            # Forces = -dE/dq = -k * delta_cross * dcross/dq
            F_local = -k_torsion * delta_cross * dcross_dq
            
            # Assemble forces
            Fb[dofs_a[0]] += F_local[0]
            Fb[dofs_a[1]] += F_local[1]
            Fb[dofs_b[0]] += F_local[2]
            Fb[dofs_b[1]] += F_local[3]
            Fb[dofs_c[0]] += F_local[4]
            Fb[dofs_c[1]] += F_local[5]
            
            # === FULL JACOBIAN ===
            # J = dF/dq = -d²E/dq² = -k * (dcross/dq ⊗ dcross/dq + cross * d²cross/dq²)
            
            # Term 1: Gauss-Newton term (outer product of gradients)
            J_local = -k_torsion * np.outer(dcross_dq, dcross_dq)
            
            # Term 2: Second derivative term (only significant when delta_cross ≠ 0)
            # d²cross/dq² requires second derivatives of t1, t2
            if abs(delta_cross) > 1e-12:
                d2cross_dq2 = compute_d2cross_dq2(t1, t2, e1, e2, len1, len2)
                J_local -= k_torsion * delta_cross * d2cross_dq2
            
            # Assemble Jacobian
            for ii, di in enumerate(all_dofs):
                for jj, dj in enumerate(all_dofs):
                    Jb[di, dj] += J_local[ii, jj]
    
    return Fb, Jb


def compute_d2cross_dq2(t1, t2, e1, e2, len1, len2):
    """
    Compute the second derivative of cross product w.r.t. positions.
    Returns a 6x6 Hessian matrix for [pa, pb, pc] (each 2D).
    """
    I2 = np.eye(2)
    
    # Projection matrices
    P1 = (I2 - np.outer(t1, t1)) / len1
    P2 = (I2 - np.outer(t2, t2)) / len2
    
    # dcross/dt1 = [t2[1], -t2[0]], dcross/dt2 = [-t1[1], t1[0]]
    # d²cross/dt1dt2 = [[0, 1], [-1, 0]] (rotation matrix)
    d2cross_dt1dt2 = np.array([[0, 1], [-1, 0]])
    
    # dt1/de1 = P1, dt2/de2 = P2
    # d²t1/de1² involves third-order terms, approximate as:
    # d²t/de² ≈ -(t⊗P + P⊗t + (t·dcross/dt)*(I-3t⊗t)) / len²
    # For simplicity, use a first-order approximation
    
    # The dominant cross-terms come from d²cross/(de1 de2)
    # d²cross/de1de2 = dt1/de1 @ d²cross/dt1dt2 @ dt2/de2.T
    d2cross_de1de2 = P1 @ d2cross_dt1dt2 @ P2.T
    
    # Now build the 6x6 Hessian
    # Ordering: [pa_x, pa_y, pb_x, pb_y, pc_x, pc_y]
    # de1/dpa = -I, de1/dpb = +I
    # de2/dpb = -I, de2/dpc = +I
    
    H = np.zeros((6, 6))
    
    # d²cross/(dpa dpa) = d²cross/de1² (need second derivative of t1)
    # Approximate: d²t1/de1² ≈ -(P1@t1⊗t1 + t1⊗t1@P1 + ...)/len1
    # For stability, use simplified form
    d2t1_de1_contracted = -2 * np.outer(P1 @ t1, t1) / len1  # Simplified
    d2cross_de1de1 = d2t1_de1_contracted @ np.array([t2[1], -t2[0]])
    # This is a 2x2 approximation - reshape properly
    
    # For numerical stability, focus on the cross-terms which are more important
    # d²cross/(de1 de2) contributions:
    
    # (dpa, dpb): de1 depends on both, de2 depends on pb
    # (dpa, dpc): de1 depends on pa, de2 depends on pc
    # etc.
    
    # dpa-dpb block: involves de1/dpa @ d²cross/de1de2 @ de2/dpb = (-I) @ d2cross_de1de2 @ (-I)
    H[0:2, 2:4] = d2cross_de1de2
    H[2:4, 0:2] = d2cross_de1de2.T
    
    # dpa-dpc block: de1/dpa and de2/dpc → (-I) @ 0 @ (+I) = 0 (no direct coupling)
    
    # dpb-dpb block: de1/dpb @ ... @ de2/dpb = (+I) @ d2cross_de1de2 @ (-I) + self terms
    H[2:4, 2:4] = -d2cross_de1de2 - d2cross_de1de2.T
    
    # dpb-dpc block: de1/dpb @ d²cross/de1de2 @ de2/dpc = (+I) @ d2cross_de1de2 @ (+I)
    H[2:4, 4:6] = d2cross_de1de2
    H[4:6, 2:4] = d2cross_de1de2.T
    
    # dpa-dpa and dpc-dpc blocks are smaller (second derivative of single t)
    # Approximate as zero for stability
    
    return H

# def getFb(worm, q):
#   # q - DOF vector of size N
#   # deltaL - undeformed Voronoi length (assume to be a scalar for this simple example)
#   # Output:
#   # Fb - a vector (negative gradient of elastic stretching force)
#   # Jb - a matrix (negative hessian of elastic stretching force)

#   deltaL = worm.deltaL
#   Fb = np.zeros(worm.ndof) # bending force
#   Jb = np.zeros((worm.ndof, worm.ndof))
#   for k in range(1, worm.n-1):  # First bending spring (USE A LOOP for the general case)
#     xkm1 = q[2*k-2] # x coordinate of the first node
#     ykm1 = q[2*k-1] # y coordinate of the first node
#     xk = q[2*k] # x coordinate of the second node
#     yk = q[2*k+1] # y coordinate of the second node
#     xkp1 = q[2*k+2] # x coordinate of the third node
#     ykp1 = q[2*k+3] # y coordinate of the third node
#     ind = np.arange(2*k-2, 2*k+4)
#     k_bend = worm.node_k[k]
#     gradEnergy = gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, k_bend)
#     hessEnergy = hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, k_bend)

#     Fb[ind] -= gradEnergy # force = - gradient of energy. Fb is the stretching force
#     Jb[np.ix_(ind, ind)] -= hessEnergy # index vector: 0:6
#   return Fb, Jb

## Bending
def computeNaturalCurvature(xkm1, ykm1, xk, yk, xkp1, ykp1):
    """
    Compute the discrete curvature for a triplet of points.
    Returns the curvature value (kappa).
    """
    node0 = np.array([xkm1, ykm1, 0.0])
    node1 = np.array([xk, yk, 0.0])
    node2 = np.array([xkp1, ykp1, 0.0])
    
    ee = node1 - node0
    ef = node2 - node1
    
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)
    
    if norm_e < 1e-10 or norm_f < 1e-10:
        return 0.0
    
    te = ee / norm_e
    tf = ef / norm_f
    
    denom = 1.0 + np.dot(te, tf)
    if abs(denom) < 1e-10:
        return 0.0
    
    kb = 2.0 * np.cross(te, tf) / denom
    return kb[2]  # z-component is the 2D curvature

def gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, k_bend):
    """
    Returns the derivative of bending energy E_k^b with respect to
    x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    Parameters:
    xkm1, ykm1 : float
        Coordinates of the previous node (x_{k-1}, y_{k-1}).
    xk, yk : float
        Coordinates of the current node (x_k, y_k).
    xkp1, ykp1 : float
        Coordinates of the next node (x_{k+1}, y_{k+1}).
    curvature0 : float
        Discrete natural curvature at node (xk, yk).
    l_k : float
        Voronoi length of node (xk, yk).
    k_bend : float
        Bending stiffness.

    Returns:
    dF : np.ndarray
        Derivative of bending energy.
    """
    # Nodes in 3D
    node0 = np.array([xkm1, ykm1, 0.0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])
    # Unit vectors along z-axis
    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])

    kappaBar = curvature0
    gradKappa = np.zeros(6) # Initialize gradient of curvature

    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1

    # Norms of edge vectors
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    # Unit tangents
    te = ee / norm_e
    tf = ef / norm_f

    
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf)) # Curvature binormal

    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    kappa1 = kb[2] # Curvature
    # Gradient of kappa1 with respect to edge vectors
    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))
    # Populate the gradient of kappa
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] = Dkappa1Df[0:2]
    # Gradient of bending energy
    dkappa = kappa1 - kappaBar
    dF = gradKappa * k_bend * dkappa / l_k
    return dF


def hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, k_bend):
    """
    Returns the Hessian (second derivative) of bending energy E_k^b
    with respect to x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    Parameters:
    xkm1, ykm1 : float
        Coordinates of the previous node (x_{k-1}, y_{k-1}).
    xk, yk : float
        Coordinates of the current node (x_k, y_k).
    xkp1, ykp1 : float
        Coordinates of the next node (x_{k+1}, y_{k+1}).
    curvature0 : float
        Discrete natural curvature at node (xk, yk).
    l_k : float
        Voronoi length of node (xk, yk).
    k_bend : float
        Bending stiffness.
    Returns:
    dJ : np.ndarray
        Hessian of bending energy.
    """

    # Nodes in 3D
    node0 = np.array([xkm1, ykm1, 0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])

    # Unit vectors along z-axis
    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])

    kappaBar = curvature0

    # Initialize gradient of curvature
    gradKappa = np.zeros(6)
    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1
    # Norms of edge vectors
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)
    # Unit tangents
    te = ee / norm_e
    tf = ef / norm_f
    # Curvature binormal
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvature
    kappa1 = kb[2]
    # Gradient of kappa1 with respect to edge vectors
    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))
    # Populate the gradient of kappa
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] = Dkappa1Df[0:2]
    # Compute the Hessian (second derivative of kappa)
    DDkappa1 = np.zeros((6, 6))

    norm2_e = norm_e**2
    norm2_f = norm_f**2

    Id3 = np.eye(3)
    # Helper matrices for second derivatives
    tt_o_tt = np.outer(tilde_t, tilde_t)
    tmp = np.cross(tf, tilde_d2)
    tf_c_d2t_o_tt = np.outer(tmp, tilde_t)
    kb_o_d2e = np.outer(kb, m2e)

    D2kappa1De2 = (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tf_c_d2t_o_tt.T) / norm2_e - \
                  kappa1 / (chi * norm2_e) * (Id3 - np.outer(te, te)) + \
                  (kb_o_d2e + kb_o_d2e.T) / (4 * norm2_e)

    tmp = np.cross(te, tilde_d2)
    te_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d2t = te_c_d2t_o_tt.T
    kb_o_d2f = np.outer(kb, m2f)

    D2kappa1Df2 = (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + te_c_d2t_o_tt.T) / norm2_f - \
                  kappa1 / (chi * norm2_f) * (Id3 - np.outer(tf, tf)) + \
                  (kb_o_d2f + kb_o_d2f.T) / (4 * norm2_f)
    D2kappa1DeDf = -kappa1 / (chi * norm_e * norm_f) * (Id3 + np.outer(te, tf)) \
                  + 1.0 / (norm_e * norm_f) * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + \
                  tt_o_te_c_d2t - helper.crossMat(tilde_d2))
    D2kappa1DfDe = D2kappa1DeDf.T

    # Populate the Hessian of kappa
    DDkappa1[0:2, 0:2] = D2kappa1De2[0:2, 0:2]
    DDkappa1[0:2, 2:4] = -D2kappa1De2[0:2, 0:2] + D2kappa1DeDf[0:2, 0:2]
    DDkappa1[0:2, 4:6] = -D2kappa1DeDf[0:2, 0:2]
    DDkappa1[2:4, 0:2] = -D2kappa1De2[0:2, 0:2] + D2kappa1DfDe[0:2, 0:2]
    DDkappa1[2:4, 2:4] = D2kappa1De2[0:2, 0:2] - D2kappa1DeDf[0:2, 0:2] - \
                         D2kappa1DfDe[0:2, 0:2] + D2kappa1Df2[0:2, 0:2]
    DDkappa1[2:4, 4:6] = D2kappa1DeDf[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 0:2] = -D2kappa1DfDe[0:2, 0:2]
    DDkappa1[4:6, 2:4] = D2kappa1DfDe[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 4:6] = D2kappa1Df2[0:2, 0:2]

    # Hessian of bending energy
    dkappa = kappa1 - kappaBar
    dJ = 1.0 / l_k * k_bend * np.outer(gradKappa, gradKappa)
    dJ += 1.0 / l_k * dkappa * k_bend * DDkappa1
    return dJ
