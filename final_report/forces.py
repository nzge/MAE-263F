import numpy as np
import parameters as param

# Mass vector and matrix
m = np.zeros( 2 * param.nv )
for k in range(0, param.nv):
  m[2*k] = 4/3 * np.pi * param.R[k]**3 * param.rho_metal # mass of k-th node along x
  m[2*k + 1] = 4/3 * np.pi * param.R[k]**3 * param.rho_metal # mass of k-th node along y
mMat = np.diag(m)

# Gravity (external force)
W = np.zeros( 2 * param.nv)
g = np.array([0, -9.8]) # m/s^2
for k in range(0, param.nv):
  W[2*k] = 4.0 / 3.0 * np.pi * param.R[k]**3 * param.rho * g[0] # Weight along x
  W[2*k+1] = 4.0 / 3.0 * np.pi * param.R[k]**3 * param.rho * g[1] # Weight along y
# Gradient of W = 0

# Viscous damping (external force)
C = np.zeros((2 * param.nv, 2 * param.nv))
for k in range(0, param.nv):
  C[2*k, 2*k] = 6.0 * np.pi * param.visc * param.R[k] # Damping along x for k-th node
  C[2*k+1, 2*k+1] = 6.0 * np.pi * param.visc * param.R[k] # Damping along y for k-th node
