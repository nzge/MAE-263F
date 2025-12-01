import numpy as np

#-----------Worm Parameters------------#

# Parameters for the worm model 
length=1.0 # length of the worm
n=6 # number of segments
 
# Densities
rho_metal = 10 # Density of metal # kg/m^3
rho_gl = 10 # kg/m^3
rho = rho_metal  - rho_gl # Difference in density # kg/m^3

r0 = 1e-3 # Cross sectional radius # meter
Y = 1e9 # Young's modulus # Pa
visc = 1000.0 # Viscosity# Pa-s
EI = Y * np.pi * r0**4 / 4
EA = Y * np.pi * r0**2

#-----------Environment Parameters------------#
mu_static=0.12
mu_sliding=0.1

#-----------Solver Parameters------------#

# Time integration parameters
tol = EI / length ** 2 * 1e-3 # Tolerance
dt = 0.01 # time step size
total_time = 1.0 # total simulation time

# Solver parameters
maximum_iter = 100 # Maximum number of iterations
dt = 0.01 # Time step # second
totalTime = 5 # Total time # second

# Variables related to plotting
saveImage = 0 # Set to 1 to save images
n_plot = 5 
plotStep = (totalTime/dt) / n_plot 


