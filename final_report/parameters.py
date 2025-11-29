import numpy as np

# Parameters for the worm model 
length=0.25 # length of the worm
n=6 # number of segments
 
# Densities
rho_metal = 7000 # Density of metal # kg/m^3
rho_gl = 1000 # kg/m^3
rho = rho_metal  - rho_gl # Difference in density # kg/m^3

r0 = 1e-3 # Cross sectional radius # meter
Y = 1e9 # Young's modulus # Pa
visc = 1000.0 # Viscosity# Pa-s
EI = Y * np.pi * r0**4 / 4
EA = Y * np.pi * r0**2

#-----------------------#


# Time integration parameters
tol = EI / length ** 2 * 1e-3 # Tolerance
dt = 0.01 # time step size
total_time = 1.0 # total simulation time

# Solver parameters
maximum_iter = 1000 # Maximum number of iterations
dt = 0.01 # Time step # second
totalTime = 5 # Total time # second

# Variables related to plotting
saveImage = 0 # Set to 1 to save images
plotStep = 250 # Every 5-th step will be plotted
