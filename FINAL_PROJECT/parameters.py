import numpy as np

#-----------Worm Parameters------------#

# Parameters for the worm model 
length=1.0 # length of the worm
n=6 # number of segments
 
# Mass and density parameters
total_mass = 0.01 # total mass of the worm # kg
rho_material = 10 # Density of metal # kg/m^3
rho_gl = 10 # kg/m^3
rho = rho_material  - rho_gl # Difference in density # kg/m^3

visc = 1000.0 # Viscosity# Pa-s

k_spring = 13 # spring constant
k_link = 1e4 # link constant
k_torsion = 0.3 # bend constant (gentle coupling between adjacent segments)
stretch_fraction = 0.5 # stretch fraction

#-----------Environment Parameters------------#
mu_static=0.12
mu_sliding=0.1

#-----------Solver Parameters------------#

# Time integration parameters
tol = k_torsion / length ** 2 * 1e-3 # Tolerance

# Solver parameters
maximum_iter = 100 # Maximum number of iterations
dt = 0.01 # Time step # second
totalTime = 10.0 # Total time # second

# Variables related to plotting
saveImage = 0 # Set to 1 to save images
n_plot = 5 
plotStep = (totalTime/dt) / n_plot 


