Homework 1: Spring Networks

The spring network simulation can be run using the jupyter notebook or the python file by hitting run.

To select which integrator to use, find these 3 lines in the main loop:

x_new, u_new = myInt(t_new, x_old, u_old, free_DOF, stiffness_matrix, index_matrix, m, dt)
#x_new, u_new, a_old = myIntNewmark(t_new, x_old, u_old, a_old, free_DOF, stiffness_matrix, index_matrix, m, dt, beta=1/4, gamma=1/2)
#x_new, u_new = myIntE(t_new, x_old, u_old, free_DOF, stiffness_matrix, index_matrix, m, dt)

Leave the line to the corresponding desired integrator uncommented to use it. 
The time steps and time of simnulation can also be tweaked from the main loop.