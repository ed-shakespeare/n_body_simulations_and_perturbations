import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from Integration_methods_bodies import Euler_step,Euler_Cromer_step,Leapfrog_step
from visualisation import visualise
from test_constants import CoM_pos, angular_momentum, eccentricity, total_energy
from ode_function import ode_func

################################ Setup ################################ 
######## CHANGE VALUES BELOW ########
# Setup values
n_steps = 10000
t = 0 # Start time
dt = 0.01

# Setup array for trajectories
x_ICs = (1,0,0,0,0,0) # x,y,z,x,y,z,...
v_ICs = (-0.5,-1,0,0.5,0.5,0) # x,y,z,x,y,z,...
mass = (3,1)
######## CHANGE VALUES ABOVE ########

n_bodies = int(len(x_ICs)/3)

# For equal weights of 1, can use []
if mass == []:
    mass = np.ones(n_bodies)
# Check number of bodies is consistent
if len(x_ICs) != len(v_ICs) or len(x_ICs) != len(mass)*3:
    raise Exception('Position and velocity IC vectors must both have 3 times the length of the masses vector')

x_array = np.empty((3*n_bodies,n_steps+1))
v_array = np.empty((3*n_bodies,n_steps+1))
x_array[:,0] = np.transpose(x_ICs)
v_array[:,0] = np.transpose(v_ICs)

##  Tracking 'conserved' quantities
# CoM
CoM = np.empty((3,n_steps+1))
CoM[:,0] = CoM_pos(x_array[:,0],mass) # Initial CoM
# Angular momentum
L = np.empty((3,n_steps+1))
L[:,0] = angular_momentum(x_array[:,0],v_array[:,0],mass) # Initial angular momentum
# Energy
E = np.empty(n_steps+1)
E[0] = total_energy(x_array[:,0],v_array[:,0],mass) # Initial energy
### Should eccentricity be conserved? Seems to contain other measurements that should, so I'll track it similarly
# Eccentricity
e = np.empty(n_steps+1)
e[0] = eccentricity(x_array[:,0],v_array[:,0],mass) # Initial eccentricity


################################ Run method ################################
for i in range(n_steps):
    # For each body, update position and velocity
    for j in range(n_bodies):
        x_array[j*3:(j+1)*3,i+1],v_array[j*3:(j+1)*3,i+1] = Euler_Cromer_step(x_array[j*3:(j+1)*3,i],v_array[j*3:(j+1)*3,i],t,ode_func,dt,np.delete(x_array,(j*3,j*3+1,j*3+2),0)[:,i],mass = np.delete(mass,j))
        # np.delete is technically slow but hopefully good enough.
    ## Track conserved quantities
    CoM[:,i+1] = CoM_pos(x_array[:,i+1],mass)
    L[:,i+1] = angular_momentum(x_array[:,i+1],v_array[:,i+1],mass)
    E[i+1] = total_energy(x_array[:,i+1],v_array[:,i+1],mass)
    e[i+1] = eccentricity(x_array[:,i+1],v_array[:,i+1],mass)
    
    t += dt
   
################################ Visualise ################################
######## CHANGE VALUES BELOW ########
visualise(x_array,mass,'animation',dims = 2,CoM = CoM, CoM_frame=True) 
######## CHANGE VALUES ABOVE ########

plot_more = True
if plot_more:
    ### Eccentricity and energy over time
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    # Plot eccentricity over time
    ax1.plot(np.arange(n_steps+1)*dt,e)
    ax1.set_xlabel('time')
    ax1.set_ylabel('eccentricity')

    # Plot energy over time
    ax2.plot(np.arange(n_steps+1)*dt,E)
    ax1.set_xlabel('time')
    ax2.set_ylabel('energy')
    plt.show()

    ### Plotting angular momentum
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(n_bodies):
        plt.plot(np.transpose(L[0,:]),np.transpose(L[1,:]),np.transpose(L[2,:]))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('Angular momentum vector')
    plt.show()