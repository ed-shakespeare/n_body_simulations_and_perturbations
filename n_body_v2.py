import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from Integration_methods_bodies import Euler_step,Euler_Cromer_step,Leapfrog_step
from visualisation import visualise
from test_constants import CoM_pos, CoM_vel, angular_momentum, eccentricity, total_energy, kinetic_energy
from ode_function import ode_func

################################ Setup ################################ 
######## CHANGE VALUES BELOW ########
# Setup values
n_steps = 10000
t = 0 # Start time
dt = 0.001

## Courant number
C = 1/1000

bodies = '2d'
# Setup array for trajectories
if bodies == '2d':
    x_ICs = (1,0,0,0,0,0) # x,y,z,x,y,z,...
    v_ICs = (-0.25,-0.5,0,0.25,0.25,0) # x,y,z,x,y,z,...
    mass = (1,1)
elif bodies == '3d':
    x_ICs = (1,0,0,0,0,0,1,-1,0) # x,y,z,x,y,z,...
    v_ICs = (-0.25,-0.5,0,0.25,0.25,0,0,-0.5,0) # x,y,z,x,y,z,...
    mass = (1,1,1)
#x_ICs = (0,0,0,150e9,0,0) # x,y,z,x,y,z,...
#v_ICs = (0,0,0,0,29784.934,0) # x,y,z,x,y,z,...
#mass = (10e30,10e24)

# Gravitational constant
G = 1
#G = 6.67408e-11

# Visualisation
vis_type = 'graph'
dims = 2
plot_more = True
######## CHANGE VALUES ABOVE ########

n_bodies = round(len(x_ICs)/3)

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

##  Tracking 'conserved' quantities, and reset to CoM frame
# CoM position
CoM = np.empty((3,n_steps+1))
CoM[:,0] = CoM_pos(x_array[:,0],mass) # Initial CoM
# CoM velocity
CoM_v = np.empty((3,n_steps+1))
CoM_v[:,0] = CoM_vel(v_array[:,0],mass) # Initial CoM

# In centre of mass frame
CoM_pos_array = np.tile(CoM,(n_bodies,1))
x_array[:,0] = x_array[:,0] - CoM_pos_array[:,0]
CoM_vel_array = np.tile(CoM_v,(n_bodies,1))
v_array[:,0] = v_array[:,0] - CoM_vel_array[:,0]

# Angular momentum
L = np.empty((3,n_steps+1))
L[:,0] = angular_momentum(x_array[:,0],v_array[:,0],mass) # Initial angular momentum
# Energy
E = np.empty(n_steps+1)
E[0],r = total_energy(x_array[:,0],v_array[:,0],mass,G) # Initial energy
### Should eccentricity be conserved? Seems to contain other measurements that should, so I'll track it similarly
# Eccentricity
e = np.empty(n_steps+1)
e[0] = eccentricity(x_array[:,0],v_array[:,0],mass,G) # Initial eccentricity

## dt_array is for plotting over time
dt_array = np.empty(n_steps+1)
dt_array[0] = t
## dt_store is storing each dt
dt_store= np.empty(n_steps)

################################ Run method ################################
for i in range(n_steps):
    ## Finding CFL dt values
    del_t = np.ones(n_bodies)
    for j in range(n_bodies):
        for k in range(n_bodies):
            if r[j,k] != 0:
                del_t[j] = (C * r[j,k])/np.linalg.norm(v_array[3*j:(j+1)*3,i])
            else:
                del_t[j] = np.max(del_t)

    dt = np.min(del_t)
    # Update position and velocity 
    x_array[:,i+1],v_array[:,i+1] = Euler_Cromer_step(x_array[:,i],v_array[:,i],ode_func,dt,mass,G)
        # np.delete is technically slow but hopefully good enough.
    # Store dt in both arrays
    dt_array[i+1] = dt_array[i] + dt
    dt_store[i] = dt
    t += dt
    ## Track conserved quantities
    CoM[:,i+1] = CoM_pos(x_array[:,i+1],mass)
    CoM_v[:,i+1] = CoM_vel(x_array[:,i+1],mass)
    L[:,i+1] = angular_momentum(x_array[:,i+1],v_array[:,i+1],mass)
    E[i+1],r = total_energy(x_array[:,i+1],v_array[:,i+1],mass,G)
    e[i+1] = eccentricity(x_array[:,i+1],v_array[:,i+1],mass,G)
    
    
   
################################ Visualise ################################
#visualise(x_array,mass,vis_type,dims) 

if plot_more:
    ### Trajectory
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    # x,y plane
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_title('x-y plane')
    elements = np.arange(start = 0,stop = (n_bodies)*3,step = 3).astype(int)
    ax0.plot(np.transpose(x_array[elements,:]),np.transpose(x_array[elements+1,:]))
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    labels = list(map(str,mass))
    ax0.legend(labels,title = 'Masses of bodies')
    # x and y against t
    ax1 = fig.add_subplot(gs[1, :-1])
    ax1.set_title('x against t')
    for i in range(n_bodies):
        ax1.plot(dt_array,x_array[i*3,:])
    ax1.set_xlabel('time')
    ax1.set_ylabel('x')

    ax2 = fig.add_subplot(gs[1:, -1])
    ax2.set_title('y against t')
    for i in range(n_bodies):
        ax2.plot(dt_array,x_array[(i*3) + 1,:])
    ax2.set_xlabel('time')
    ax2.set_ylabel('y')
    fig.tight_layout()
    plt.show()

    # Plot delta t
    plt.plot(np.arange(n_steps),dt_store)
    plt.xlabel('steps')
    plt.ylabel('dt')
    plt.title('Plot of dt chosen for each timestep, with C = {}'.format(C))
    plt.show()

    # Plot energy over time
    if E[0] != 0:
        print('E0 not 0')
        energy_error = np.abs((E - E[0])/E[0])
    else:
        energy_error = np.abs(E - E[0])/np.sum(kinetic_energy(v_array[:,0],mass))
    plt.plot(dt_array,energy_error)
    plt.xlabel('time')
    plt.ylabel('energy error')
    plt.title('Relative error of energy against time')
    plt.show()

    # Plot L_z over time
    if L[2,0] != 0:
        print('Lz0 not 0')
        L_z_error = (L[2,:] - L[2,0])/L[2,0]
    else:
        x_0 = (x_array[:,0].reshape((3,-1),order = 'F'))
        v_0 = (v_array[:,0].reshape((3,-1),order = 'F'))
        CoM = CoM_pos(x_array[:,0],mass).reshape((3,-1))
        p_0 = v_0*mass[0] # Momentum
        r_0 = x_0 - CoM # All r vectors
        L_0 = np.cross(r_0[:,0],p_0[:,0])
        L_z_error = (L[2,:] - L[2,0])/L_0[2]
    plt.plot(dt_array,L_z_error)
    plt.xlabel('time')
    plt.ylabel('L_z error')
    plt.title('Relative error of angular momentum (z) against time')
    plt.show()


    ### Eccentricity and energy over time
    #fig, (ax1, ax2) = plt.subplots(nrows=2)
    # Plot eccentricity over time
    #ax1.plot(np.arange(n_steps+1)*dt,e)
    #ax1.set_xlabel('time')
    #ax1.set_ylabel('eccentricity')

    # Plot energy over time
    #ax2.plot(np.arange(n_steps+1)*dt,E)
    #ax2.set_xlabel('time')
    #ax2.set_ylabel('energy')
    #plt.show()

    
    #### Plotting angular momentum
    #fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
    #plt.title('Angular momentum')
    #ax0.plot(np.arange(n_steps+1)*dt,L[0,:])
    #ax0.set_xlabel('time')
    #ax0.set_ylabel('L_x')
    #ax1.plot(np.arange(n_steps+1)*dt,L[1,:])
    #ax1.set_xlabel('time')
    #ax1.set_ylabel('L_y')
    #ax2.plot(np.arange(n_steps+1)*dt,L[2,:])
    #ax2.set_xlabel('time')
    #ax2.set_ylabel('L_z')
    #plt.show()

    
