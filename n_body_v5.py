import time 
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
dt = 0.001 # Not needed if CFL stepping, but needs to be declared

## Courant number
C = 0.01

# Energy error threshold
e_thresh = 0.05

# Gravitational constant
G = 1
#G = 6.67408e-11

# Number of bodies: 2, 3, 4. (str) Or can do space
sim_type = '3'
# Setup array for trajectories
if sim_type == '2':
    x_ICs = (1,0,0,0,0,0) # x,y,z,x,y,z,...
    v_ICs = (-0.25,-0.5,0,0.25,0.25,0) # x,y,z,x,y,z,...
    mass = (1,1)
elif sim_type == '3':
    mass = (1,1,1) # Equal masses
    # Can be fig-8 or Euler-line or Lagrange
    orbit_type = 'Lagrange'
    if orbit_type == 'fig-8':
        # Fig 8
        x_IC_x = 0.97000436
        x_IC_y = 0.24308753
        v_IC_x = -0.93240737
        v_IC_y = -0.86473146
        x_ICs = (x_IC_x,-x_IC_y,0,-x_IC_x,x_IC_y,0,0,0,0) # x,y,z,x,y,z,...
        v_ICs = (-v_IC_x/2,-v_IC_y/2,0,-v_IC_x/2,-v_IC_y/2,0,v_IC_x,v_IC_y,0) # x,y,z,x,y,z,...
    elif orbit_type == 'Euler-line':
        # Euler line
        radius = 1
        x_ICs = (-radius,0,0,0,0,0,radius,0,0) # x,y,z,x,y,z,...
        v_ICs = (0,-np.sqrt(G*(5/4)/radius),0,0,0,0,0,np.sqrt(G*(5/4)),0) # x,y,z,x,y,z,...
    elif orbit_type == 'Lagrange':
        # Lagrange
        h = 1
        hypotenuse = 2*h/np.sqrt(3)
        M = 3 # Needs equal masses
        radius = 2*h/3
        v_ = np.sqrt(G*(M-1)/(2*h))  # speed of each body
        ### My calcs
        x_ICs = (0,h,0,hypotenuse/2,0,0,-hypotenuse/2,0,0) # x,y,z,x,y,z,...
        v_ICs = (-v_,0,0,np.sin(np.pi/6)*v_,np.cos(np.pi/6)*v_,0,np.sin(np.pi/6)*v_,-np.cos(np.pi/6)*v_,0) # x,y,z,x,y,z,...
        ### Max's calcs
        #x_ICs = (0,1,0,-np.sqrt(3)/2,-1/2,0,np.sqrt(3)/2,-1/2,0) # x,y,z,x,y,z,...
        #v_ICs = (-1/np.sqrt(np.sqrt(3)),0,0,1/(2*np.sqrt(np.sqrt(3))),-np.sqrt(np.sqrt(3))/2,0,1/(2*np.sqrt(np.sqrt(3))),np.sqrt(np.sqrt(3))/2,0) # x,y,z,x,y,z,...


elif sim_type == '4':
    mass = (1,1,1,1)
    M = 4
    x_ = 1
    radius = 1
    x_ICs = (x_,0,0,0,x_,0,-x_,0,0,0,-x_,0)
    v_ = np.sqrt(G*(5)/(4*radius))
    v_ICs = (0,v_,0,-v_,0,0,0,-v_,0,v_,0,0)

    #x_ICs = (1.382857,0,0,0,0.157030,0,-1.382857,0,0,0,-0.157030,0)
    #v_ICs = (0,0.584873,0,1.871935,0,0,0,-0.584873,0,-1.871935,0,0)
    
elif sim_type == 'space':
    x_ICs = (0,0,0,1.5e11,0,0) # x,y,z,x,y,z,...
    v_ICs = (0,0,0,0,30000e5,0) # x,y,z,x,y,z,...
    mass = (1.9e30,5.9e24)
    ## NEED TO CHANGE G FOR THIS TO WORK
    # Earth to Sun is 150 million km, or 1.5 x 10^11 m
    # Earth orbital speed is 30km/s, so 30000m/s
    # Sun mass is 1.9 x 10^30 kg , Earth mass is 5.9 x 10^24 kg

# Visualisation
vis_type = 'animation'
dims = 2
plot_more = True
######## CHANGE VALUES ABOVE ########

n_bodies = round(len(x_ICs)/3)

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
print(x_array[:,0])
CoM_vel_array = np.tile(CoM_v,(n_bodies,1))
v_array[:,0] = v_array[:,0] - CoM_vel_array[:,0]
print(v_array[:,0])

# Angular momentum
L = np.empty((3,n_steps+1))
L[:,0] = angular_momentum(x_array[:,0],v_array[:,0],mass) # Initial angular momentum
# Energy
E = np.empty(n_steps+1)
E[0],r = total_energy(x_array[:,0],v_array[:,0],mass,G) # Initial energy

## dt_prog is for plotting over time
dt_prog = np.empty(n_steps+1)
dt_prog[0] = t

## These are similar to above, except for just CFL values
## dt_CFL_prog is for plotting over time
dt_CFL_prog = np.empty(n_steps+1)
dt_CFL_prog[0] = t
## dt_CFL_store is storing each dt
dt_CFL_store= np.empty(n_steps)

################################ Run method ################################
t0 = time.time()
for i in range(n_steps):
    #### Step forward
    ## Finding CFL dt values
    del_t = np.ones((n_bodies,n_bodies))
    for j in range(n_bodies):
        for k in range(j+1,n_bodies):
            if r[j,k] != 0:
                del_t[j,k] = (C * r[j,k])/np.linalg.norm(v_array[3*j:(j+1)*3,i] - v_array[3*k:(k+1)*3,i])
            else:
                del_t[j,k] = np.max(del_t)
            del_t[k,j] = del_t[j,k]
    dt_CFL_0 = np.min(del_t)
    dt = dt_CFL_0 # Actual timestep to be used
    # Update position and velocity 
    x_array_test,v_array_test = Leapfrog_step(x_array[:,i],v_array[:,i],ode_func,dt,mass,G)

    #Calculate energy to get r
    _,r_1 = total_energy(x_array_test,v_array_test,mass,G)
    #### Recalculate dt at future timestep
    ## Finding CFL dt values
    del_t = np.ones((n_bodies,n_bodies))
    for j in range(n_bodies):
        for k in range(j+1,n_bodies):
            if r_1[j,k] != 0:
                del_t[j,k] = (C * r_1[j,k])/np.linalg.norm(v_array_test[3*j:(j+1)*3] - v_array_test[3*k:(k+1)*3])
            else:
                del_t[j,k] = np.max(del_t)
            del_t[k,j] = del_t[j,k]

    dt_CFL_1 = np.min(del_t) # Current variable timestep
    # Calculate timestep to use
    dt_CFL = (1/2) * (dt_CFL_0 + dt_CFL_1)
    # Store and use this dt_CFL
    dt_CFL_prog[i+1] = dt_CFL_prog[i] + dt_CFL # This is cumulative vector of dt
    dt_CFL_store[i] = dt_CFL # This stores dt at each time
    dt = dt_CFL
    # Update position and velocity 
    x_array[:,i+1],v_array[:,i+1] = Leapfrog_step(x_array[:,i],v_array[:,i],ode_func,dt,mass,G)

    ## Track conserved quantities
    L[:,i+1] = angular_momentum(x_array[:,i+1],v_array[:,i+1],mass)
    E[i+1],r = total_energy(x_array[:,i+1],v_array[:,i+1],mass,G)

    ## Stop if position or energy error grows too large
    final_it = i+1 # Final iteration
    box_size = 100
    if np.mod(i,100) == 0 and sim_type != 'space':
        # Every 100 steps, check
        if any(np.abs(x_array[:,i+1]) > box_size):
            # If position goes outside square/cube of side length 2*box_size, centred on origin
            print('Outside box of size {}, so terminated at time {}'.format(box_size,dt_CFL_prog[i+1]))
            break
        elif np.abs((E[i+1] - E[0])/E[0]) > e_thresh:
            # If energy error goes above the threshold
            print('Rel energy error is above {}, so terminated at time {}'.format(e_thresh,dt_CFL_prog[i+1]))
            break    

t1 = time.time() - t0    
print('Time for computation: {} seconds'.format(t1))
   
################################ Visualise ################################
#visualise(x_array,mass,vis_type,dims) 

if plot_more:
    t_axis = dt_CFL_prog[:final_it+1]
    ### Trajectory
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    # x,y plane
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.axis('equal')
    ax0.set_title('x-y plane')
    elements = np.arange(start = 0,stop = (n_bodies)*3,step = 3).astype(int)
    ax0.plot(np.transpose(x_array[elements,:final_it+1]),np.transpose(x_array[elements+1,:final_it+1]))
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    labels = list(map(str,mass))
    ax0.legend(labels,title = 'Masses of bodies')
    # x and y against t
    ax1 = fig.add_subplot(gs[:-1, 1])
    ax1.set_title('x against t')
    centre_pos = np.zeros((2,n_bodies))
    for i in range(n_bodies):
        ax1.plot(t_axis,x_array[i*3,:final_it+1])
        #centre_pos[0,i] = (np.max(x_array[i*3,:final_it+1]) + np.min(x_array[i*3,:final_it+1]))/2
    ax1.set_xlabel('time')
    ax1.set_ylabel('x')
    #
    ax2 = fig.add_subplot(gs[-1, 1:])
    ax2.set_title('y against t')
    for i in range(n_bodies):
        ax2.plot(t_axis,x_array[(i*3) + 1,:final_it+1])
        #centre_pos[1,i] = (np.max(x_array[(i*3) + 1,:final_it+1]) + np.min(x_array[(i*3) + 1,:final_it+1]))/2
    ax2.set_xlabel('time')
    ax2.set_ylabel('y')
    fig.tight_layout()
    #ax0.plot(centre_pos[0,:],centre_pos[1,:],'r+')
    plt.show()

    # Plot delta t
    plt.plot(np.arange(final_it),dt_CFL_store[:final_it])
    plt.xlabel('steps')
    plt.ylabel('dt')
    plt.title('Plot of dt chosen for each timestep, with C = {}'.format(C))
    plt.show()


    # Plot energy over time
    E = E[:final_it+1]
    if E[0] != 0:
        print('E0 not 0')
        energy_error = np.abs((E - E[0])/E[0])
    else:
        print('E0 is 0')
        energy_error = np.abs(E - E[0])/np.sum(kinetic_energy(v_array[:,0],mass))
    plt.plot(t_axis,energy_error)
    plt.xlabel('time')
    plt.ylabel('energy error')
    plt.title('Relative error of energy against time')
    plt.show()

    # Plot L_z over time
    L = L[:,:final_it+1] 
    if L[2,0] != 0:
        print('Lz0 not 0')
        L_z_error = (L[2,:] - L[2,0])/L[2,0]
    else:
        print('Lz0 is 0')
        x_0 = (x_array[:,0].reshape((3,-1),order = 'F'))
        v_0 = (v_array[:,0].reshape((3,-1),order = 'F'))
        CoM = CoM_pos(x_array[:,0],mass).reshape((3,-1))
        p_0 = v_0*mass[0] # Momentum
        r_0 = x_0 - CoM # All r vectors
        L_0 = np.cross(r_0[:,0],p_0[:,0])
        L_z_error = (L[2,:] - L[2,0])/L_0[2]
    plt.plot(t_axis,L_z_error)
    plt.xlabel('time')
    plt.ylabel('L_z error')
    plt.title('Relative error of angular momentum (z) against time')
    plt.show()




    
