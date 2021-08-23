import time 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from Integration_methods_bodies import Euler_step,Euler_Cromer_step,Leapfrog_step
from visualisation import visualise
from test_constants import CoM_pos, CoM_vel, angular_momentum, eccentricity, total_energy, kinetic_energy
from ode_function import ode_func
from mpl_toolkits.mplot3d import Axes3D


################################ Setup ################################ 
######## CHANGE VALUES BELOW ########
# Setup values
n_steps = 100000
duration = 50
t = 0 # Start time
dt = 0.001 # Not needed if CFL stepping, but needs to be declared

## Courant number
C =  0.01

# Energy error threshold
e_thresh = 0.05
box_size = 10
timestep_size = 10e-9
timestep_count = 100000

# Gravitational constant
G = 1
#G = 6.67408e-11

# Visualisation
vis_type = 'animation'
dims = 2
plot_more = 1   

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
    orbit_type = 'fig-8'
    if orbit_type == 'fig-8':
        # Fig 8
        x_IC_x = 0.97000436
        x_IC_y = 0.24308753
        v_IC_x = -0.93240737
        v_IC_y = -0.86473146
        x_ICs = (x_IC_x,-x_IC_y,0,-x_IC_x,x_IC_y,0,0,0,0) # x,y,z,x,y,z,...
        v_ICs = (-v_IC_x/2,-v_IC_y/2,0,-v_IC_x/2,-v_IC_y/2,0,v_IC_x,v_IC_y,0) # x,y,z,x,y,z,...
        print(x_ICs)
        print(v_ICs)
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
        v_ = np.sqrt(np.sqrt(3)*G/2)  # speed of each body
        ### My calcs
        x_ICs = (0,h,0,hypotenuse/2,0,0,-hypotenuse/2,0,0) # x,y,z,x,y,z,...
        v_ICs = (-v_,0,0,np.sin(np.pi/6)*v_,np.cos(np.pi/6)*v_,0,np.sin(np.pi/6)*v_,-np.cos(np.pi/6)*v_,0) # x,y,z,x,y,z,...
        ### Max's calcs
        #x_ICs = (0,1,0,-np.sqrt(3)/2,-1/2,0,np.sqrt(3)/2,-1/2,0) # x,y,z,x,y,z,...
        #v_ICs = (-1/np.sqrt(np.sqrt(3)),0,0,1/(2*np.sqrt(np.sqrt(3))),-np.sqrt(np.sqrt(3))/2,0,1/(2*np.sqrt(np.sqrt(3))),np.sqrt(np.sqrt(3))/2,0) # x,y,z,x,y,z,...
    elif orbit_type == 'other1':
        # I.A.1 butterfly I
        x_dot = 0.30689
        y_dot = 0.12551
        x_ICs = (-1,0,0,1,0,0,0,0,0) # x,y,z,x,y,z,...
        v_ICs = (x_dot,y_dot,0,x_dot,y_dot,0,-2*x_dot,-2*y_dot,0) # x,y,z,x,y,z,...
    elif orbit_type == 'other2':
        # II.C.2b yin-yang I
        x_dot = 0.28270
        y_dot = 0.32721
        x_ICs = (-1,0,0,1,0,0,0,0,0) # x,y,z,x,y,z,...
        v_ICs = (x_dot,y_dot,0,x_dot,y_dot,0,-2*x_dot,-2*y_dot,0) # x,y,z,x,y,z,...
    elif orbit_type == 'other3':
        # I.B.2 moth II
        x_dot = 0.43917
        y_dot = 0.45297
        x_ICs = (-1,0,0,1,0,0,0,0,0) # x,y,z,x,y,z,...
        v_ICs = (x_dot,y_dot,0,x_dot,y_dot,0,-2*x_dot,-2*y_dot,0) # x,y,z,x,y,z,...

elif sim_type == '4':
    mass = (1,1,1,1)
    orbit_type = 'central'
    if orbit_type == 'Lagrange':
        M = 4
        x_ = 1
        x_ICs = (x_,0,0,0,x_,0,-x_,0,0,0,-x_,0)
        v_ = np.sqrt(G*(1+2*np.sqrt(2))/4)
        v_ICs = (0,v_,0,-v_,0,0,0,-v_,0,v_,0,0)
    elif orbit_type == 'chain':
        x_ICs = (1.382857,0,0,0,0.157030,0,-1.382857,0,0,0,-0.157030,0)
        v_ICs = (0,0.584873,0,1.871935,0,0,0,-0.584873,0,-1.871935,0,0)
    elif orbit_type == 'central':
        h = 1
        hypotenuse = 2*h/np.sqrt(3)
        M = 1 # Needs equal masses
        v_ = np.sqrt((3+np.sqrt(3))*G*M/2)  # speed of each body
        ### My calcs
        x_ICs = (0,h,0,hypotenuse/2,0,0,-hypotenuse/2,0,0,0,1/3,0) # x,y,z,x,y,z,...
        v_ICs = (-v_,0,0,np.sin(np.pi/6)*v_,np.cos(np.pi/6)*v_,0,np.sin(np.pi/6)*v_,-np.cos(np.pi/6)*v_,0,0,0,0) # x,y,z,x,y,z,...


elif sim_type == '5':
    mass = (1,1,1,1,1)
    x2 = 0.439775
    x3 = -1.268608
    y2 = -0.169717
    y3 = -0.267651 
    xdot2 = 1.822785 
    xdot3 = 1.271564
    ydot2 = 0.128248
    ydot3 = 0.168645
    x_ICs = (-2*x2 - 2*x3,0,0,x2,y2,0,x3,y3,0,x3,-y3,0,x2,-y2,0)
    v_ICs = (0,-2*ydot2 - 2*ydot3,0,xdot2,ydot2,0,xdot3,ydot3,0,-xdot3,ydot3,0,-xdot2,ydot2,0)
    
elif sim_type == 'space':
    x_ICs = (0,0,0,1.5e11,0,0) # x,y,z,x,y,z,...
    v_ICs = (0,0,0,0,3e5,0) # x,y,z,x,y,z,...
    mass = (1.9e32,5.9e24)
    G = 6.67408e-11
    ## NEED TO CHANGE G FOR THIS TO WORK
    # Earth to Sun is 150 million km, or 1.5 x 10^11 m
    # Earth orbital speed is 30km/s, so 30000m/s
    # Sun mass is 1.9 x 10^30 kg , Earth mass is 5.9 x 10^24 kg

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
CoM_vel_array = np.tile(CoM_v,(n_bodies,1))
v_array[:,0] = v_array[:,0] - CoM_vel_array[:,0]

print(x_array[:,0])
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
    del_t = np.ones((n_bodies,n_bodies))*10e3
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
    x_array_test,v_array_test,_ = Leapfrog_step(x_array[:,i],v_array[:,i],ode_func,dt,mass,G,force_eval = [])

    #Calculate energy to get r
    _,r_1 = total_energy(x_array_test,v_array_test,mass,G)
    #### Recalculate dt at future timestep
    ## Finding CFL dt values
    del_t = np.ones((n_bodies,n_bodies))*10e3
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
    if i > 0:
        x_array[:,i+1],v_array[:,i+1],force_eval = Leapfrog_step(x_array[:,i],v_array[:,i],ode_func,dt,mass,G,force_eval)
    else: 
        x_array[:,i+1],v_array[:,i+1],force_eval = Leapfrog_step(x_array[:,i],v_array[:,i],ode_func,dt,mass,G,force_eval=[])
    ## Track conserved quantities
    L[:,i+1] = angular_momentum(x_array[:,i+1],v_array[:,i+1],mass)
    E[i+1],r = total_energy(x_array[:,i+1],v_array[:,i+1],mass,G)

    ## Stop if position or energy error grows too large
    final_it = i+1 # Final iteration
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
        elif dt < timestep_size:
            # If timestep size goes below the threshold
            print('Timestep size is below {}, so terminated at time {}'.format(timestep_size,dt_CFL_prog[i+1]))
            break
        elif i > timestep_count:
            # If too many timesteps
            print('More than {} timesteps, so terminated at time {}'.format(timestep_count,dt_CFL_prog[i+1]))
            break

    if np.max(dt_CFL_prog) > duration:
        break 

t1 = time.time() - t0    
print('Time for computation: {} seconds'.format(t1))
print(i)
print(np.max(dt_CFL_prog))
   
################################ Visualise ################################
#visualise(x_array,mass,vis_type,dims) 

if plot_more == 1:
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
    ax0.plot(x_array[elements,0],x_array[elements+1,0],'ko',ms=10)
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
    fig,ax = plt.subplots()
    ax.plot(np.arange(final_it),dt_CFL_store[:final_it])
    ax.ticklabel_format(useOffset=False)
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
    #plt.plot(np.arange(final_it+1),L_z_error)
    plt.plot(t_axis,L_z_error)
    plt.xlabel('time')
    #plt.xlabel('steps')
    plt.ylabel('L_z error')
    plt.title('Relative error of $L_z$ against time')
    plt.show()

elif plot_more == 2:
    t_axis = dt_CFL_prog[:final_it+1]
    ### Trajectory
    fig,ax0 = plt.subplots()
    #gs = fig.add_gridspec(2, 2)
    # x,y plane
    ax0.axis('equal')
    ax0.set_title('x-y plane')
    elements = np.arange(start = 0,stop = (n_bodies)*3,step = 3).astype(int)
    ax0.plot(x_array[0,:final_it+1],x_array[1,:final_it+1],'k')
    ax0.plot(x_array[elements,0],x_array[elements+1,0],'ko',ms=10)
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    #labels = list(map(str,mass))
    #ax0.legend(labels,title = 'Masses of bodies')
    plt.show()

if sim_type == '3':
    r1 = x_array[:3,:final_it]
    r2 = x_array[3:6,:final_it]
    r3 = x_array[6:,:final_it]
    R1 = r1 - r2
    R2 = r2 - r3
    R3 = r1 - r3
    N = np.linalg.norm(R1,axis = 0) + np.linalg.norm(R2,axis = 0) + np.linalg.norm(R3,axis = 0)
    X1 = R1/N
    X2 = R2/N
    X3 = R3/N
    X1_norm = np.linalg.norm(X1,axis = 0)
    X2_norm = np.linalg.norm(X2,axis = 0)
    X3_norm = np.linalg.norm(X3,axis = 0)
    # These sum to 1 for all time

    fig,ax = plt.subplots()
    ax.plot(X1_norm,X2_norm)
    ax.ticklabel_format(useOffset=False)
    plt.xlabel('|X1|')
    plt.ylabel('|X2|')
    plt.show()

    fill_matrix = np.zeros((1000,1000))
    X1_round = np.ceil(X1_norm*1000).astype(int) - 1
    X2_round = np.ceil(X2_norm*1000).astype(int) - 1
    for i in range(len(X1_round)):
        fill_matrix[X1_round[i],X2_round[i]] = 1
    #print(fill_matrix)

    print(np.sum(fill_matrix))

elif sim_type == '4':
    r1 = x_array[:3,:final_it]
    r2 = x_array[3:6,:final_it]
    r3 = x_array[6:9,:final_it]
    r4 = x_array[9:,:final_it]
    R1 = r1 - r2
    R2 = r1 - r3
    R3 = r1 - r4
    R4 = r4 - r2
    N = np.linalg.norm(R1,axis = 0) + np.linalg.norm(R2,axis = 0) + np.linalg.norm(R3,axis = 0) + np.linalg.norm(R4,axis = 0)
    X1 = R1/N
    X2 = R2/N
    X3 = R3/N
    X4 = R4/N
    X1_norm = np.linalg.norm(X1,axis = 0)
    X2_norm = np.linalg.norm(X2,axis = 0)
    X3_norm = np.linalg.norm(X3,axis = 0)
    X4_norm = np.linalg.norm(X4,axis = 0)
    # These sum to 1 for all time

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X1_norm,X2_norm,X3_norm)
    ax.ticklabel_format(useOffset=False)
    #plt.xlabel('|X1|')
    #plt.ylabel('|X2|')
    plt.show()

    
