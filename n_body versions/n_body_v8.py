import time 
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.core import shape_base
from numpy.core.fromnumeric import shape
from Integration_methods_bodies import Leapfrog_step
from test_constants import CoM_pos, CoM_vel, angular_momentum, total_energy, kinetic_energy
from ode_function import ode_func

### Changing from timesteps to time, so for loop to while loop

### Using just fig-8, starting to scan range of initial conditions
### This will just perturb the x velocity of the centre particle, and adjust others accordingly
### Including x perturbation too now


################################ Setup ################################ 
######## CHANGE VALUES BELOW ########
# Setup values
# Approx 16 orbits per 100 time units
duration = 450 # time units, not seconds
dt = 0.001 # Not needed if CFL stepping, but needs to be declared

## Courant number
C = 0.1

# Energy error threshold
e_thresh = 0.05
box_size = 10

# Gravitational constant
G = 1
#G = 6.67408e-11

# Visualisation
#vis_type = 'graph'
#dims = 2
#plot_more = True
######## CHANGE VALUES ABOVE ########


def perturb_fig_8(x_perturbation = 0,v_perturbation = 0,plot_more = False):
    '''
    Runs the simulation with a perturbation to initial conditions
    Defaults to no perturbation
    '''
    # Fig 8
    mass = (1,1,1) # Equal masses
    x_IC_x = 0.97000436
    x_IC_y = 0.24308753
    v_IC_x = -0.93240737
    v_IC_y = -0.86473146
    x_ICs = (x_IC_x,-x_IC_y,0,-x_IC_x,x_IC_y,0,0,0,0) # x,y,z,x,y,z,...
    v_ICs = (-v_IC_x/2,-v_IC_y/2,0,-v_IC_x/2,-v_IC_y/2,0,v_IC_x,v_IC_y,0) # x,y,z,x,y,z,...
    x_ICs = np.asarray(x_ICs)
    v_ICs = np.asarray(v_ICs)
    ###### Perturb the x_ICs
    x_perturb_x = -x_IC_x + x_perturbation
    x_perturb_y = np.sqrt(x_ICs[3]**2 + x_ICs[4]**2-x_perturb_x**2)
    x_ICs = (-x_perturb_x,-x_perturb_y,0,x_perturb_x,x_perturb_y,0,0,0,0) # x,y,z,x,y,z,...
    ###### Perturb the v_ICs
    v_perturb_x = v_ICs[6]+v_perturbation
    v_perturb_y = -np.sqrt(v_ICs[6]**2 + v_ICs[7]**2-v_perturb_x**2)
    v_ICs = (-v_perturb_x/2,-v_perturb_y/2,0,-v_perturb_x/2,-v_perturb_y/2,0,v_perturb_x,v_perturb_y,0)

    n_bodies = round(len(x_ICs)/3)

    # Check number of bodies is consistent
    if len(x_ICs) != len(v_ICs) or len(x_ICs) != len(mass)*3:
        raise Exception('Position and velocity IC vectors must both have 3 times the length of the masses vector')

    x_array = np.empty((3*n_bodies,1))
    v_array = np.empty((3*n_bodies,1))
    x_array[:,0] = np.transpose(x_ICs)
    v_array[:,0] = np.transpose(v_ICs)

    ##  Tracking 'conserved' quantities, and reset to CoM frame
    # CoM position
    CoM = np.empty((3,1))
    CoM[:,0] = CoM_pos(x_array[:,0],mass) # Initial CoM
    # CoM velocity
    CoM_v = np.empty((3,1))
    CoM_v[:,0] = CoM_vel(v_array[:,0],mass) # Initial CoM

    # In centre of mass frame
    CoM_pos_array = np.tile(CoM,(n_bodies,1))
    x_array[:,0] = x_array[:,0] - CoM_pos_array[:,0]
    CoM_vel_array = np.tile(CoM_v,(n_bodies,1))
    v_array[:,0] = v_array[:,0] - CoM_vel_array[:,0]

    # Angular momentum
    L = np.empty((3,1))
    L[:,0] = angular_momentum(x_array[:,0],v_array[:,0],mass) # Initial angular momentum
    # Energy
    E = np.empty(1)
    E[0],r = total_energy(x_array[:,0],v_array[:,0],mass,G) # Initial energy

    ## These are similar to above, except for just CFL values
    ## dt_CFL_prog is for plotting over time
    dt_CFL_prog = np.empty(1)
    dt_CFL_prog[0] = 0
    ## dt_CFL_store is storing each dt
    dt_CFL_store= []

    ################################ Run method ################################
    #t0 = time.time()
    i = -1
    while dt_CFL_prog[-1] < duration:
        i += 1
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
        print(del_t)
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
        dt_CFL_prog = np.append(dt_CFL_prog,dt_CFL_prog[i] + dt_CFL) # This is cumulative vector of dt
        dt_CFL_store = np.append(dt_CFL_store,dt_CFL) # This stores dt at each time
        dt = dt_CFL
        # Update position and velocity 
        x_array_hold,v_array_hold = Leapfrog_step(x_array[:,i],v_array[:,i],ode_func,dt,mass,G)
        x_array = np.append(x_array,np.reshape(np.asarray(x_array_hold),(-1,1)),axis=1)
        v_array = np.append(v_array,np.reshape(np.asarray(v_array_hold),(-1,1)),axis=1)
        ## Track conserved quantities
        L = np.append(L,np.reshape(np.asarray(angular_momentum(x_array[:,i+1],v_array[:,i+1],mass)),(-1,1)),axis=1)
        E_hold,r = total_energy(x_array[:,i+1],v_array[:,i+1],mass,G)
        E = np.append(E,E_hold)
        ## Stop if position or energy error grows too large
        final_it = i+1 # Final iteration
        if np.mod(i,100) == 0:
            print(x_array_hold)
            # Every 100 steps, check
            if any(np.abs(x_array[:,i+1]) > box_size):
                # If position goes outside square/cube of side length 2*box_size, centred on origin
                print('Outside box of size {}, so terminated at time {}'.format(box_size,dt_CFL_prog[i+1]))
                shape_value = dt_CFL_prog[i+1]
                return shape_value, E[0], L[2,0]

            elif np.abs((E[i+1] - E[0])/E[0]) > e_thresh:
                # If energy error goes above the threshold
                print('Rel energy error is above {}, so terminated at time {}'.format(e_thresh,dt_CFL_prog[i+1]))
                shape_value = dt_CFL_prog[i+1]
                return shape_value, E[0], L[2,0]

            elif dt < 10e-4:
                # If timestep size goes below the threshold
                print('Timestep size is below {}, so terminated at time {}'.format(dt,dt_CFL_prog[i+1]))
                shape_value = dt_CFL_prog[i+1]
                return shape_value, E[0], L[2,0]

    #t1 = time.time() - t0    
    #print('Real time for computation: {} seconds'.format(t1))

       
    ################################ Visualise ################################
    #visualise(x_array,mass,'animation',2) 
    # Want this false for grid search simulations
    if plot_more:
        t_axis = dt_CFL_prog[:final_it+1]
        ### Trajectory
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)
        # x,y plane
        ax0 = fig.add_subplot(gs[0, :])
        ax0.set_title('x-y plane')
        elements = np.arange(start = 0,stop = (n_bodies)*3,step = 3).astype(int)
        ax0.plot(np.transpose(x_array[elements,:final_it+1]),np.transpose(x_array[elements+1,:final_it+1]))
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        labels = list(map(str,mass))
        ax0.legend(labels,title = 'Masses of bodies')
        # x and y against t
        ax1 = fig.add_subplot(gs[1, :-1])
        ax1.set_title('x against t')
        for i in range(n_bodies):
            ax1.plot(t_axis,x_array[i*3,:final_it+1])
        ax1.set_xlabel('time')
        ax1.set_ylabel('x')
        #
        ax2 = fig.add_subplot(gs[1:, -1])
        ax2.set_title('y against t')
        for i in range(n_bodies):
            ax2.plot(t_axis,x_array[(i*3) + 1,:final_it+1])
        ax2.set_xlabel('time')
        ax2.set_ylabel('y')
        fig.tight_layout()
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

    shape_value = duration  #If it reaches the end, it stays as a fig-8
    print('Ran for full duration (finish time: {})'.format(dt_CFL_prog[final_it]))
    return shape_value, E[0], L[2,0]

gridsearch = False
if gridsearch:
    perturb_1 = np.linspace(-0.02,0.02,3)
    perturb_2 = np.linspace(-0.02,0.02,3)
    #perturb_2 = (0,1)

    shape_value = np.empty((len(perturb_1),len(perturb_2)))
    E = np.empty((len(perturb_1),len(perturb_2)))
    L_z = np.empty((len(perturb_1),len(perturb_2)))

    for i in range(len(perturb_1)):
        for j in range(len(perturb_2)):
            shape_value[i,j], E[i,j], L_z[i,j] = perturb_fig_8(x_perturbation = perturb_1[i],v_perturbation = perturb_2[j],plot_more = False)
    print('Perturbation = {}'.format(perturb_1))
    print('Shape_value = {}'.format(shape_value))
    print('Energy = {}'.format(E))
    print('Ang mom = {}'.format(L_z))

    plt.pcolormesh(perturb_1,perturb_2,shape_value,shading='nearest',cmap='Greens')
    plt.xlabel('Perturbation of x position of particle 2') # Particle 2 is on the top left of fig 8 starting positions
    plt.ylabel('Perturbation of x component of centre particle velocity')
    plt.show()
    #-0.02692308 for x perturb seemed stableish but not for as long as originally thought
else:
    # Single perturbation, for testing
    x_perturbation = 0.0
    v_perturbation = 0.0
    plot_more = True
    _,_,_ = perturb_fig_8(x_perturbation,v_perturbation,plot_more)