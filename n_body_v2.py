import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from Integration_methods_bodies import Euler_step,Euler_Cromer_step,Leapfrog_step
from visualisation import visualise
from test_constants import CoM_pos
from ode_function import ode_func

### Testing 
# Setup values
n_steps = 10000
t = 0 # Start time
dt = 0.01

# Setup array for trajectories
x_ICs = (0,0,0,4,0,4) # x,y,z,x,y,z,...
v_ICs = (0,0,0,0,5,0) # x,y,z,x,y,z,...
n_bodies = int(len(x_ICs)/3)
mass = (150,1) # For equal weights of 1, use []
if mass == []:
    mass = np.ones(n_bodies)

if len(x_ICs) != len(v_ICs) or len(x_ICs) != len(mass)*3:
    raise Exception('Position and velocity IC vectors must both have 3 times the length of the masses vector')

x_array = np.empty((3*n_bodies,n_steps+1))
v_array = np.empty((3*n_bodies,n_steps+1))
x_array[:,0] = np.transpose(x_ICs)
v_array[:,0] = np.transpose(v_ICs)

CoM = np.empty((3,n_steps+1))
CoM[:,0] = CoM_pos(x_array[:,0],mass) # Initial CoM

# Run method
for i in range(n_steps):
    # For each body, update position and velocity
    for j in range(n_bodies):
        x_array[j*3:(j+1)*3,i+1],v_array[j*3:(j+1)*3,i+1] = Leapfrog_step(x_array[j*3:(j+1)*3,i],v_array[j*3:(j+1)*3,i],t,ode_func,dt,np.delete(x_array,(j*3,j*3+1,j*3+2),0)[:,i],mass = np.delete(mass,j))
        # np.delete is technically slow but hopefully good enough.
    ## Perform checks:
    # Centre of Mass position
    CoM[:,i+1] = CoM_pos(x_array[:,i+1],mass)

    t += dt
   

### Visualising
visualise(x_array,mass,'graph',dims = 3)


### Plotting centre of mass

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#for i in range(n_bodies):
#    plt.plot(np.transpose(CoM[0,:]),np.transpose(CoM[1,:]),np.transpose(CoM[2,:]))
#plt.xlabel('x')
#plt.ylabel('y')
#ax.set_zlabel('z')
##labels = list(map(str,mass))
##plt.legend(labels,title = 'Masses of bodies')
#plt.show()