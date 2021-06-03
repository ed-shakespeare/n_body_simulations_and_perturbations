import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from Integration_methods_bodies import Euler_step,Euler_Cromer_step,Leapfrog_step
from visualisation import visualise

# Trial function
def f(component,x,v,t,x_other,mass = np.NAN):
    ''' Functions for dynamical system. Contains position and velocity
    Input
        component is for choosing position or velocity
        x is current position
        v is current velocity
        t is current time
        x_other is current position of other bodies
        masses is the masses of the other bodies
    Output
        func_value is value of function
    '''
    n_other = int(len(x_other)/3) # Number of other bodies
    G = 1
    if mass == np.NAN:
        # If mass has not been set, use 1 for every mass
        mass = np.ones(len(n_other))
    func_value = np.zeros_like(x) # This will become vector of length 3, like xdot or vdot
    
    # Position
    if component == 0:
        func_value = v
        return func_value
    
    # Velocity
    elif component == 1:
        for i in range(n_other):
            m = mass[i]
            r = x - x_other[i*3:(i+1)*3]
            r_len = np.linalg.norm(r)
            r_hat = r / r_len
            func_value += -(G*m/(r_len**2)) * r_hat
        return func_value


### Testing 
# Setup values
n_steps = 10000
t = 0 # Start time
dt = 0.01

# Setup array for trajectories
x_ICs = (0,0,0,4,0,4) # x,y,z,x,y,z,...
v_ICs = (0,0,0,0,5,0) # x,y,z,x,y,z,...
masses = (150,1)
if len(x_ICs) != len(v_ICs) or len(x_ICs) != len(masses)*3:
    raise Exception('Position and velocity IC vectors must both have 3 times the length of the masses vector')
n_bodies = len(masses)
x_array = np.empty((3*n_bodies,n_steps+1))
v_array = np.empty((3*n_bodies,n_steps+1))
x_array[:,0] = np.transpose(x_ICs)
v_array[:,0] = np.transpose(v_ICs)

# Run method
for i in range(n_steps):
    for j in range(n_bodies):
        x_array[j*3:(j+1)*3,i+1],v_array[j*3:(j+1)*3,i+1] = Euler_step(x_array[j*3:(j+1)*3,i],v_array[j*3:(j+1)*3,i],t,f,dt,np.delete(x_array,(j*3,j*3+1,j*3+2),0)[:,i],mass = np.delete(masses,j))
        # np.delete is technically slow but hopefully good enough.
    t += dt
   
### Visualising
visualise(x_array,masses,'graph',dims = 3)