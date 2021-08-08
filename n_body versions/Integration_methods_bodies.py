import numpy as np


def Euler_step(x_initial,v_initial,f,dt,mass,G):
    ''' Single step of Euler method
    Input 
        x_initial is all positions at current iteration
        v_initial is all velocities at current iteration
        f is vector of functions for position [0] and velocity [1]
        dt is stepsize
        mass is masses of all bodies
    Output
        x_final is all positions at next iteration
        v_final is all velocities at next iteration
    '''   
    n_bodies = int(len(x_initial)/3)
    x_final = np.empty_like(x_initial)
    v_final = np.empty_like(v_initial)
    for j in range(n_bodies):
        # Position 
        x_final[j*3:(j+1)*3] = x_initial[j*3:(j+1)*3] + f(0,x_initial[j*3:(j+1)*3],v_initial[j*3:(j+1)*3],np.delete(x_initial,(j*3,j*3+1,j*3+2),0),np.delete(mass,j),G)*dt 
    for j in range(n_bodies):
        # Velocity 
        v_final[j*3:(j+1)*3] = v_initial[j*3:(j+1)*3] + f(1,x_initial[j*3:(j+1)*3],v_initial[j*3:(j+1)*3],np.delete(x_initial,(j*3,j*3+1,j*3+2),0),np.delete(mass,j),G)*dt
    return x_final, v_final

def Euler_Cromer_step(x_initial,v_initial,f,dt,mass,G):
    ''' Single step of Euler-Cromer method
    Input 
        x_initial is all positions at current iteration
        v_initial is all velocities at current iteration
        f is vector of functions for position [0] and velocity [1]
        dt is stepsize
        mass is masses of all bodies
    Output
        x_final is all positions at next iteration
        v_final is all velocities at next iteration
    '''
    n_bodies = int(len(x_initial)/3)
    x_final = np.empty_like(x_initial)
    v_final = np.empty_like(v_initial)
    for j in range(n_bodies):
        # Velocity 
        v_final[j*3:(j+1)*3] = v_initial[j*3:(j+1)*3] + f(1,x_initial[j*3:(j+1)*3],v_initial[j*3:(j+1)*3],np.delete(x_initial,(j*3,j*3+1,j*3+2),0),np.delete(mass,j),G)*dt
    for j in range(n_bodies):
        # Position 
        x_final[j*3:(j+1)*3] = x_initial[j*3:(j+1)*3] + f(0,x_initial[j*3:(j+1)*3],v_final[j*3:(j+1)*3],np.delete(x_initial,(j*3,j*3+1,j*3+2),0),np.delete(mass,j),G)*dt 
    return x_final, v_final

def Leapfrog_step(x_initial,v_initial,f,dt,mass,G):
    ''' Single step of Leapfrog method
    Input 
        x_initial is all positions at current iteration
        v_initial is all velocities at current iteration
        f is vector of functions for position [0] and velocity [1]
        dt is stepsize
        mass is masses of all bodies
    Output
        x_final is all positions at next iteration
        v_final is all velocities at next iteration
    '''   
    n_bodies = int(len(x_initial)/3)
    x_final = np.empty_like(x_initial)
    v_inter = np.empty_like(v_initial)
    v_final = np.empty_like(v_initial)
    for j in range(n_bodies):
        # Intermediate velocity 
        v_inter[j*3:(j+1)*3] = v_initial[j*3:(j+1)*3] + f(1,x_initial[j*3:(j+1)*3],v_initial,np.delete(x_initial,(j*3,j*3+1,j*3+2),0),np.delete(mass,j),G)*dt/2
    for j in range(n_bodies):
        # Position 
        x_final[j*3:(j+1)*3] = x_initial[j*3:(j+1)*3] + f(0,x_initial[j*3:(j+1)*3],v_inter[j*3:(j+1)*3],np.delete(x_initial,(j*3,j*3+1,j*3+2),0),np.delete(mass,j),G)*dt 
    for j in range(n_bodies):
        # Final velocity 
        v_final[j*3:(j+1)*3] = v_inter[j*3:(j+1)*3] + f(1,x_final[j*3:(j+1)*3],v_inter[j*3:(j+1)*3],np.delete(x_final,(j*3,j*3+1,j*3+2),0),np.delete(mass,j),G)*dt/2 
    return x_final, v_final