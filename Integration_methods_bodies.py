import numpy as np


def Euler_step(x_initial,v_initial,t,f,dt,x_other,mass):
    ''' Single step of Euler method
    Input 
        x_initial is position at current iteration
        v_initial is velocity at current iteration
        t is current time
        f is vector of functions for position [0] and velocity [1]
        dt is stepsize
        x_other is position of other bodies
    Output
        x_final is position at next iteration
        v_final is velocity at next iteration
    '''
    # Position 
    x_final = x_initial + f(0,x_initial,v_initial,t,x_other,mass)*dt
    # Velocity 
    v_final = v_initial + f(1,x_initial,v_initial,t,x_other,mass)*dt
    return x_final, v_final

def Euler_Cromer_step(x_initial,v_initial,t,f,dt,x_other,mass):
    ''' Single step of Euler-Cromer method
    Input 
        x_initial is position at current iteration
        v_initial is velocity at current iteration
        t is current time
        f is vector of functions for position [0] and velocity [1]
        dt is stepsize
        x_other is position of other bodies
    Output
        x_final is position at next iteration
        v_final is velocity at next iteration
    '''
  
    # Velocity 
    v_final = v_initial + f(1,x_initial,v_initial,t,x_other,mass)*dt
    # Position 
    x_final = x_initial + f(0,x_initial,v_final,t,x_other,mass)*dt
    return x_final, v_final

def Leapfrog_step(x_initial,v_initial,t,f,dt,x_other,mass):
    ''' Single step of Leapfrog method
    Input 
        x_initial is position at current iteration
        v_initial is velocity at current iteration
        t is current time
        f is vector of functions for position [0] and velocity [1]
        dt is stepsize
        x_other is position of other bodies
    Output
        x_final is position at next iteration
        v_final is velocity at next iteration
    '''   
    # Intermediate velocity 
    v_inter = v_initial + f(1,x_initial,v_initial,t,x_other,mass)*dt/2
    # Position 
    x_final = x_initial + f(0,x_initial,v_inter,t,x_other,mass)*dt 
    # Final velocity 
    v_final = v_inter + f(1,x_final,v_inter,t,x_other,mass)*dt/2 
    return x_final, v_final