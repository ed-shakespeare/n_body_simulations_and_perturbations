import numpy as np
# Trial function
def ode_func(component,x,v,x_other,mass,G):
    ''' Functions for dynamical system. Contains position and velocity
    Input
        component is for choosing position or velocity
        x is current position
        v is current velocity
        x_other is current position of other bodies
        masses is the masses of the other bodies
        G is gravitational constant
    Output
        func_value is value of function
    '''
    
    func_value = np.zeros_like(x) # This will become vector of length 3, like xdot or vdot
    
    # Position
    if component == 0:
        func_value = v
        return func_value
    
    # Velocity
    elif component == 1:
        n_other = int(len(x_other)/3) # Number of other bodies
        for i in range(n_other):
            m = mass[i]
            r = x - x_other[i*3:(i+1)*3] # Vector between bodies
            r_len = np.linalg.norm(r) # Euclidean distance between bodies
            r_hat = r / r_len # Unit vector
            func_value += -(G*m/(r_len**2)) * r_hat
        return func_value

