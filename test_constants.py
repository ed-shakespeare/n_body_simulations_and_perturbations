import numpy as np

def CoM_pos(x,mass):
    ''' Provide the centre of mass for a given set of positions and masses
    Input
        x is current positions of bodies, in an array repeating x,y,z,x,y,z,...
        mass is the masses of the bodies
    Output
        CoM is the current centre of mass
    '''
    x = x.reshape((3,-1),order = 'F')
    CoM = np.sum(x*mass,axis = 1)/np.sum(mass)
    return CoM

def potential_energy(x,mass):
    ''' Provides the potential energy between all pairs of bodies
    Input  
        x is the current positions of all bodies, as a 3 x n array
        mass is the masses of the bodies
    Output
        U is a n x n array of potential energies    
    '''
    x = x.reshape((3,-1),order = 'F')
    n_bodies = x.shape[1]
    G = 1
    U = np.empty((n_bodies,n_bodies))
    for i in range(n_bodies):
        for j in range(i,n_bodies):
            # Loop over each pair once and then dupilicate, so save computing each time
            if i == j:
                pass
            else:
                r = x[:,i] - x[:,j]
                U[i,j] = G*mass[i]*mass[j]/np.linalg.norm(r)
                U[j,i] = U[i,j]
    return U   

def kinetic_energy(v,mass):
    ''' Provides the kinetic energy of each body at the current time
    Input
        v is array of current velocities of all bodies, as a 3 x n array
        mass is the masses of all bodies
    Output
        K is all current kinetic energies    
    '''
    v = v.reshape((3,-1),order = 'F')
    n_bodies = v.shape[1]
    K = np.empty(n_bodies)
    for i in range(n_bodies):
        K[i] = (1/2) * mass[i] * np.linalg.norm(v[:,i])**2
    return K 

def total_energy(x,v,mass):
    ''' Provides the total energy of the system, given the potential and kinetic energy of each body
    Input
        x is the current positions of all bodies, as a 3 x n array
        v is array of current velocities of all bodies, as a 3 x n array
        mass is the masses of all bodies
    Output
        E is total energy of system
    '''
    x = x.reshape((3,-1),order = 'F')
    v = v.reshape((3,-1),order = 'F')
    # Gravitational Potential Energy sum
    U = np.sum(potential_energy(x,mass))
    # Kinetic energy
    K = np.sum(kinetic_energy(v,mass))
    # Total
    #if U > 10000 or K > 10000:
    #    print(U,K)
    #    raise Exception('Energy error')
    E = U + K
    return E

def angular_momentum(x,v,mass):
    ''' Provides the angular momentum of all the bodies at the current time
    Input
        x is the current positions of all bodies, as a 3 x n array
        v is array of current velocities of all bodies, as a 3 x n array
        mass is the masses of all bodies
    Output
        L is the angular momentum of each body 
    '''
    x = x.reshape((3,-1),order = 'F')
    v = v.reshape((3,-1),order = 'F')
    n_bodies = x.shape[1]
    CoM = CoM_pos(x,mass).reshape((3,-1))
    p = v*mass # Momentum
    r = x - CoM # All r vectors
    L = 0
    for i in range(n_bodies):
        L += np.cross(r[:,i],p[:,i]) # Angular momentum
    return L

def eccentricity(x,v,mass):
    ''' Gives the eccentricity of the system 
    Input
        x is current positions of bodies, in an array repeating x,y,z,x,y,z,...
        v is current velocities of bodies, in an array repeating x,y,z,x,y,z,...
        mass is the masses of the bodies
    Output
        e is the eccentricity of the system as the current time
    '''
    G = 1
    red_m = np.sum(np.reciprocal(mass)) # Reduced mass

    ### Specific angular momentum
    L = angular_momentum(x,v,mass)
    h = L/red_m # Specific angular momentum
    
    ### Specific total energy
    E = total_energy(x,v,mass)
    # So STE is...
    Eps = E/red_m
    
    ### Standard gravitational parameter
    mu = G*np.sum(mass)

    ### Eccentricity
    e = np.sqrt(1 + (2*Eps*np.linalg.norm(h)**2)/(mu**2))

    return e

        