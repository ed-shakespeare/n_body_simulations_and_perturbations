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

def CoM_vel(v,mass):
    ''' Provide the centre of mass for a given set of positions and masses
    Input
        v is current positions of bodies, in an array repeating x,y,z,x,y,z,...
        mass is the masses of the bodies
    Output
        CoM is the current centre of mass
    '''
    v = v.reshape((3,-1),order = 'F')
    CoM_v = np.sum(v*mass,axis = 1)/np.sum(mass)
    return CoM_v

def potential_energy(x,mass,G):
    ''' Provides the potential energy between all pairs of bodies
    Input  
        x is the current positions of all bodies, in an array repeating x,y,z,x,y,z,...
        mass is the masses of the bodies
    Output
        U is a n x n array of potential energies    
        r is a n x n array of distances between bodies
    '''
    x = x.reshape((3,-1),order = 'F')
    n_bodies = x.shape[1]
    U = np.zeros((n_bodies,n_bodies))
    r = np.zeros((n_bodies,n_bodies))
    for i in range(n_bodies):
        for j in range(i+1,n_bodies):
            # Loop over each pair once and then dupilicate, so save computing each time
            r[i,j] = np.linalg.norm(x[:,i] - x[:,j])
            U[i,j] = -(1/2)*G*mass[i]*mass[j]/r[i,j]
            r[j,i] = r[i,j]
            U[j,i] = U[i,j]
    return U,r   

def kinetic_energy(v,mass):
    ''' Provides the kinetic energy of each body at the current time
    Input
        v is array of current velocities of all bodies, in an array repeating x,y,z,x,y,z,...
        mass is the masses of all bodies
    Output
        K is all current kinetic energies    
    '''
    v = v.reshape((3,-1),order = 'F')
    n_bodies = v.shape[1]
    K = np.zeros(n_bodies)
    for i in range(n_bodies):
        K[i] = (1/2) * mass[i] * np.linalg.norm(v[:,i])**2
    return K 

def total_energy(x,v,mass,G):
    ''' Provides the total energy of the system, given the potential and kinetic energy of each body
    Input
        x is the current positions of all bodies, in an array repeating x,y,z,x,y,z,...
        v is array of current velocities of all bodies, in an array repeating x,y,z,x,y,z,...
        mass is the masses of all bodies
    Output
        E is total energy of system
    '''
    # Gravitational Potential Energy sum
    U,r = potential_energy(x,mass,G)
    # Kinetic energy
    K = kinetic_energy(v,mass)
    # Total
    E = np.sum(U) + np.sum(K)
    return E,r

def angular_momentum(x,v,mass):
    ''' Provides the angular momentum of all the bodies at the current time
    Input
        x is the current positions of all bodies, in an array repeating x,y,z,x,y,z,...
        v is array of current velocities of all bodies, in an array repeating x,y,z,x,y,z,...
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
