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

def eccentricity(x,v,mass,G):
    ''' Gives the eccentricity of the system 
    Input
        x is current positions of bodies, in an array repeating x,y,z,x,y,z,...
        v is current velocities of bodies, in an array repeating x,y,z,x,y,z,...
        mass is the masses of the bodies
    Output
        e is the eccentricity of the system as the current time
    '''
    red_m = 1/np.sum(np.reciprocal(mass)) # Reduced mass

    ### Specific angular momentum
    L = angular_momentum(x,v,mass)
    h = L/red_m # Specific angular momentum
    
    ### Specific total energy
    E,_ = total_energy(x,v,mass,G)
    # So STE is...
    Eps = E/red_m
    
    ### Standard gravitational parameter
    mu = G*np.sum(mass)

    ### Eccentricity
    e = np.sqrt(1 + (2*Eps*np.linalg.norm(h)**2)/(mu**2))

    return e

    
### Eccentricity and energy over time
#fig, (ax1, ax2) = plt.subplots(nrows=2)
# Plot eccentricity over time
#ax1.plot(np.arange(n_steps+1)*dt,e)
#ax1.set_xlabel('time')
#ax1.set_ylabel('eccentricity')

# Plot energy over time
#ax2.plot(np.arange(n_steps+1)*dt,E)
#ax2.set_xlabel('time')
#ax2.set_ylabel('energy')
#plt.show()


#### Plotting angular momentum
#fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
#plt.title('Angular momentum')
#ax0.plot(np.arange(n_steps+1)*dt,L[0,:])
#ax0.set_xlabel('time')
#ax0.set_ylabel('L_x')
#ax1.plot(np.arange(n_steps+1)*dt,L[1,:])
#ax1.set_xlabel('time')
#ax1.set_ylabel('L_y')
#ax2.plot(np.arange(n_steps+1)*dt,L[2,:])
#ax2.set_xlabel('time')
#ax2.set_ylabel('L_z')
#plt.show()