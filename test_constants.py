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

    x = x.reshape((3,-1),order = 'F')
    

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
    CoM = CoM_pos(x,mass).reshape((3,-1))
    x = x.reshape((3,-1),order = 'F')
    v = v.reshape((3,-1),order = 'F')
    p = v*mass # Momentum
    r = x - CoM
    L = 0
    for i in range(p.shape[1]):
        L += np.cross(r[:,i],p[:,i]) # Angular momentum
    red_m = np.sum(np.reciprocal(mass)) # Reduced mass
    h = L/red_m # Specific angular momentum
    
    


    pass
#eccentricity(np.transpose((1,2,3,0,0,0)),np.transpose((1,1,1,1,1,1)),[1,1])