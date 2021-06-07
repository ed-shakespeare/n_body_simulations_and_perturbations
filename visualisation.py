import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def visualise(x_array,mass,vis_type = 'graph',dims = 2,CoM = [],CoM_frame = False):
    ''' Takes an array of trajectories and plots as either a graph or an animation
    Input
        x_array is numpy array of trajectories. Each row is a coordinate, starting x,y,z,x,y,z,...
        mass is the masses of the bodies
        vis_type is a string with the type of visualisation required (default = 'graph')
        dims is number of dimensions of output (2 or 3), where if 2 is selected, 
                only x and y will be plotted (default = 2)
    Output
        The visualisation
    
    '''
    # Find number of bodies, so know number of orbits to visualise
    n_rows,_ = x_array.shape
    n_bodies = int(n_rows/3)
    elements = np.arange(start = 0,stop = (n_bodies)*3,step = 3).astype(int)

    # For plotting in CoM frame, adjust coords
    if CoM_frame:
        CoM_array = np.tile(CoM,(n_bodies,1))
        x_array = x_array - CoM_array
        CoM = CoM - CoM # Trivial, but clear to show what's happening

    ### Graph
    if vis_type == 'graph':
        if dims == 2:
            if CoM != []:
                ### Include CoM in plot
                plt.plot(np.transpose(x_array[elements,:]),np.transpose(x_array[elements+1,:]))
                plt.plot(np.transpose(CoM[0,:]),np.transpose(CoM[1,:]))
                plt.xlabel('x')
                plt.ylabel('y')
                labels = list(map(str,mass))
                labels.append('CoM')
                plt.legend(labels,title = 'Masses of bodies')
                plt.show()
            else:
                plt.plot(np.transpose(x_array[elements,:]),np.transpose(x_array[elements+1,:]))
                plt.xlabel('x')
                plt.ylabel('y')
                labels = list(map(str,mass))
                plt.legend(labels,title = 'Masses of bodies')
                plt.show()
        elif dims == 3:
            if CoM != []:
                ### Include CoM in plot
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                for i in range(n_bodies):
                    plt.plot(np.transpose(x_array[elements[i],:]),np.transpose(x_array[elements[i]+1,:]),np.transpose(x_array[elements[i]+2,:]))
                plt.plot(np.transpose(CoM[0,:]),np.transpose(CoM[1,:]),np.transpose(CoM[2,:]))
                plt.xlabel('x')
                plt.ylabel('y')
                ax.set_zlabel('z')
                labels = list(map(str,mass))
                labels.append('CoM')
                plt.legend(labels,title = 'Masses of bodies')
                plt.show()
            else:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                for i in range(n_bodies):
                    plt.plot(np.transpose(x_array[elements[i],:]),np.transpose(x_array[elements[i]+1,:]),np.transpose(x_array[elements[i]+2,:]))
                plt.xlabel('x')
                plt.ylabel('y')
                ax.set_zlabel('z')
                labels = list(map(str,mass))
                plt.legend(labels,title = 'Masses of bodies')
                plt.show()

    ### Animation
    elif vis_type == 'animation':
        colors = ['red', 'green', 'blue', 'orange','black','grey','cyan','olive','pink','brown'] # Will cycle through these, for n > 10
        if dims == 2:
            fig = plt.figure()
            plt.xlabel('x')
            plt.ylabel('y')
            # Uncomment this to set the axes at the maximum from the start
            #plt.xlim(((np.min(x_array[elements,:]),np.max(x_array[elements,:]))))
            #plt.ylim(((np.min(x_array[elements+1,:]),np.max(x_array[elements+1,:]))))
            def graph_animation(i):
                p = plt.plot(np.transpose(x_array[elements,:i]),np.transpose(x_array[elements+1,:i])) 
                for j in range(n_bodies):
                    p[j].set_color(colors[j%10]) 

            animator = FuncAnimation(fig, graph_animation, interval = 0.001)
            plt.show()
        elif dims == 3:
            #raise Exception('Animation currently only 2d')
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            plt.xlabel('x')
            plt.ylabel('y')
            ax.set_zlabel('z')
            def graph_animation(i):
                for j in range(n_bodies):
                    ### FuncAnimation slows down over time, so jump between frames as i increases. 
                    plt.plot(np.transpose(x_array[elements[j],:int(np.sqrt(i)*i)]),np.transpose(x_array[elements[j]+1,:int(np.sqrt(i)*i)]),np.transpose(x_array[elements[j]+2,:int(np.sqrt(i)*i)]),color = colors[j%10])
              
            animator = FuncAnimation(fig, graph_animation, interval = 0.01)
            plt.show()
    else:
        raise Exception('Possible visualisations are graph and animation')

        