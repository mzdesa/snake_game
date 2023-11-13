"""
Create a CalSim snake dynamics object
"""
import numpy as np
import CalSim as cs

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import seaborn as sns

class SnakeDyn(cs.Dynamics):

    def __init__(self, x0, snake):
        """
        Init function for a snake dynamics object
        Inputs:
            x0 (NumPy Array): initial condition of snake
            snake (Snake): Snake object
        """
        #extract the snake functions
        Mfunc = snake.Mfunc
        Rfunc = snake.Rfunc
        Upsilonfunc = snake.Upsilonfunc

        #get the dimension of the snake state vector from testing Upsilonfunc
        numGC = snake.N + 2 #get number of GCs for snake
        singleStateDimn = 2 * numGC
        singleInputDimn = 3 #input dimension is always 3 -> two force and one torque

        def f(xi, u, t):
            """
            State Space Snake dynamics function
            Inputs:
                xi: state vector
                u (2x1 NumPy Array): input vector
                t (float): current time
            """
            #Get, M, R, Upsilon
            M = Mfunc(xi)
            R = Rfunc(xi)
            Upsilon = Upsilonfunc(xi)

            #extract derivative terms from xi
            qDot = xi[numGC: , :].reshape((numGC, 1))

            #calculate second derivative terms
            qDDot = np.linalg.pinv(M) @ (-R + Upsilon @ u)

            #return xiDot
            return np.vstack((qDot, qDDot))

        #call the super init function in dynamics -> default to one snake
        super().__init__(x0, singleStateDimn, singleInputDimn, f, N = 1)

        #store the snake as an attribute
        self.snake = snake
    
    def show_animation(self, xData, uData, tData, animate = True, obsManager = None):
        #Set constant animtion parameters
        FREQ = 50 #control frequency, same as data update frequency

        #define function to calculuate XY position of each link at an index
        def linkPos(idx):
            """
            Calculate the positions of each endpoint for an index
            """
            #first, get (x, y) base position at the index
            linkArr = xData[0:2, idx].reshape((2, 1))

            #next, loop over the links
            for i in range(self.snake.N):
                #get the angle of link i
                theta = xData[i + 2, idx]
                pos = self.snake.l * np.array([[np.cos(theta), np.sin(theta)]]).T
                pos = pos + linkArr[:, -1].reshape((2, 1)) #add the next link position to this one
                linkArr = np.hstack((linkArr, pos))
            return linkArr
        
        fig = plt.figure()
        ax = fig.add_subplot(autoscale_on=False)
        ax.set_aspect('equal')
        offset = 0.5
        ax.set_xlim(xmin = -offset, xmax = offset)
        ax.set_ylim(ymin = -offset, ymax = offset)
        
        #define the line
        line, = ax.plot([], [], 'o-', lw=2)

        def update(idx):
            #compute the link array of all of the positions
            linkArr = linkPos(idx)

            #update the line
            thisX = [linkArr[0, i] for i in range(self.snake.N + 1)]
            thisY = [linkArr[1, i] for i in range(self.snake.N + 1)]

            line.set_data(thisX, thisY)

            #update axis limits
            if idx != num_frames - 1:
                ax.set_xlim(xmin = -offset + linkArr[0, 0], xmax = offset + linkArr[0, 0])
                ax.set_ylim(ymin = -offset + linkArr[1, 0], ymax = offset + linkArr[1, 0])
            else:
                #reset last frame to avoid repeat frames
                ax.set_xlim(xmin = -offset, xmax = offset)
                ax.set_ylim(ymin = -offset, ymax = offset)

            return line,

        num_frames = xData.shape[1]-1
        anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=1/FREQ*1000, blit=True)
        plt.show()