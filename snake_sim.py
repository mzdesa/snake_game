import numpy as np
import CalSim as cs
from snake_dyn import *
from snake import *
from snake_controller import *

#create a snake
N = 3
m = 0.1
l = 0.1
J = m*(l/2)**2
snake = SnakeV2(N, mVal = m, lVal = l, JVal= J)

#define an initial condition for the snake
x0 = np.zeros((2*(N + 2), 1))
# x0 = np.array([[0, 0, 0, np.pi/2, 0, 0, 0, 0]]).T

#create a snake dynamics object using CalSim
snakeDyn = SnakeDyn(x0, snake)

#create an observer
observerManager = cs.ObserverManager(snakeDyn)

#create a snake controller manager
controllerManager = cs.ControllerManager(observerManager, SnakeUnderact)

#create a snake environment
env = cs.Environment(snakeDyn, controllerManager, observerManager, T = 15)

#run the simulation
env.run()