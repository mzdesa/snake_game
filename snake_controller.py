"""
Define a controller for the snake
"""
import numpy as np
import CalSim as cs
import casadi as ca
from control import lqr
import osqp
from scipy import sparse
from CalSim import hat

class SnakeFF(cs.Controller):
    """
    Define a test feedforward snake controller
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a ff snake controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

    def eval_input(self, t):
        """
        Evaluate the input to a snake
        Inputs:
            t (float): current time in simulation
        """
        self._u = 0.1 * np.array([[np.sin(t), np.cos(t), 0.01]]).T
        return self._u
    

class SnakeFullState(cs.Controller):
    """
    Define a min-norm to a full state linearization controller
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a min norm to full state linearization controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #extract snake link number
        self.N = self.observer.dynamics.snake.N

        #define relative or absolute desired
        self.rel = False

        #solve LQR for gains
        Q = np.eye(2*(self.N + 2))
        R = np.eye(self.N + 2)
        Q[0, 0] = 0.25 #de-bias position
        Q[1, 1] = 0.25
        A1 = np.zeros((2*(self.N + 2), self.N + 2))
        A2 = np.vstack((np.eye(self.N + 2), np.zeros((self.N + 2, self.N + 2))))
        A = np.hstack((A1, A2))
        B = np.vstack((np.zeros((self.N + 2, self.N + 2)), np.eye(self.N + 2)))
        K, S, E = lqr(A, B, Q, R)
        self.K = K

    def get_des_state(self):
        """
        Evaluate the desired generalized coordinates
        """
        if self.rel:
            #use "relative" tracking
            xi = self.observer.get_state()

            #get current angles
            theta = xi[2: self.N + 2]

            #assemble desired angles
            thetaD = np.vstack((np.ones((1, 1)), theta[1:, :].reshape((self.N - 1, 1))))
        else:
            #use absolute tracking of pi
            piList = [np.pi*(i + 1) for i in range(self.N)]

            #assmeble desired angles
            thetaD = np.array([piList]).T

        #stack with desired (x, y) position and velocities -> set to zero for now
        return np.vstack((np.zeros((2, 1)), thetaD, np.zeros((self.N + 2, 1))))
    
    def get_config_error(self, theta, thetaD):
        #Returns error in angle using SO(2) formula
        return 1 - np.cos((thetaD - theta))

    def eval_input(self, t):
        """
        Evaluate the input to a snaks
        Inputs:
            t (float): current time in simulation
        """
        #get state vector from observer
        xi = self.observer.get_state()

        #get desired state
        xiD = self.get_des_state()
        q_d_ddot = np.zeros((self.N + 2, 1))

        #define error for full state xi
        eXi = xiD - xi
        eTheta = self.get_config_error(xi[2:self.N + 2], xiD[2:self.N + 2])
        eXi[2: self.N + 2] = eTheta #update with geom. tracking formula

        #first, calculate the matrices
        M = self.observer.dynamics.snake.Mfunc(xi)
        R = self.observer.dynamics.snake.Rfunc(xi)
        Upsilon = self.observer.dynamics.snake.Upsilonfunc(xi)

        #calculate nominal gamma using FB linearization
        Gamma = R + M @ (q_d_ddot + self.K @ eXi)

        #solve an optimization for u
        opti = ca.Opti()
        u = opti.variable(3, 1)

        #define cost function as min norm to Gamma input
        cost = (Gamma - Upsilon @ u).T @ (Gamma - Upsilon @ u)
        
        #set up optimizatoin
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve()
        
        #store the input
        self._u = sol.value(u).reshape((3, 1))
        return self._u
    

class SnakeMPC(cs.Controller):
    """
    Define an MPC snake controller
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a min norm to full state linearization controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #extract snake link number
        self.N = self.observer.dynamics.snake.N

        #define relative or absolute desired
        self.rel = False

        #solve LQR for gains
        self.Q = np.eye(2*(self.N + 2))
        self.R = np.eye(3)
        self.Q[0, 0] = 0.25 #de-bias position
        self.Q[1, 1] = 0.25

        #define discretization time step
        self.dt = 1/50 #run at 50 Hz

        #define MPC horizon using a lookahead time
        T = 1 #lookahead time
        self.Nmpc = int(T/self.dt)
        
        #Define f and g as casadi functions functions for control affine system
        Mfunc = self.observer.dynamics.snake.Mfunc
        Rfunc = self.observer.dynamics.snake.Rfunc
        Upsilonfunc = self.observer.dynamics.snake.Upsilonfunc
        Mcas = lambda xi: self.np_2_cas(Mfunc(xi))
        Rcas = lambda xi: self.np_2_cas(Rfunc(xi))
        Upcas = lambda xi: self.np_2_cas(Upsilonfunc(xi))
        self.f = lambda xi: ca.vertcat(xi[self.N + 2:], -ca.inv(Mcas(xi)) @ Rcas(xi))
        self.g = lambda xi: ca.vertcat(ca.MX.zeros(self.N + 2, 3), ca.inv(Mcas(xi)) @ Upcas(xi))

        #Keep a stored solution -> maybe pull from this until the horizon ends to speed up (i.e. run in OL)
        self.uStored = None

    def get_des_state(self):
        """
        Evaluate the desired generalized coordinates
        """
        if self.rel:
            #use "relative" tracking
            xi = self.observer.get_state()

            #get current angles
            theta = xi[2: self.N + 2]

            #assemble desired angles
            thetaD = np.vstack((np.ones((1, 1)), theta[1:, :].reshape((self.N - 1, 1))))
        else:
            #use absolute tracking of pi
            piList = [np.pi*(i + 1) for i in range(self.N)]

            #assmeble desired angles
            thetaD = np.array([piList]).T

        #stack with desired (x, y) position and velocities -> set to zero for now
        return np.vstack((np.zeros((2, 1)), thetaD, np.zeros((self.N + 2, 1))))
    
    def get_config_error(self, theta, thetaD):
        #Returns error in angle using SO(2) formula
        return 1 - np.cos((thetaD - theta))
    
    def cas_2_np(self, x):
        """
        Convert a casadi vector to a numpy array
        """
        #extract the shape of the casadi vector
        shape = x.shape

        #define a numpy vector
        xnp = np.zeros(shape)

        #fill in the entries
        for i in range(shape[0]):
            print(x[i, 0])
            xnp[i, 0] = x[i, 0]

        #return the numpy vector
        return xnp
    
    def np_2_cas(self, x):
        """
        Convert a numpy array to casadi MX
        """
        shape = x.shape
        xMX = ca.MX.zeros(shape[0], shape[1])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                #store xij in a casadi MX matrix
                xMX[i, j] = x[i, j]
        return xMX

    def eval_input(self, t):
        """
        Evaluate the input to a snake
        Inputs:
            t (float): current time in simulation
        """
        #get state vector from observer
        xi = self.observer.get_state()

        #get desired state
        xiD = self.get_des_state()

        #Define MPC optimization
        opti = ca.Opti()
        u = opti.variable(3, self.Nmpc - 1)
        x = opti.variable(2*(self.N + 2), self.Nmpc)

        #define cost function as min norm to Gamma input
        cost = 0
        for k in range(self.Nmpc - 1):
            inputCost = u[:, k].T @ self.R @ u[:, k]
            stateCost = (xiD - x[:, k]).T @ self.Q @ (xiD - x[:, k])
            cost = cost + inputCost + stateCost
        cost = cost + (xiD - x[:, -1]).T @ self.Q @ (xiD - x[:, -1])

        useWarmStart = True
        if useWarmStart:
            xGuess = np.zeros((2*(self.N + 2), self.Nmpc))            
            for i in range(2*(self.N+2)):
                xGuess[i, :] = np.linspace(xi[i, 0], xiD[i, 0], self.Nmpc)
            opti.set_initial(x, xGuess)

        #set initial condition constraint
        opti.subject_to(x[:, 0] == xi)

        #use only twist
        onlyTwist = False
        if onlyTwist:
            opti.subject_to(u[0, :] == 0)
            opti.subject_to(u[1, :] == 0)

        #set dynamics constraints
        for k in range(self.Nmpc - 1):
            #evaluate f, g of the variable. first, insert into a numpy array
            f = self.f(x[:, k])
            g = self.g(x[:, k])
            opti.subject_to(x[:, k+1] == x[:, k] + self.dt * (f + g @ u[:, k]))
        
        #set up optimizatoin
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve()
        
        #store the input
        uMatrix = sol.value(u)
        self._u = uMatrix[:, 0].reshape((3, 1))
        return self._u
    
class SnakeCLFLegacy(cs.Controller):
    """
    Define an MPC snake controller
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a min norm to full state linearization controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #extract snake link number
        self.N = self.observer.dynamics.snake.N

        #define relative or absolute desired
        self.rel = False

        #define CLF constants
        self.gamma = 0.05
        self.gamma0 = 0.01
        self.gamma1 = 2*(self.gamma0**0.5)
        self.Q = np.diag([1, 1, 10])
        self.QLyap = np.eye(self.N + 2)
        
        #Extract matrix functions
        self.Mfunc = self.observer.dynamics.snake.Mfunc
        self.Rfunc = self.observer.dynamics.snake.Rfunc
        self.Upsilonfunc = self.observer.dynamics.snake.Upsilonfunc
        
        #Define f and g as numpy functions
        self.f = lambda xi: np.vstack((xi[self.N + 2:], -np.linalg.inv(self.Mfunc(xi)) @ self.Rfunc(xi)))
        self.g = lambda xi: np.vstack((np.zeros(self.N + 2, 3), np.linalg.inv(self.Mfunc(xi)) @ self.Upsilonfunc(xi)))

    def get_des_state(self):
        """
        Evaluate the desired generalized coordinates
        """
        if self.rel:
            #use "relative" tracking
            xi = self.observer.get_state()

            #get current angles
            theta = xi[2: self.N + 2]

            #assemble desired angles
            thetaD = np.vstack((np.ones((1, 1)), theta[1:, :].reshape((self.N - 1, 1))))
        else:
            #use absolute tracking of pi
            piList = [np.pi*(i + 1) for i in range(self.N)]

            #assmeble desired angles
            thetaD = np.array([piList]).T

        #stack with desired (x, y) position and velocities -> set to zero for now
        return np.vstack((np.zeros((2, 1)), thetaD, np.zeros((self.N + 2, 1))))
    
    def get_config_error(self, xi, xiD):
        """
        Returns errors using geometric formulas
        """
        #get angular error
        theta = xi[2: self.N + 2]
        thetaD = xiD[2: self.N + 2]
        eTheta = 1 - np.cos((theta - thetaD))

        #get velocity error
        thetaDot = xi[2 + self.N + 2:]
        thetaDotD = xiD[2 + self.N + 2:]
        eThetaDot = np.sin(theta - thetaD) * (thetaDot - thetaDotD)

        #assemble eX, eV -> (x, y, theta), (xDot, yDot, thetaDot)
        eX = np.vstack((xi[0:2] - xiD[0:2], eTheta))
        eV = np.vstack((xi[self.N + 2: self.N + 4] - xiD[self.N + 2: self.N + 4], eThetaDot))
        return eX, eV
    
    def vR2(self, xi, xiD, u = None):
        """
        Return CLF value and its first, second derivatives at a particular time
        """
        #extract values of theta
        theta = xi[2: self.N + 2]
        thetaD = xiD[2: self.N + 2]
        thetaDot = xi[self.N + 4:]
        thetaDotD = xi[self.N + 4:]

        #get the config error and velocity error
        eQ, eQDot = self.get_config_error(xi, xiD)

        #compute Lyapunov function
        V = ((eQ).T @ self.QLyap @ (eQ))[0, 0]

        #compute VDot -> this seems to blow up near pi?
        VDot = (2*(eQDot).T @ self.QLyap @ (eQ))[0, 0]

        #compute b for second derivative
        qDDotD = np.zeros((self.N + 2, 1))
        xDDotD, yDDotD, thetaDDotD = qDDotD[0, 0], qDDotD[1, 0], qDDotD[2:]
        b = np.array([[-xDDotD, -yDDotD]]).T
        b = np.vstack((b, np.cos(theta - thetaD) * (thetaDot - thetaDotD)**2 - np.sin(theta - thetaD)*(thetaDDotD)))

        #compute H matrix
        H1 = np.vstack((np.eye(2), np.zeros((self.N, 2))))
        sineMat = np.diag(np.sin(theta - thetaD).reshape((self.N, )))
        H2 = np.vstack((np.zeros((2, self.N)), sineMat))
        H = np.hstack((H1, H2))

        #compute matrices for second derivative of CLF
        s = b - H @ np.linalg.inv(self.Mfunc(xi)) @ self.Rfunc(xi)
        T = H @ np.linalg.inv(self.Mfunc(xi)) @ self.Upsilonfunc(xi)

        #print values of s and T
        # print(s)
        # print(T)
        # print(2 * eQDot.T @ self.QLyap @ eQDot + 2 * eQ.T @ self.QLyap @ s)

        #compute VDDot
        if u is not None:
            VDDot = (2 * eQDot.T @ self.QLyap @ eQDot + 2 * eQ.T @ self.QLyap @ s  + 2 * eQ.T @ self.QLyap @ T @ u)[0, 0]

            #return Lyapunov values
            return V, VDot, VDDot
        else:
            VDDOT_1 = 2 * eQDot.T @ self.QLyap @ eQDot + 2 * eQ.T @ self.QLyap @ s
            VDDOT_2 =  2 * eQ.T @ self.QLyap @ T
            if np.linalg.norm(VDDOT_2) < 10**(-5):
                #have a default value for cutoff
                print("Cutoff")
                VDDOT_2 = 0.05 * np.ones(VDDOT_2.shape)
            return V, VDot, VDDOT_1, VDDOT_2
    
    def vR1(self, xi, xiD, u):
        """
        Rel degree 1 CLF
        """
        #extract values of theta
        theta = xi[2: self.N + 2]
        thetaD = xiD[2: self.N + 2]
        thetaDot = xi[self.N + 4:]
        thetaDotD = xi[self.N + 4:]

        #define pos and vel error
        eX, eV = self.get_config_error(xi, xiD)

        #set desired ddot as zero
        qDDotD = np.zeros((self.N + 2, 1))

        #compute b for second derivative
        qDDotD = np.zeros((self.N + 2, 1))
        xDDotD, yDDotD, thetaDDotD = qDDotD[0, 0], qDDotD[1, 0], qDDotD[2:]
        b = np.array([[-xDDotD, -yDDotD]]).T
        b = np.vstack((b, np.cos(theta - thetaD) * (thetaDot - thetaDotD)**2 - np.sin(theta - thetaD)*(thetaDDotD)))

        #compute H matrix
        H1 = np.vstack((np.eye(2), np.zeros((self.N, 2))))
        sineMat = np.diag(np.sin(theta - thetaD).reshape((self.N, )))
        H2 = np.vstack((np.zeros((2, self.N)), sineMat))
        H = np.hstack((H1, H2))

        #define quadr constants
        alpha = 4
        epsilon = 1

        #define V
        V = 0.5 * eV.T @ eV + 0.5 * alpha * eX.T @ eX + epsilon * eX.T @ eV

        #define VDot
        qDDot = -np.linalg.inv(self.Mfunc(xi)) @ self.Rfunc(xi) + np.linalg.inv(self.Mfunc(xi)) @ self.Upsilonfunc(xi) @ u
        VDot = (H @ qDDot + b).T @ (eV.T + epsilon * eX) + alpha * eV.T @ eX + epsilon * eV.T @ eV 

        #return lyap and deriv
        return V, VDot
    
    def get_R(self, theta):
        """
        Returns rotation matrix in SO(2) for a given angle
        """
        R = np.array([[np.cos(theta), -np.sin(theta), 
                       np.sin(theta), np.cos(theta)]])
        return R

    def eval_input_cas(self, t):
        """
        Evaluate a CLF input to the system with casadi
        """
        #compute current and desired states
        xi = self.observer.get_state()
        xiD = self.get_des_state()

        #define an optimization
        opti = ca.Opti()
        u = opti.variable(3, 1)

        #define cost function
        cost = u.T @ self.Q @ u

        #compute CLF and derivatives
        # V, VDot, VDDot = self.vR2(xi, xiD, np.array([[1, 1, 0]]).T)
        # print([V, VDot, VDDot])
        # V, VDot, VDDot = self.vR2(xi, xiD, u)
        V, VDot = self.vR1(xi, xiD, u)

        #define CLF constraint
        # opti.subject_to(VDDot[0, 0] + self.gamma1 * VDot + self.gamma0 * V <= 0)
        opti.subject_to(VDot + self.gamma * V <= 0)

        #perform minimization
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve()
        
        #return input
        return sol.value(u).reshape((3, 1))
    
    def eval_input_osqp(self, t):
        """
        Evaluate input using OSQP
        """
        #compute current and desired states
        xi = self.observer.get_state()
        xiD = self.get_des_state()

        #evaluate CLF and derivs
        V, VDot, VDDot1, VDDot2 = self.vR2(xi, xiD)

        #evaluate singular values of VDdot2
        U, S, _ = np.linalg.svd(VDDot2)
        print("SVD: ", S)

        #assemble cost matrices
        P = sparse.csc_matrix(self.Q)
        q = np.zeros((3, ))

        #assemble constraint matrices
        A = sparse.csc_matrix(VDDot2)
        u = -VDDot1 - self.gamma1 * VDot - self.gamma0 * V
        
        #create OSQP problem
        prob = osqp.OSQP()

        # Setup workspace and change alpha parameter
        try:
            prob.setup(P = P, q = q, A = A, u = u,  verbose = False, alpha=1.0)
        except:
            print("ill posed")
            return np.zeros((3, 1))

        # Solve problem
        res = prob.solve().x

        print("INPUT: ", res)

        #return optimization output
        return res.reshape((res.size, 1))

    def eval_input(self, t):
        self._u = self.eval_input_cas(t)
        # self._u = self.eval_input_osqp(t)
        return self._u
    
class SnakeCLF(cs.Controller):
    """
    Define an MPC snake controller
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a min norm to full state linearization controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #extract snake link number
        self.N = self.observer.dynamics.snake.N

        #define relative or absolute desired
        self.rel = False

        #define CLF constants
        self.gamma = 0.05
        self.gamma0 = 0.01
        self.gamma1 = 2*(self.gamma0**0.5)
        self.Q = np.diag([1, 1, 10])
        self.QLyap = np.eye(self.N + 2)
        
        #Extract matrix functions
        self.Mfunc = self.observer.dynamics.snake.Mfunc
        self.Rfunc = self.observer.dynamics.snake.Rfunc
        self.Upsilonfunc = self.observer.dynamics.snake.Upsilonfunc
        
        #Define f and g as numpy functions
        self.f = lambda xi: np.vstack((xi[self.N + 2:], -np.linalg.inv(self.Mfunc(xi)) @ self.Rfunc(xi)))
        self.g = lambda xi: np.vstack((np.zeros(self.N + 2, 3), np.linalg.inv(self.Mfunc(xi)) @ self.Upsilonfunc(xi)))

    def calc_q(self, theta):
        """
        Returns a vector in R2 corresponding to an orientation theta
        """
        return np.array([[np.cos(theta), np.sin(theta), 0]]).T
    
    def calc_q_dot(self, theta, thetaDot):
        """
        Return rate of q for theta, thetaDot
        """
        return np.array([-thetaDot * np.sin(theta), thetaDot * np.cos(theta), 0]).T
    
    def calc_q_ddot(self, theta, thetaDot, u):
        """
        Return qDDot for vectors theta, thetaDot, u
        """
    
    def psi(self, theta, thetaD):
        """
        Evaluate the configuration error on S1 as a scalar
        """
        return 1 - np.cos(theta - thetaD)
    
    def psiDot(self, theta, thetaD, thetaDot, thetaD_Dot):
        """
        Evaluate derivative of psi
        """
        return np.sin(theta - thetaD) * (thetaDot - thetaD_Dot)
    
    def eQ(self, xi, xiD):
        """
        Evaluate a vector orientation error
        """
        #extract angles from the system
        theta = xi[2 : self.N + 2]
        thetaD = xiD[2: self.N + 2]

        #calculate the orientation vectors in R2 corresponding to each
        q = self.calc_q(theta[0, 0])
        qD = self.calc_q(thetaD[0, 0])
        eQMatrix = hat(q) @ hat(q) @ qD
        for i in range(1, self.N):
            q = self.calc_q(theta[0, 0])
            qD = self.calc_q(thetaD[0, 0])
            eQMatrix = np.vstack((eQMatrix, hat(q) @ hat(q) @ qD))
        return eQMatrix
    
    def eQDot(self, xi, xiD):
        """
        Evaluate derivative of eQ
        """
        #extract angles from the system
        theta = xi[2 : self.N + 2]
        thetaD = xiD[2: self.N + 2]
        thetaDot = xi[self.N + 4: ]
        thetaDotD = xiD[self.N + 4: ]

        #calculate the orientation vectors in R2 corresponding to each
        q = self.calc_q(theta[0, 0])
        qD = self.calc_q(thetaD[0, 0])
        qDot = self.calc_q_dot(theta[0, 0], thetaDot[0, 0])
        qD_Dot = self.calc_q_dot(thetaD[0, 0], thetaDotD[0, 0])

        #form the derivative matrix
        eQDot = qDot - hat(hat(qD) @ qD_Dot) @ q
        for i in range(1, self.N):
            q = self.calc_q(theta[i, 0])
            qD = self.calc_q(thetaD[i, 0])
            qDot = self.calc_q_dot(theta[i, 0], thetaDot[i, 0])
            qD_Dot = self.calc_q_dot(thetaD[i, 0], thetaDotD[i, 0])
            eQDot = np.vstack((eQDot, qDot - hat(hat(qD) @ qD_Dot) @ q))
        return eQDot

    def v(self, xi, xiD, u):
        """
        Evaluate the Lyapunov function V and its derivatives
        """
        #extract angles from the system
        theta = xi[2 : self.N + 2]
        thetaD = xiD[2: self.N + 2]
        thetaDot = xi[self.N + 4: ]
        thetaDotD = xiD[self.N + 4: ]

        #evaluate Lyapunov function
        alpha = 4
        epsilon = 1
        V = alpha * self.psi(theta, thetaD) + self.eQDot(xi, xiD) @ self.eQDot(xi, xiD) + epsilon * self.eQ(xi, xiD).T @ self.eQDot(xi, xiD)

        #evaluate derivative of Lyapunov function
        

class SnakeUnderact(cs.Controller):
    """
    Define an underactuated snake controller
    Works in N = 2 case.
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for an underactuated controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #extract snake link number
        self.N = self.observer.dynamics.snake.N
        
        #Extract matrix functions
        self.Mfunc = self.observer.dynamics.snake.Mfunc
        self.Rfunc = self.observer.dynamics.snake.Rfunc
        self.Upsilonfunc = self.observer.dynamics.snake.Upsilonfunc

        #Define gains
        self.Kp = 8 * np.eye(self.N)
        self.Kd = 6 * np.eye(self.N)

    def get_des_state(self):
        """
        Evaluate the desired generalized coordinates
        """
        #use absolute tracking of pi
        piList = [np.pi*(i + 1) for i in range(self.N)]

        #assmeble desired angles
        thetaD = np.array([piList]).T

        #stack with desired (x, y) position and velocities -> set to zero for now
        return np.vstack((np.zeros((2, 1)), thetaD, np.zeros((self.N + 2, 1))))
    
    def get_config_error(self, theta, thetaD, thetaDot, thetaDotD):
        """
        Return position and velocity error for angles
        """
        #first, map each theta to between 0 and 2pi radians
        theta = theta - 2*np.pi*np.floor(theta * 1/(2*np.pi))
        thetaD = thetaD - 2*np.pi*np.floor(thetaD * 1/(2*np.pi))

        #compute the error
        eTheta = thetaD - theta

        #compute velocity error
        eThetaDot = thetaDotD - thetaDot

        #return errors
        return eTheta, eThetaDot
    
    def get_theta_ddot_d(self, theta, thetaD, thetaDot, thetaDotD):
        """
        Return the desired thetaDDot
        """
        #compute config errors
        eTheta, eThetaDot = self.get_config_error(theta, thetaD, thetaDot, thetaDotD)

        #compute PD value for thetaDDot
        return self.Kp @ eTheta + self.Kd @ eThetaDot

    def eval_input_N2(self, t):
        """
        Determine the input to the snake using an underactuated controller in the case of N = 2
        This method works for evaluting inputs to control two links simultaneously.
        """
        #get state and desired state
        xi = self.observer.get_state()
        xiD = self.get_des_state()

        #define theta and thetaDot
        theta, thetaD = xi[2:self.N + 2], xiD[2:self.N + 2]
        thetaDot, thetaDotD = xi[self.N + 4: ], xiD[self.N + 4: ]

        #get desired thetaDDot
        thetaDDotD = self.get_theta_ddot_d(theta, thetaD, thetaDot, thetaDotD)

        #Partition M and R into blocks

        #Partition into Chi, theta1
        M = self.Mfunc(xi)
        Mchi, Mchitheta1 = M[0:2, 0:2], M[0:2, 2].reshape((2, 1))
        Mtheta1chi, Mtheta1 = M[2, 0:2], np.array([[M[2, 2]]])

        R = self.Rfunc(xi)
        Rchi = R[0:2]
        Rtheta1 = R[2]

        #Determine Tau based on size of Theta
        if self.N != 1:
            #define the remaining blocks
            MchithetaN = M[0:2, 3:].reshape((2, self.N - 1))
            Mtheta1thetaN = M[2, 3:].reshape((1, self.N - 1))
            MthetaNchi = M[3:, 0:2].reshape((self.N - 1, 2))
            MthetaNtheta1 = M[3: , 2].reshape((self.N - 1, 1))
            MthetaN = M[3:, 3:].reshape((self.N - 1, self.N - 1))
            RthetaN = R[3:].reshape((self.N - 1, 1))

            #determine P1, P2, P3
            P1 = np.hstack((Mchitheta1, MchithetaN))
            P2 = np.hstack((Mtheta1, Mtheta1thetaN))
            P3 = np.hstack((MthetaNtheta1, MthetaN))

            #determine chiDDotD
            chiDDotD = -np.linalg.pinv(MthetaNchi) @ (P3 @ thetaDDotD + RthetaN)
        else:
            P1, P2, P3 = Mchitheta1, Mtheta1, MthetaNtheta1

            #determine chiDDOTD -> this is not well-posed for N  = 1 !
            chiDDotD = -np.linalg.pinv(MthetaNchi) @ (P3 @ thetaDDotD + RthetaN)

        #determine tauX, tautheta1
        tauX = Mchi @ chiDDotD + Rchi + P1 @ thetaDDotD
        tautheta1 = Mtheta1chi @ chiDDotD  + Rtheta1 + P2 @ thetaDDotD
        tauVec = np.vstack((tauX, tautheta1))

        #Now, go from tau to u using Upsilon
        Upsilon = self.Upsilonfunc(xi)
        UpsilonBlock = Upsilon[0:3, :]
        self._u = np.linalg.pinv(UpsilonBlock) @ tauVec

        #return input
        return self._u
    
    def eval_input_j(self, t, j):
        """
        Returns a control input u that allows the system to control angle theta_j
        Inputs:
            t (float): time in simulation
            j (int): index of link we wish to control (zero-indexed). Assumes j =/= 0 (base link)
        """
        #get state and desired state
        xi = self.observer.get_state()
        xiD = self.get_des_state()

        #define theta and thetaDot
        theta, thetaD = xi[2:self.N + 2], xiD[2:self.N + 2]
        thetaDot, thetaDotD = xi[self.N + 4: ], xiD[self.N + 4: ]

        #get desired thetaDDot
        thetaDDotD = self.get_theta_ddot_d(theta, thetaD, thetaDot, thetaDotD)
        thetaDDotDJ = thetaDDotD[j]

        #Partition M and R into blocks

        #Partition into blocks -> NOTE: if we only have two links, M2j-1 disappears!
        M = self.Mfunc(xi)
        Mchi, Mchitheta1, Mchitheta2jm1, Mchithetaj, Mchithetajp1 = M[0:2, 0:2], M[0:2, 2], M[0:2, 2 : j - 1 + 2], M[0:2, j + 2], M[0:2, j+1 + 2:]
        Mtheta1chi, Mtheta1, Mtheta1theta2jm1, Mtheta1thetaj, Mtheta1thetajp1 = M[2, 0:2], M[2, 2], M[2, 2 : j - 1 + 2], M[2, j + 2], M[2, j+1 + 2:]
        Mtheta2jm1chi, Mtheta2jm1theta1, Mtheta2jm1, Mtheta2jm1thetaj, Mtheta2jm1thetajp1 = M[3:j-1 + 2, 0:2], M[3:j-1 + 2, 2], M[3:j-1 + 2, 2 : j - 1 + 2], M[3:j-1 + 2, j + 2], M[3:j-1 + 2, j+1 + 2:]
        Mthetajchi, Mthetajtheta1, Mthetajtheta2jm1, Mtthetajthetaj, Mthetajthetajp1 = M[j + 2, 0:2], M[j + 2, 2], M[j + 2, 2 : j - 1 + 2], M[j + 2, j + 2], M[j + 2, j+1 + 2:]
        Mthetajp1chi, Mthetajp1theta1, Mthetajp1theta2jm1, Mtthetajp1thetaj, Mthetajp1 = M[j + 3:, 0:2], M[j + 3, 2], M[j + 3, 2 : j - 1 + 2], M[j + 3, j + 2], M[j + 3, j+1 + 2:]

        R = self.Rfunc(xi)
        Rchi = R[0:2]
        Rtheta1 = R[2]
        Rtheta2jm1 = R[2 : j-1 + 2]
        Rthetaj = R[2: j + 2]
        Rthetajp1 = R[j + 2 + 1: ]

        #Assemble M and R matrices minus theta1 row
        MmRtheta1 = np.vstack((np.hstack((Mchi, Mchitheta1, Mchitheta2jm1, Mchithetaj, Mchithetajp1)), 
                              np.hstack((Mtheta2jm1chi, Mtheta2jm1theta1, Mtheta2jm1, Mtheta2jm1thetaj, Mtheta2jm1thetajp1)),
                              np.hstack((Mthetajchi, Mthetajtheta1, Mthetajtheta2jm1, Mtthetajthetaj, Mthetajthetajp1)),
                              np.hstack((Mthetajp1chi, Mthetajp1theta1, Mthetajp1theta2jm1, Mtthetajp1thetaj, Mthetajp1))))
        RmRtheta1 = np.vstack((Rchi, Rtheta2jm1, Rthetaj, Rthetajp1))

        #solve for the desired terms
        MZeroMat = MmRtheta1 @ np.diag(np.eye(...))
        thetaDDotExt = np.vstack((np.zeros(...), thetaDDotDJ, np.zeros(...)))
        ddotRthetaJ =  np.linalg.pinv(MZeroMat) @ (-MmRtheta1 @ thetaDDotExt - RmRtheta1)

        #Reassemble the second derivative vector
        ddotVec = ...

        #compute tauTheta1
        MRtheta1 = M[2, :]
        tauTheta1 = MRtheta1 @ ddotVec + R

        #Now, go from tau to u
        self._u = np.vstack((np.zeros((2, 1)), tauTheta1))

        #return input
        return self._u

    def eval_input(self, t):
        """
        Return the control input u
        """
        if self.N == 2:
            return self.eval_input_N2(t)
        else:
            #control theta j
            return self.eval_input_j(t)