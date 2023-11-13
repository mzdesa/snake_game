import sympy as sp
import numpy as np
import casadi as ca
from sympy.core.function import diff
from sympy.simplify.simplify import simplify, expand

class Snake:
  """
  Class for generating snakes and snake functions
  """
  def __init__(self, N, mVal = 1, lVal = 1, JVal = 1):
    """
    Init function for snake attributes
    N: number of snake disks
    mVal: mass of each disk (kg)
    lVal: diameter of each disk (m)
    JVal: rotational inertia of disk
    """
    #define a NAME ERROR FLAG -> Defaults to False
    self.nameErr = False

    #constants
    l = sp.Symbol("l", real = True)
    m = sp.Symbol("m", real = True)
    J = sp.Symbol("J", real = True)

    #define generalized coordinates as functions of time
    t = sp.Symbol("t", real = True)
    x = sp.Function("x", real = True)(t)
    y = sp.Function("y", real = True)(t)

    #define the Chi vector
    Chi = sp.Matrix.vstack(sp.Matrix([x]), sp.Matrix([y]))

    #define generalized force F
    F = sp.Symbol("F", real = True)

    #create set of generalized coordinates
    genCoordList = []
    for i in range(N):
      name = "theta" + str(i+1)
      genCoordList.append(sp.Function(name, real = True))

    #create the set of functions and put them in a matrix
    theta = sp.Matrix([genCoordList[0](t)]) #initialize matrix
    for i in range(1, N):
      #append to end of theta matrix to get a function theta
      theta = sp.Matrix.vstack(theta, sp.Matrix([genCoordList[i](t)]))

    #define the derivative coordinates
    thetaDot = diff(theta, t)

    #assemble a matrix of coefficients of the second derivative terms in each equation
    q = sp.Matrix.vstack(Chi, theta)
    qDot = diff(q, t, 1)
    qDDot = diff(q, t, 2)

    #define helper functions of rcomputing ei and ej
    def eTheta(i):
      return sp.Matrix([[-sp.sin(theta[i])], [sp.cos(theta[i])]])

    #compute velocities and kinetic energy

    #base case:
    v1Bar = diff(Chi) + l/2* thetaDot[0] * eTheta(0)

    #create a list of velocities
    vList = [v1Bar]

    #now, compute the remaining velocities recursively
    for n in range(1, N):
      vnBar = vList[n-1] + l/2*(thetaDot[n-1]*eTheta(n-1) + thetaDot[n]*eTheta(n))
      vList.append(simplify(vnBar))

    #compute the kinetic energy
    T = sp.Matrix.zeros(1)
    for n in range(N):
      #compute the KE of link n
      vnBar = vList[n]
      Tn = 1/2*m * vnBar.T @ vnBar + 1/2*J*thetaDot[n]**2*sp.Matrix.ones(1)
      T = T + Tn

    #unpack the kinetic energy from the vector
    T = T[0]

    #compute equations of motion

    #first, solve for chi equations of motion
    dTdChiDot = diff(T, diff(Chi, t))
    eqChi = diff(dTdChiDot, t) - diff(T, Chi)

    #solve for the theta equations of motion
    dTdThetaDot = diff(T, diff(theta, t))
    eqTheta = diff(dTdThetaDot, t) - diff(T, theta)

    #simplify the equations of motion
    simpl = True
    if simpl:
      eqChi = simplify(eqChi)
      eqTheta = simplify(eqTheta)

    #Set up equations of motion
    eqMotion = sp.Matrix.vstack(eqChi, eqTheta)
    eqMotion = expand(eqMotion) #make sure to expand to get the correct coefficients
    eqMotion

    #loop over the coordinates and equations to get the double dot coefficient matrix M
    M = sp.Matrix.zeros(N + 2, N + 2) #square matrix matching size of # of GC

    for i in range(N + 2):
      #loop over the equation:
      for j in range(N + 2):
        #loop over the variable
        M[i, j] = eqMotion[i].coeff(qDDot[j])

    #Get the residual (non second derivative) terms
    R = eqMotion - M @ qDDot

    #Solve for the term "Upsilon" that maps from 2D force F to generalized force Gamma
    Upsilon = sp.Matrix.vstack(sp.Matrix.eye(2), l/2*eTheta(0).T, sp.Matrix.zeros(N-1, 2))
    Upsilon = sp.Matrix.hstack(Upsilon, sp.Matrix.zeros(N + 2, 1))
    Upsilon[2, 2] = 1 #add the torque term in!    

    #substitute length, mass, inertia with numerical values
    M = M.subs([(m, mVal), (l, lVal), (J, JVal)])
    R = R.subs([(m, mVal), (l, lVal), (J, JVal)])
    Upsilon = Upsilon.subs([(m, mVal), (l, lVal), (J, JVal)])

    #define state vector xi
    xi = sp.Matrix.vstack(q, qDot)

    #Lambdify the formulas for M, R, and Upsilon
    Mfunc = sp.lambdify(xi, M, "numpy") #add modules = sp to avoid "derivative" error -> work on this!
    Rfunc = sp.lambdify(xi, R, "numpy")
    Upsilonfunc = sp.lambdify(xi, Upsilon, "numpy")

    #Turn each lambda function from a function of 2(N+2) vars into a function of a (2(N+2), 1) vector using HOF
    def cur_func(multiFunc):
      """
      Curries a multi variable function into a single variable function
      """
      def vecFunc(vec):
        """
        Vector function -> takes in a vector and returns multiFunc of the vector
        """
        #first, convert the vector to a list
        listVec = vec.reshape((vec.shape[0], )).tolist()

        #compute and return the multifunc value
        try:
          return multiFunc(*listVec)
        except NameError:
          if self.nameErr == False:
            print("Numerical Instability in R Matrix")
            print(R)
          self.nameErr = True
          return 0
      #return the output of vecFunc
      return vecFunc

    #Turn each function into a function of a numpy vector, save functions as class attributes
    self.Mfunc = cur_func(Mfunc)
    self.Rfunc = cur_func(Rfunc)
    self.Upsilonfunc = cur_func(Upsilonfunc)

    #save N as an attribute
    self.N = N

    #save numerical parameters
    self.m = mVal
    self.l = lVal
    self.J = JVal

class SnakeV2:
  """
  Class for generating snakes and snake functions
  SnakeV2 uses an additional substitution (with the replace method)
  that allows for more complex snakes (V1 would get numerical errors here)
  """
  def __init__(self, N, mVal = 1, lVal = 1, JVal = 1):
    """
    Init function for snake attributes
    N: number of snake disks
    mVal: mass of each disk (kg)
    lVal: diameter of each disk (m)
    JVal: rotational inertia of disk
    """
    #define a NAME ERROR FLAG -> Defaults to False
    self.nameErr = False

    #constants
    l = sp.Symbol("l", real = True)
    m = sp.Symbol("m", real = True)
    J = sp.Symbol("J", real = True)

    #define generalized coordinates as functions of time
    t = sp.Symbol("t", real = True)
    x = sp.Function("x", real = True)(t)
    y = sp.Function("y", real = True)(t)

    #define the Chi vector
    Chi = sp.Matrix.vstack(sp.Matrix([x]), sp.Matrix([y]))

    #define generalized force F
    F = sp.Symbol("F", real = True)

    #create set of generalized coordinates
    genCoordList = []
    for i in range(N):
      name = "theta" + str(i+1)
      genCoordList.append(sp.Function(name, real = True))

    #create the set of functions and put them in a matrix
    theta = sp.Matrix([genCoordList[0](t)]) #initialize matrix
    for i in range(1, N):
      #append to end of theta matrix to get a function theta
      theta = sp.Matrix.vstack(theta, sp.Matrix([genCoordList[i](t)]))

    #define the derivative coordinates
    thetaDot = diff(theta, t)

    #assemble a matrix of coefficients of the second derivative terms in each equation
    q = sp.Matrix.vstack(Chi, theta)
    qDot = diff(q, t, 1)
    qDDot = diff(q, t, 2)

    #define helper functions of rcomputing ei and ej
    def eTheta(i):
      return sp.Matrix([[-sp.sin(theta[i])], [sp.cos(theta[i])]])

    #compute velocities and kinetic energy

    #base case:
    v1Bar = diff(Chi) + l/2* thetaDot[0] * eTheta(0)

    #create a list of velocities
    vList = [v1Bar]

    #now, compute the remaining velocities recursively
    for n in range(1, N):
      vnBar = vList[n-1] + l/2*(thetaDot[n-1]*eTheta(n-1) + thetaDot[n]*eTheta(n))
      vList.append(simplify(vnBar))

    #compute the kinetic energy
    T = sp.Matrix.zeros(1)
    for n in range(N):
      #compute the KE of link n
      vnBar = vList[n]
      Tn = 1/2*m * vnBar.T @ vnBar + 1/2*J*thetaDot[n]**2*sp.Matrix.ones(1)
      T = T + Tn

    #unpack the kinetic energy from the vector
    T = T[0]

    #compute equations of motion

    #first, solve for chi equations of motion
    dTdChiDot = diff(T, diff(Chi, t))
    eqChi = diff(dTdChiDot, t) - diff(T, Chi)

    #solve for the theta equations of motion
    dTdThetaDot = diff(T, diff(theta, t))
    eqTheta = diff(dTdThetaDot, t) - diff(T, theta)

    #simplify the equations of motion
    simpl = False
    if simpl:
      eqChi = simplify(eqChi)
      eqTheta = simplify(eqTheta)

    #Set up equations of motion
    eqMotion = sp.Matrix.vstack(eqChi, eqTheta)
    eqMotion = expand(eqMotion) #make sure to expand to get the correct coefficients
    eqMotion

    #loop over the coordinates and equations to get the double dot coefficient matrix M
    M = sp.Matrix.zeros(N + 2, N + 2) #square matrix matching size of # of GC

    for i in range(N + 2):
      #loop over the equation:
      for j in range(N + 2):
        #loop over the variable
        M[i, j] = eqMotion[i].coeff(qDDot[j])

    #Get the residual (non second derivative) terms
    R = eqMotion - M @ qDDot

    #Solve for the term "Upsilon" that maps from 2D force F to generalized force Gamma
    Upsilon = sp.Matrix.vstack(sp.Matrix.eye(2), l/2*eTheta(0).T, sp.Matrix.zeros(N-1, 2))
    Upsilon = sp.Matrix.hstack(Upsilon, sp.Matrix.zeros(N + 2, 1))
    Upsilon[2, 2] = 1 #add the torque term in!    

    #define state vector xi
    xi = sp.Matrix.vstack(q, qDot)

    #substitute length, mass, inertia with numerical values
    M = M.subs([(m, mVal), (l, lVal), (J, JVal)])
    R = R.subs([(m, mVal), (l, lVal), (J, JVal)])
    Upsilon = Upsilon.subs([(m, mVal), (l, lVal), (J, JVal)])

    #generate a list of variables for replacements
    replList = [sp.Symbol('xi' + str(i)) for i in range(2*(N + 2))]

    #Replace time derivatives and functions with Symbolic variables
    replacementsNoDiff = [(xi[i], replList[i]) for i in range(N + 2)]

    #perform substitutions
    M = M.subs(replacementsNoDiff)
    R = R.subs(replacementsNoDiff)
    Upsilon = Upsilon.subs(replacementsNoDiff)

    #Replace derivatives
    replacementsDiff = [(xi[i].diff(t), replList[i + N + 2]) for i in range(N+2)]

    #use "replace" to replace derivative terms
    for i in range(N+2):
      M = M.replace(xi[i].diff(t), replacementsDiff[i]).doit()
      R = R.replace(xi[i].diff(t), replacementsDiff[i]).doit()
      Upsilon = Upsilon.replace(xi[i].diff(t), replacementsDiff[i]).doit()
      # DF = DF.replace(xi[i].diff(t), replacementsDiff[i]).doit()

    #Lambdify the formulas for M, R, and Upsilon
    Mfunc = sp.lambdify(replList, M, "numpy") #add modules = sp to avoid "derivative" error -> work on this!
    Rfunc = sp.lambdify(replList, R, "numpy")
    Upsilonfunc = sp.lambdify(replList, Upsilon, "numpy")

    #Turn each lambda function from a function of 2(N+2) vars into a function of a (2(N+2), 1) vector using HOF
    def cur_func(multiFunc):
      """
      Curries a multi variable function into a single variable function
      """
      def vecFunc(vec):
        """
        Vector function -> takes in a vector and returns multiFunc of the vector
        """
        #first, convert the vector to a list if it is a numpy array
        if isinstance(vec, np.ndarray):
          listVec = vec.reshape((vec.shape[0], )).tolist()
        else:
          #assume that it is a casadi array
          listVec = [vec[i, 0] for i in range(vec.shape[0])]

        #compute and return the multifunc value
        try:
          return multiFunc(*listVec)
        except NameError:
          if self.nameErr == False:
            print("Numerical Instability in R Matrix")
            print(R)
          self.nameErr = True
          return 0
      #return the output of vecFunc
      return vecFunc

    #Turn each function into a function of a numpy vector, save functions as class attributes
    self.Mfunc = cur_func(Mfunc)
    self.Rfunc = cur_func(Rfunc)
    self.Upsilonfunc = cur_func(Upsilonfunc)

    #save N as an attribute
    self.N = N

    #save numerical parameters
    self.m = mVal
    self.l = lVal
    self.J = JVal

    #Linearize system (if option is turned on)
    lin = False
    if lin:
      #define a dummy input variable u -> we will replace this with zero shortly
      u = sp.MatrixSymbol("u", 3, 1)

      # Now, define g and f for control affine dynamics
      f = sp.Matrix.vstack(qDot, -M.inv() @ R)
      g = sp.Matrix.vstack(sp.Matrix.zeros(N + 2, 3), M.inv() @ Upsilon)
      #define the dynamics
      dyn = f + g @ u

      #assembly derivative matrices -> compute gradient of entire system dynamics
      DF = sp.Matrix.zeros(2*(N+2), 2*(N+2))
      for i in range(2*(N+2)):
          for j in range(2*(N+2)):
            #take the deriv. of dynamics with respect to xi_j -> forms the first column
            print(diff(dyn, xi[j]))
            DF[i, j] = diff(dyn, xi[j])

      #Now, substitute zero into the input for linearization about zero input
      DF = DF.subs([(u[0], 0), (u[1], 0), (u[2], 0)])
      DF = DF.subs(replacementsNoDiff)
      for i in range(N+2):
        DF = DF.replace(xi[i].diff(t), replacementsDiff[i]).doit()
      print("HI")
      print(DF)
      #lambdify the function
      DFfunc = sp.lambdify(replList, DF, "numpy")
      self.DFfunc = cur_func(DFfunc)

if __name__ == "__main__":
  N = 3
  m = 1
  l = 0.1
  J = m*(l/2)**2
  snake = SnakeV2(N, mVal = m, lVal = l, JVal= J)