import pickle
import numpy as np
import sympy as sp

with open('M.pickle', 'rb') as inf:    
   M = pickle.loads(inf.read())
print(M)