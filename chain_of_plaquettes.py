import sympy as sp
import numpy as np
from numpy import transpose, real, sqrt, sin, cos, linalg, cosh, sinh
import scipy
from scipy.sparse import lil_matrix, csc_matrix
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize
import time
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from pprint import pprint
from scipy.linalg import block_diag
import qutip
from qutip import tensor, destroy, create, identity, entropy_mutual
from qutip import *
#from qutip.tensor import tensor
import winsound
import pickle
import matplotlib.colors as mcolors


#First we define a commutator that obeys commutation rules


# Dictionary to store known commutation relations
known_commutators = {(x)}

# Define a custom commutator function with anti-symmetry, product rule, and predefined relations
def commutator(A, B):
    # Check if A == B: [A, A] = 0
    if A == B:
        return sp.S(0)
    
    # Check for predefined commutators [A, B]
    if (A, B) in known_commutators:
        return known_commutators[(A, B)]
    
    # Check for the anti-symmetric relation: [A, B] = -[B, A]
    if A.compare(B) > 0:
        return -commutator(B, A)
    
    # If no predefined commutator, handle sums and products
    if A.is_Atom and B.is_Atom:
        return A * B - B * A
    
    # Product rule: [A, BC] = [A, B]C + B[A, C]
    elif B.is_Mul:
        factors = B.as_ordered_factors()
        comm = 0
        for i, factor in enumerate(factors):
            left = sp.Mul(*factors[:i])
            right = sp.Mul(*factors[i+1:])
            comm += left * commutator(A, factor) * right
        return comm
    
    # Sum rule: [A, B + C] = [A, B] + [A, C]
    elif B.is_Add:
        return sp.Add(*[commutator(A, term) for term in B.args])
    
    # Default commutator if none of the above apply
    return A * B - B * A

# Define a function to add known commutation relations
def add_commutator_relation(A, B, result):
    known_commutators[(A, B)] = result
    known_commutators[(B, A)] = -result  # Automatically add [B, A] = -result

x_k_d, x_k_u, y_k_d, px_k_d, px_k_u, py_k_d = sp.symbols('x_k_d, x_k_u, y_k_d, px_k_d, px_k_u, py_k_d', commutative=False)

x_kl_d, x_kl_u, y_kr_d, px_kl_d, px_kl_u = sp.symbols('x_kl_d, x_kl_u, y_kr_d, px_kl_d, px_kl_u', commutative=False)

print(commutator(x_k_d,x_k_u))

