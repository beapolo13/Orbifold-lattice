import sympy as sp
from sympy import Mul, Add
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



#First we calculate the commutator of 

# Dictionary to store known commutation relations
known_commutators = {}

# Define a custom commutator function with anti-symmetry, product rule, and predefined relations
def commutator(A, B):
    # Check if A == B: [A, A] = 0
    if A == B:
       
        return sp.S(0)
    
    # Check for predefined non-zero commutators [A, B]
    if (A, B) in known_commutators:
        return known_commutators[(A, B)]
    
    # If the exact commutator of A and B is not predefined, check if A and B are expressions
    if A.is_Add or B.is_Add:
    
        # Apply distributive property: [A, B + C] = [A, B] + [A, C]
        return sp.Add(*[commutator(A, term) for term in B.args]) if B.is_Add else sp.Add(*[commutator(term, B) for term in A.args])

    if A.is_Mul or B.is_Mul:
        
        # Apply product rule: [A, BC] = [A, B]C + B[A, C]
        factors_A = A.as_ordered_factors() if A.is_Mul else [A]
        factors_B = B.as_ordered_factors() if B.is_Mul else [B]
        #print(factors_A,factors_B)
        comm_result = 0
        for i, factor_A in enumerate(factors_A):
            for j, factor_B in enumerate(factors_B):
                left_A = sp.Mul(*factors_A[:i]) if i > 0 else 1
                right_A = sp.Mul(*factors_A[i+1:]) if i+1 < len(factors_A) else 1
                
                left_B = sp.Mul(*factors_B[:j]) if j > 0 else 1
                right_B = sp.Mul(*factors_B[j+1:]) if j+1 < len(factors_B) else 1
                
                comm_result += left_A * left_B * commutator(factor_A, factor_B) * right_A * right_B
        return comm_result
    if A.is_Pow:
        base_A, exp_A = A.args
        if exp_A.is_Number:  # Ensure exponent is a number (e.g., integer)
            return exp_A * base_A**(exp_A - 1) * commutator(base_A, B)
        
    if B.is_Pow:
        base, exp = B.args
        if exp.is_Number:  # Handle only integer powers
            return exp * base**(exp - 1) * commutator(A, base)
    
    # If no predefined commutator exists, return 0 since A and B commute
    return sp.S(0)

# Define a function to add known commutation relations
def add_commutator_relation(A, B, result):
    known_commutators[(A, B)] = result
    known_commutators[(B, A)] = -result  # Automatically add [B, A] = -result


#Define all of our variables
lu, lu_i, u, u_i, ru, ru_i, u2, u2_i, ld, ld_i, rd, rd_i = sp.symbols('lu, lu_i, u, u_i, ru, ru_i, u2, u2_i, ld, ld_i, rd, rd_i', commutative=False)
p_lu, p_lu_i, p_u, p_u_i, p_ru, p_ru_i, p_u2, p_u2_i, p_ld, p_ld_i, p_rd, p_rd_i = sp.symbols('p_lu, p_lu_i, p_u, p_u_i, p_ru, p_ru_i, p_u2, p_u2_i, p_ld, p_ld_i, p_rd, p_rd_i ', commutative=False)
g_1d,a,mu = sp.symbols('g_1d,a,mu', commutative= True)


#and our commutator relations
add_commutator_relation(lu,p_lu,1j)
add_commutator_relation(lu_i,p_lu_i,1j)
add_commutator_relation(u,p_u,1j)
add_commutator_relation(u_i,p_u_i,1j)
add_commutator_relation(ru,p_ru,1j)
add_commutator_relation(ru_i,p_ru_i,1j)
add_commutator_relation(u2,p_u2,1j)
add_commutator_relation(u2_i,p_u2_i,1j)
add_commutator_relation(ld,p_ld,1j)
add_commutator_relation(ld_i,p_ld_i,1j)
add_commutator_relation(rd,p_rd,1j)
add_commutator_relation(rd_i,p_rd_i,1j)

def H_k(g_1d,a,mu):  #hamiltonian of the k-th site of the chain
  H_kin=(1/2)*(p_u**2+p_u_i**2+p_ru**2+p_ru_i**2+p_rd**2+p_rd_i**2)
  #print('H_kin', H_kin)

  H_b=0
  H_b+= (rd*u2 - rd_i*u2_i - u*ru + u_i*ru_i)**2     #square of the real part
  H_b+= (rd*u2_i + rd_i*u2 - u*ru_i - u_i*ru)**2    #square of the imaginary part
  H_b*=(g_1d**2)
  #print('H_b',H_b)

  H_el=0
  H_el+= (ru**2 + ru_i**2 - lu**2 - lu_i**2 - u**2 - u_i**2)**2
  H_el+= (rd**2 + rd_i**2 - ld**2 - ld_i**2 + u**2 + u_i**2)**2
  H_el*=g_1d**2/(4*a)
  #print('H_el',H_el)

  delta_H=0
  C=(mu*a*g_1d)**2/2
  D= 1/(2*a**2*(g_1d**2))
  delta_H+= (rd**2 + rd_i**2 -D )**2
  delta_H+= (ru**2 + ru_i**2 -D )**2
  delta_H+= (u**2 + u_i**2 -D )**2
  delta_H *= C
  #print('delta H', delta_H)

  return H_kin + H_b + H_el + delta_H

def G_kd():
    return 6 - 2* (rd * p_rd_i - rd_i * p_rd + ld * p_ld_i - ld_i * p_ld + u * p_u_i - u_i * p_u)

def G_ku():
    return 6 - 2* (ru * p_ru_i - ru_i * p_ru + lu * p_lu_i - lu_i * p_lu + u * p_u_i - u_i * p_u)

h= H_k(1,1,1)
g = G_kd()
g2= G_ku()

print(h)
print('')
print('')

print('first real',sp.simplify(sp.expand(commutator(h,g))))

print('second real', sp.simplify(sp.expand(commutator(h,g2))))

print(commutator(h,g) == commutator(h,g2))