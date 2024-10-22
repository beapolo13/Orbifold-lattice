import sympy as sp
from sympy import Mul, Add, factor, expand, collect, sympify
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
a, a_i, b, b_i, c, c_i, d, d_i, e, e_i, f, f_i = sp.symbols('a, a_i, b, b_i, c, c_i, d, d_i,e, e_i, f, f_i', commutative=False)
p_a, p_a_i, p_b, p_b_i, p_c, p_c_i, p_d,p_d_i, p_e,p_e_i, p_f, p_f_i  = sp.symbols('p_a, p_a_i, p_b, p_b_i, p_c, p_c_i, p_d, p_d_i,p_e,p_e_i, p_f, p_f_i', commutative=False)
g_1d,mu = sp.symbols('g_1d,mu', commutative= True)


#and our commutator relations
add_commutator_relation(a,p_a,1j)
add_commutator_relation(a_i,p_a_i,1j)
add_commutator_relation(b,p_b,1j)
add_commutator_relation(b_i,p_b_i,1j)
add_commutator_relation(c,p_c,1j)
add_commutator_relation(c_i,p_c_i,1j)
add_commutator_relation(d,p_d,1j)
add_commutator_relation(d_i,p_d_i,1j)
add_commutator_relation(e,p_e,1j)
add_commutator_relation(e_i,p_e_i,1j)
add_commutator_relation(f,p_f,1j)
add_commutator_relation(f_i,p_f_i,1j)


def H_1(g_1d,mu):  #hamiltonian of the 1st site of the chain
  H_kin=(1/2)*(p_a**2+p_a_i**2+p_b**2+p_b_i**2+p_c**2+p_c_i**2+p_d**2+p_d_i**2)
  #print('H_kin', H_kin)

  H_b=0
  H_b+= (a*b - a_i*b_i - d*c + d_i*c_i)**2     #square of the real part
  H_b+= (a*b_i + a_i*b - d*c_i - d_i*c)**2    #square of the imaginary part
  H_b*=(g_1d**2)
  #print('H_b',H_b)


  delta_H=0
  C=(mu*g_1d)**2/2
  D= 1/(2*(g_1d**2))
  delta_H+= ((a**2 + a_i**2)/2 -D )**2
  delta_H+= ((b**2 + b_i**2)/2 -D )**2
  delta_H+= ((c**2 + c_i**2)/2 -D )**2
  delta_H+= ((d**2 + d_i**2)/2 -D )**2
  delta_H *= C
  #print('delta H', delta_H)

  return H_kin + H_b  + delta_H

def H_k(g_1d,mu):  #hamiltonian of the k-th plaquette of the chain, with k different than 1
  H_kin=(1/2)*(p_a**2+p_a_i**2+p_b**2+p_b_i**2+p_c**2+p_c_i**2)
  #print('H_kin', H_kin)

  H_b=0
  H_b+= (a*b - a_i*b_i - d*c + d_i*c_i)**2     #square of the real part
  H_b+= (a*b_i + a_i*b - d*c_i - d_i*c)**2    #square of the imaginary part
  H_b*=(g_1d**2)
  #print('H_b',H_b)

  

  delta_H=0
  C=(mu*g_1d)**2/2
  D= 1/(2*(g_1d**2))
  delta_H+= (a**2 + a_i**2 -D )**2
  delta_H+= (b**2 + b_i**2 -D )**2
  delta_H+= (c**2 + c_i**2 -D )**2
  delta_H *= C
  #print('delta H', delta_H)

  return H_kin + H_b + delta_H

def H_el_lower(g_1d):
    return (g_1d**2/8) * (a**2 + a_i**2 -f**2 - f_i**2 + d**2 + d_i**2)**2

def H_el_upper(g_1d):
    return (g_1d**2/8) * (c**2 + c_i**2 - e**2 - e_i**2 - d**2 - d_i**2)**2

#gauss law
#operators to the left side of the site
def G_l_down():
    return (1j/2)* (-(b+1j*b_i)*(p_b-1j*p_b_i) + (p_b+1j*p_b_i)*(b-1j*b_i) - (a-1j*a_i)*(p_a+1j*p_a_i) + (p_a-1j*p_a_i)*(a+1j*a_i))
def G_l_up():
    return (1j/2) * (-(c-1j*c_i)*(p_c+1j*p_c_i) + (p_c-1j*p_c_i)*(c+1j*c_i) - (b-1j*b_i)*(p_b+1j*p_b_i) + (p_b-1j*p_b_i)*(b+1j*b_i))

#operators to the right side of the site
def G_r_down():
    return (1j/2)* (-(a+1j*a_i)*(p_a-1j*p_a_i) + (p_a+1j*p_a_i)*(a-1j*a_i) )
def G_r_up():
    return (1j/2)* (-(c+1j*c_i)*(p_c-1j*p_c_i) + (p_c+1j*p_c_i)*(c-1j*c_i))



h1= H_1(g_1d,mu)
h2 =H_k(g_1d,mu)
h_el_low=H_el_lower(g_1d)
h_el_up=H_el_upper(g_1d)
print('')
print('')

lower_h1_comm_gl = str(commutator(h1,G_l_down())).replace('I','1j')
print('lower left site',lower_h1_comm_gl)
print('')
lower_h2_comm_gr = str(commutator(h2,G_r_down())).replace('I','1j')
print('lower right site',lower_h2_comm_gr)
print('')
lower_el = str(commutator(h_el_low,G_l_down()+G_r_down())).replace('I','1j')
print('lower site el',lower_el)
print('')

lower_h1_comm_gl = str(commutator(h1,G_l_down())).replace('I','1j')
print('lower left site',lower_h1_comm_gl)
print('')
lower_h2_comm_gr = str(commutator(h2,G_r_down())).replace('I','1j')
print('lower right site',lower_h2_comm_gr)
print('')
lower_el = str(commutator(h_el_low,G_l_down()+G_r_down())).replace('I','1j')
print('lower site el',lower_el)
print('')

# upper_left_site_comm = str(commutator(h,g01)).replace('I','1j')
# print('upper left site',upper_left_site_comm)
# print('')
# upper_right_site_comm = str(commutator(h,g11)).replace('I','1j')
# print('upper right site',upper_right_site_comm)
# print('')




# H_el=0
#   H_el += (a**2 + a_i**2 + d**2 + d_i**2)**2  #lower left site
#   H_el += (-a**2 - a_i**2 + b**2 + b_i**2)**2  #lower right site
#   H_el += (- c**2 - c_i**2 - b**2 - b_i**2)**2  #upper left site
#   H_el += (c**2 + c_i**2 - d**2 - d_i**2)**2  #upper right site
#   H_el*=g_1d**2/8
#   #print('H_el',H_el)
# H_el=0
#   H_el += (a**2+ a_i**2 + d**2 + d_i**2)**2  #lower left site
#   H_el += (-a**2 - a_i**2 + b**2 + b_i**2)**2  #lower right site
#   H_el += (-c**2 - c_i**2 - b**2 - b_i**2)**2  #upper left site
#   H_el += (c**2 + c_i**2 - d**2 - d_i**2)**2  #upper right site
#   H_el*=g_1d**2/(8)
#   #no está terminada, porque habría que quitar el término correspondiente a los sites de la derecha de la plaqueta anterior y sumar los de la nueva
#   #print('H_el',H_el)