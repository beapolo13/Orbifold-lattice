from utils import *
import numpy as np
import sympy as sp
from sympy import simplify, expand, factor

# Define ladder operator symbols
a = sp.symbols('a', commutative=False)
a_dag = sp.symbols('a^\\dagger', commutative=False)

#define labels
labels = {'alpha 1': 1, 'alpha 2': 2, 'beta 1': 3, 'beta 2': 4,
              'gamma 1': 5, 'gamma 2': 6, 'delta 1': 7, 'delta 2': 8}
inverse_labels = {value: key for key, value in labels.items()}

def ladder_sym(creation, index):
    if creation:
        return sp.Symbol(f'a^\\dagger_{inverse_labels[index]}')
    else:
        return sp.Symbol(f'a_{inverse_labels[index]}')

def map_sym(type, label):
    labels = {'alpha_1': 1, 'alpha_2': 2, 'beta_1': 3, 'beta_2': 4,
              'gamma_1': 5, 'gamma_2': 6, 'delta_1': 7, 'delta_2': 8}
    inverse_labels = {value: key for key, value in labels.items()}
    if type == 'x':
        return (1/sp.sqrt(2)) * (ladder_sym(True, labels[label]) + ladder_sym(False, labels[label]))
    elif type == 'p':
        return (1j/sp.sqrt(2)) * (ladder_sym(False, labels[label]) - ladder_sym(True, labels[label]))
    


def to_latex(expr):
    #expr = map_sym(type, label)
    return sp.latex(sp.simplify(expr))

def H_plaquette_sym(g_1d, a, mu):
    labels = {'alpha_1': 1, 'alpha_2': 2, 'beta_1': 3, 'beta_2': 4, 'gamma_1': 5, 'gamma_2': 6, 'delta_1': 7, 'delta_2': 8}
    inverse_labels = {value: key for key, value in labels.items()}  # this will be useful for the definition of H_el and delta_H
    g_1d, a, mu = sp.symbols('g_1d a mu')

    H_kin = 0
    for label in labels:
        H_kin += (1/2) * (map_sym('p', label)**2)
    #print('H_kin', to_latex(H_kin))
    
    H_b = 0
    H_b += (map_sym('x', 'beta_1') * map_sym('x', 'alpha_1') - map_sym('x', 'beta_2') * map_sym('x', 'alpha_2') + map_sym('x', 'gamma_1') * map_sym('x', 'delta_1') - map_sym('x', 'gamma_2') * map_sym('x', 'delta_2'))**2  # first term
    H_b += (map_sym('x', 'beta_1') * map_sym('x', 'alpha_2') + map_sym('x', 'beta_2') * map_sym('x', 'alpha_1') + map_sym('x', 'gamma_1') * map_sym('x', 'delta_2') + map_sym('x', 'gamma_2') * map_sym('x', 'delta_1'))**2  # second term
    H_b *= (g_1d**2)
    #print('H_b', to_latex(H_b))
    
    H_el = 0
    for m in range(1, 5):
        H_el += (map_sym('x', inverse_labels[2*m-1])**2 + map_sym('x', inverse_labels[2*m])**2)**2  # first term
    H_el += (map_sym('x', inverse_labels[1])**2 + map_sym('x', inverse_labels[2])**2) * (map_sym('x', inverse_labels[7])**2 + map_sym('x', inverse_labels[8])**2)
    H_el -= (map_sym('x', inverse_labels[1])**2 + map_sym('x', inverse_labels[2])**2) * (map_sym('x', inverse_labels[3])**2 + map_sym('x', inverse_labels[4])**2)
    H_el -= (map_sym('x', inverse_labels[5])**2 + map_sym('x', inverse_labels[6])**2) * (map_sym('x', inverse_labels[7])**2 + map_sym('x', inverse_labels[8])**2)
    H_el += (map_sym('x', inverse_labels[5])**2 + map_sym('x', inverse_labels[6])**2) * (map_sym('x', inverse_labels[3])**2 + map_sym('x', inverse_labels[4])**2)
    H_el *= g_1d**2 / (4 * a)
    #print('H_el', to_latex(H_el))
    
    delta_H = 0
    # C = (mu * a * g_1d)**2 / 8
    # D = 2 / (a * (g_1d**2))
    for m in range(1, 5):
        delta_H += ((mu * a * g_1d)**2 / 8) * (map_sym('x', inverse_labels[2*m-1])**2 + map_sym('x', inverse_labels[2*m])**2)**2
        delta_H -= (mu**2 * a / 4) * (map_sym('x', inverse_labels[2*m-1])**2 + map_sym('x', inverse_labels[2*m])**2)
        delta_H += (mu * a * g_1d)**2 / (2 * (2 * a * g_1d**2)**2)  # these are the corrections, but they depend on g_1d inversely-quadratic
    #print('delta H', to_latex(delta_H))
    
    return H_kin + H_b + H_el + delta_H

def plaquette_operator_sym(g,a,N): #N is the number of plaquettes, one in our case
    def P_operator():
        x_alpha_dag=(1/np.sqrt(2))*(map_sym('x','alpha_1')- 1j*map_sym('x','alpha_2'))
        x_beta_dag=(1/np.sqrt(2))*(map_sym('x','beta_1')- 1j*map_sym('x','beta_2'))
        x_delta=(1/np.sqrt(2))*(map_sym('x','delta_1')+ 1j*map_sym('x','delta_2'))
        x_gamma=(1/np.sqrt(2))*(map_sym('x','gamma_1')+ 1j*map_sym('x','gamma_2'))
        return x_alpha_dag*x_beta_dag*x_delta*x_gamma
    def P_dag_operator():
        x_alpha=(1/np.sqrt(2))*(map_sym('x','alpha_1')+ 1j*map_sym('x','alpha_2'))
        x_beta=(1/np.sqrt(2))*(map_sym('x','beta_1')+ 1j*map_sym('x','beta_2'))
        x_delta_dag=(1/np.sqrt(2))*(map_sym('x','delta_1')- 1j*map_sym('x','delta_2'))
        x_gamma_dag=(1/np.sqrt(2))*(map_sym('x','gamma_1')- 1j*map_sym('x','gamma_2'))
        return x_gamma_dag*x_delta_dag*x_beta*x_alpha
    return (1/(2*N)) *(P_operator()+P_dag_operator())

g_1d, a, mu = sp.symbols('g_1d a mu')
print(to_latex(expand(simplify(plaquette_operator_sym(g_1d, a, 1)))))


# Example usage
#print(to_latex('x', 'alpha_1'))
#print(to_latex('p', 'beta_2'))
