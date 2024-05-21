from utils import *


import numpy as np
import sympy as sp

# Define ladder operator symbols
a = sp.symbols('a', commutative=False)
a_dag = sp.symbols('a^\\dagger', commutative=False)

#define labels
labels = {'alpha 1': 1, 'alpha 2': 2, 'beta 1': 3, 'beta 2': 4,
              'gamma 1': 5, 'gamma 2': 6, 'delta 1': 7, 'delta 2': 8}
inverse_labels = {value: key for key, value in labels.items()}

def ladder_sym(creation, index, cutoff):
    if creation:
        return sp.Symbol(f'a^\\dagger_{inverse_labels[index]}')
    else:
        return sp.Symbol(f'a_{inverse_labels[index]}')

def map_sym(type, label, cutoff):
    labels = {'alpha_1': 1, 'alpha_2': 2, 'beta_1': 3, 'beta_2': 4,
              'gamma_1': 5, 'gamma_2': 6, 'delta_1': 7, 'delta_2': 8}
    inverse_labels = {value: key for key, value in labels.items()}
    if type == 'x':
        return (1/sp.sqrt(2)) * (ladder_sym(True, labels[label], cutoff) + ladder_sym(False, labels[label], cutoff))
    elif type == 'p':
        return (1j/sp.sqrt(2)) * (ladder_sym(False, labels[label], cutoff) - ladder_sym(True, labels[label], cutoff))

def to_latex(type, label, cutoff):
    expr = map_sym(type, label, cutoff)
    return sp.latex(sp.simplify(expr))

# Example usage
print(to_latex('x', 'alpha_1', 3))
print(to_latex('p', 'beta_2', 3))
