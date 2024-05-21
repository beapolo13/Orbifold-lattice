import pickle
import os
from utils import *

CACHE_FILE = 'cache.pkl'


def exact_diagonalization(hamiltonian,parameter,cutoff,parameter1=None,parameter2=None):
    gs_vectors=[]
    gs_energies=[]
    for i in range(len(parameter)):
        H= hamiltonian(parameter[i],parameter1,parameter2, cutoff)
        energy, vector = qutip.Qobj.groundstate(H)
        gs_vectors+=[vector]
        gs_energies+=[energy]
    gs=[[gs_vectors], [gs_energies]]
    plt.plot(parameter,gs[1],'r--', label='Energy of groundstate of H')
    plt.title('Hamiltonian diagonalization')
    plt.show()
    # A possible sanity check is to see that H with corrections is positive semidefinite (its lowest eigenvalue is greater than 0)
    beep()
    return gs_energies

def save_cache(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def get_diagonalization(hamiltonian,parameter,cutoff,parameter1=None,parameter2=None):
    cache = load_cache(CACHE_FILE)
    if cache is not None and cache.get(hamiltonian,parameter,cutoff,parameter1=None,parameter2=None) is not None:
        return cache[hamiltonian,parameter,cutoff,parameter1,parameter2]
    
    result = exact_diagonalization(hamiltonian,parameter,cutoff,parameter1=None,parameter2=None)
    if cache is None:
        cache = {}
    cache[hamiltonian,parameter,cutoff,parameter1,parameter2] = result
    save_cache(cache, CACHE_FILE)
    return result

def other_function(hamiltonian,parameter,cutoff,parameter1=None,parameter2=None): #esta es el formato de como tengo que poner todas ahora
    result = get_diagonalization(hamiltonian,parameter,cutoff,parameter1=None,parameter2=None)
    # Use the result in some way
    return ...

# Example usage
result = other_function(hamiltonian,parameter,cutoff,parameter1=None,parameter2=None)
