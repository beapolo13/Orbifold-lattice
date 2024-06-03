from utils import *

n_modes=8
cutoff=3

#definition of specific ladder operators: the labels that have a 1 refer to the real part and those with a 2 refer to the imaginary part
ladder_ops={'a_alpha_1':ladder(True,1,cutoff),'a_alpha_2':ladder(True,2,cutoff),'a_beta_1':ladder(True,3,cutoff),'a_beta_2':ladder(True,4,cutoff),
     'a_gamma_1':ladder(True,5,cutoff),'a_gamma_2':ladder(True,6,cutoff),'a_delta_1':ladder(True,7,cutoff),'a_delta_2':ladder(True,8,cutoff),
     'adag_alpha_1':ladder(False,1,cutoff),'adag_alpha_2':ladder(False,2,cutoff),'adag_beta_1':ladder(False,3,cutoff),'adag_beta_2':ladder(False,4,cutoff),
     'adag_gamma_1':ladder(False,5,cutoff),'adag_gamma_2':ladder(False,6,cutoff),'adag_delta_1':ladder(False,7,cutoff),'adag_delta_2':ladder(False,8,cutoff)}

#define mapping from x,p to ladder operators

labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}

inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H

#parameters
a=1
mu=1
g_vec=np.arange(0.1,10,0.1) #free parameter
 


#exact_diagonalization_and_save('diagonalisation.pkl',H_plaquette,1/g_vec**2,a,mu)

#once diagonalisation has been performed, we extract the energy and eigenvectors arrays:
with open('diagonalisation.pkl', 'rb') as file:
     result = pickle.load(file)
     result_energies=result[-1]
     result_vectors=result[-2]

plt.plot(1/g_vec**2,result_energies,'r--', label='Energy of groundstate of H')
plt.xscale('log')
plt.title('Hamiltonian diagonalization')
plt.savefig('hamiltonian diagonalisation')
plt.show()


expectation_value_on_gs('plaquette',plaquette_operator, H_plaquette, result_vectors, 1/g_vec**2, parameter1=1,parameter2=1)
expectation_value_on_gs('magnetic field',H_b, H_plaquette, result_vectors, 1/g_vec**2, parameter1=1,parameter2=1)