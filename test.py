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
a=1  #a=1 by default, a=0.0001 for the continuum limit
if a==1:
     filename='diagonalisation.pkl'
if a==0.0001:
     filename= 'diagonalisation_continuum.pkl'
mu=1
N=1
g_vec=np.arange(0.1,10,0.1) #free parameter
 

#check if the stored files have the parameters that we want:

with open(filename, 'rb') as file:
     result = pickle.load(file)
     print(result[0])
     print(result[1])
     print(result[-1])

#RUN ALL THE FOLLOWING 5 FUNCTIONS (BOTH FOR a=1 and for a=0)

#exact_diagonalization_and_save(filename,f'diagonalisation a={int(a)}',H_plaquette,1/g_vec**2,a,mu)

#expectation_value_on_gs(['gauss law (0,0)','gauss law (1,0)','gauss law (0,1)','gauss law (1,1)'],[gauss_law_operator], H_plaquette, result_vectors, 1/g_vec**2)
#expectation_value_on_gs(['-H_b','plaquette'],[minus_H_b,plaquette_operator], H_plaquette, result_vectors, 1/g_vec**2,a,mu)

#plot_energy_gap(H_plaquette,1/g_vec**2,a,mu)


regime_comparison(filename, f'diag_el a={int(a)}', f'diag_b a={int(a)}',g_vec,a,mu)