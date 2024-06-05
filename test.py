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
a=0.0001  #a=1 by default, a=0.0001 for the continuum limit
mu=1
N=1
g_vec=np.arange(0.1,10,0.1) #free parameter





#RUN ALL THE FOLLOWING 5 FUNCTIONS (BOTH FOR a=1 and for a=0)

#exact_diagonalization_and_save(filename,f'diagonalisation a={int(a)}',H_plaquette,1/g_vec**2,a,mu)

#exact_diagonalization_and_save('diag H_b','diag H_b',H_b,1/g_vec**2,a,mu)

#expectation_value_on_gs('diagonalisation_continuum.pkl', 'gauss law cont',[gauss_law_operator], H_plaquette, 1/g_vec**2)
#expectation_value_on_gs('diagonalisation_continuum.pkl','-H_b, plaquette continuum',[minus_H_b,plaquette_operator], H_plaquette, 1/g_vec**2, a,mu)

#plot_energy_gap('full diag',H_plaquette,1/g_vec**2,a,mu)


regime_comparison('diagonalisation_continuum.pkl', f'diag H_el a={int(a)}', 'diag H_b','regime_comparison_cont', 1/g_vec**2,a,mu)