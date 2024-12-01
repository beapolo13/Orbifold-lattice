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
mu=1
N=1
g_vec=np.arange(0.1,10,0.1) #free parameter




#RUN ALL THE FOLLOWING 5 FUNCTIONS (BOTH FOR a=1 and for a=0)

#exact_diagonalization_and_save('diagonalisation.pkl',f'diagonalisation a={int(a)}.pdf',H_plaquette,1/g_vec**2,a,mu)

#exact_diagonalization_and_save('diag H_kin','diag H_kin',H_kin,1/g_vec**2,a,mu)
#run these four functions:
#exact_diagonalization_and_save('diag deltaH','diag deltaH',deltaH,1/g_vec**2,a,mu)
#exact_diagonalization_and_save('diagonalisation mu=10.plk','diagonalisation mu=10',H_plaquette,1/g_vec**2,1,10)
#exact_diagonalization_and_save('diagonalisation mu=100.plk','diagonalisation mu=100',H_plaquette,1/g_vec**2,1,100)
#density_plot_plaquette(['diagonalisation.pkl', 'diagonalisation mu=10.plk', 'diagonalisation mu=100.plk'],1/g_vec**2)
#exact_diagonalization_and_save_2('2lowest_energy_states.pkl',H_plaquette,1/g_vec**2,a,mu)

#y me falta todavía una funcion (que será eterna) para ver el scaling con el cutoff 

#expectation_value_on_gs('diagonalisation_continuum.pkl', 'gauss law cont',[gauss_law_operator], H_plaquette, 1/g_vec**2)
#expectation_value_on_gs('diagonalisation.pkl','-H_b, plaquette.pdf',[minus_H_b,plaquette_operator], [r'$-H_{B}$', r'Plaquette operator'],H_plaquette, 1/g_vec**2, a,1)

#plot_energy_gap('full diag',H_plaquette,1/g_vec**2,a,mu)

#expectation_value_on_gs('diagonalisation.pkl','contribution of H terms.pdf',[H_el,H_b,H_kin,deltaH], [r'$H_{e l}$',r'$H_{B}$',r'$H_{k i n}$', r'$\Delta H$'],H_plaquette, 1/g_vec**2, a,mu)

#regime_comparison('diagonalisation.pkl', f'diag H_el a={int(a)}', 'diag H_b','diag H_kin','diag deltaH','regime_comparison.pdf', 1/g_vec**2,a,mu)

#bipartite_ent_entropy_plot('diagonalisation.pkl', 'Entanglement entropy.pdf', 1/g_vec**2,a, mu)

#test_function(1/g_vec**2)

#full_full_diagonalization_and_save('full_full_diag.pkl', H_plaquette,1,1,1)
with open('full_full_diag.pkl', 'rb') as file:
      result = pickle.load(file)
      result_energies=result[-2]
      print(result_energies)
      print(np.shape(result_energies))
      print(result_energies[0],result_energies[5000])