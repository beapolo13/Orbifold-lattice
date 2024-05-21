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
g_vec=np.arange(0.3,10,1) #free parameter
#sometimes we need to compute quantities and plots with respect to 1/g^2, so we compute this vector too: 


#expectation_value_on_gs(plaquette_operator,H_plaquette, g_vec,3,parameter1=1,parameter2=1)
exact_diagonalization(H_plaquette,g_vec,cutoff=cutoff,parameter1=1,parameter2=1)
