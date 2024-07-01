#pip install qutip
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
import os
import winsound
import pickle
import matplotlib.colors as mcolors

#Parameters for plots
params = {'axes.linewidth': 1.4,
         'axes.labelsize': 20,
         'axes.titlesize': 25,
         'axes.linewidth': 1.5,
         'lines.markeredgecolor': "black",
     	'lines.linewidth': 1.5,
         'xtick.labelsize': 18,
         'ytick.labelsize': 18,
         "text.usetex": True,
         "font.family": "serif",
         "font.serif": ["Palatino"]
         }
plt.rcParams.update(params)


n_modes=8
cutoff=3

def beep():
    #os.system("afplay /System/Library/Sounds/Ping.aiff")
    freq=800
    dur=500
    winsound.Beep(freq,dur)
  


#definition of ladder operator generation function(create/destroy on mode i)
def ladder(type,mode,cutoff):     #type=True for destroy(a), =False for create (adag)  #cutoff is the maximum dimension of Fock space for ladder operators
  operator_string=[qeye(cutoff) for _ in range(n_modes)]
  if type==True:
    operator_string[mode-1]=create(cutoff)
  elif type==False:
    operator_string[mode-1]=destroy(cutoff)
  #print(operator_string)
  return tensor(operator_string)


#variables will be defined by a type=('x' or 'p'), and a mode label (string)
def map(type,label):
  labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
  inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
  if type=='x':
    return (1/np.sqrt(2))*(ladder(True,labels[label],cutoff) + ladder(False,labels[label],cutoff))
  elif type=='p':
    return (1j/np.sqrt(2))*(ladder(False,labels[label],cutoff) - ladder(True,labels[label],cutoff))
  
#construct hamiltonian

def H_plaquette(g_1d,a,mu):
  labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
  inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
  H_kin=0
  for label in labels:
    H_kin+=(1/2)*(map('p',label)**2)
  #print('H_kin', H_kin)

  H_b=0
  H_b+= (map('x','beta_1')*map('x','alpha_1')-map('x','beta_2')*map('x','alpha_2')+map('x','gamma_1')*map('x','delta_1')-map('x','gamma_2')*map('x','delta_2'))**2     #first term
  H_b+= (map('x','beta_1')*map('x','alpha_2')+map('x','beta_2')*map('x','alpha_1')+map('x','gamma_1')*map('x','delta_2')+map('x','gamma_2')*map('x','delta_1'))**2    #second term
  H_b*=(g_1d**2)
  #print('H_b',H_b)

  H_el=0
  for m in range(1,5):
    H_el+= (map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)**2  #first term
  H_el+=(map('x',inverse_labels[1])**2 + map('x',inverse_labels[2])**2)*(map('x',inverse_labels[7])**2 + map('x',inverse_labels[8])**2)
  H_el-=(map('x',inverse_labels[1])**2 + map('x',inverse_labels[2])**2)*(map('x',inverse_labels[3])**2 + map('x',inverse_labels[4])**2)
  H_el-=(map('x',inverse_labels[5])**2 + map('x',inverse_labels[6])**2)*(map('x',inverse_labels[7])**2 + map('x',inverse_labels[8])**2)
  H_el+=(map('x',inverse_labels[5])**2 + map('x',inverse_labels[6])**2)*(map('x',inverse_labels[3])**2 + map('x',inverse_labels[4])**2)
  H_el*=g_1d**2/(4*a)
  #print('H_el',H_el)

  delta_H=0
  #C=(mu*a*g_1d)**2/8
  #D= 2/(a*(g_1d**2))
  for m in range(1,5):
    delta_H+= ((mu*a*g_1d)**2/8)*(map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)**2
    delta_H-=(mu**2*a/4)* (map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)
    #delta_H += (mu*a*g_1d)**2 / (2*(2*a*g_1d**2)**2) #this are the corrections, but they depend on g_1d inversely-quadratic
  #print('delta H', delta_H)

  return H_kin + H_b + H_el + delta_H

def H_kin(g_1d,a,mu):
  labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
  inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
  H_kin=0
  for label in labels:
    H_kin+=(1/2)*(map('p',label)**2)
  return H_kin

def H_el(g_1d,a,mu):
  labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
  inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
  H_el=0
  for m in range(1,5):
    H_el+= (map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)**2  #first term
  H_el+=(map('x',inverse_labels[1])**2 + map('x',inverse_labels[2])**2)*(map('x',inverse_labels[7])**2 + map('x',inverse_labels[8])**2)
  H_el-=(map('x',inverse_labels[1])**2 + map('x',inverse_labels[2])**2)*(map('x',inverse_labels[3])**2 + map('x',inverse_labels[4])**2)
  H_el-=(map('x',inverse_labels[5])**2 + map('x',inverse_labels[6])**2)*(map('x',inverse_labels[7])**2 + map('x',inverse_labels[8])**2)
  H_el+=(map('x',inverse_labels[5])**2 + map('x',inverse_labels[6])**2)*(map('x',inverse_labels[3])**2 + map('x',inverse_labels[4])**2)
  H_el*=g_1d**2/(4*a)
  return H_el

def H_b(g_1d,a,mu):
  labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
  inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
  H_b=0
  H_b+= (map('x','beta_1')*map('x','alpha_1')-map('x','beta_2')*map('x','alpha_2')+map('x','gamma_1')*map('x','delta_1')-map('x','gamma_2')*map('x','delta_2'))**2     #first term
  H_b+= (map('x','beta_1')*map('x','alpha_2')+map('x','beta_2')*map('x','alpha_1')+map('x','gamma_1')*map('x','delta_2')+map('x','gamma_2')*map('x','delta_1'))**2    #second term
  H_b*=(g_1d**2)
  return H_b

def minus_H_b(g_1d,a,mu):
   return -H_b(g_1d,a,mu)

def deltaH(g_1d,a,mu):
  labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
  inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
  delta_H=0
  #C=(mu*a*g_1d)**2/8
  #D= 2/(a*(g_1d**2))
  for m in range(1,5):
    delta_H+= ((mu*a*g_1d)**2/8)*(map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)**2
    delta_H-=(mu**2*a/4)* (map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)
    #delta_H += (mu*a*g_1d)**2 / (2*(2*a*g_1d**2)**2) #this are the corrections, but they depend on g_1d inversely-quadratic
  return delta_H

def Hfree(a,mu,cutoff):
    labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
    inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
    H_kin=0
    for label in labels:
        H_kin+=(1/2)*(map('p',label)**2)

    decoupledH=0
    for m in range(1,5):
        decoupledH-=(mu**2*a/4)* (map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)

    return H_kin + decoupledH

def Hint(g_1d,a,mu):
    labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
    inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
    H_b=0
    H_b+= (map('x','beta_1')*map('x','alpha_1')-map('x','beta_2')*map('x','alpha_2')+map('x','gamma_1')*map('x','delta_1')-map('x','gamma_2')+map('x','delta_2'))**2     #first term
    H_b+= (map('x','beta_1')*map('x','alpha_2')+map('x','beta_2')*map('x','alpha_1')+map('x','gamma_1')*map('x','delta_2')+map('x','gamma_2')+map('x','delta_1'))**2    #second term
    H_b*=(g_1d**2)

    H_el=0
    for m in range(1,5):
        H_el+= (map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)**2  #first term
    H_el+=(map('x',inverse_labels[1])**2 + map('x',inverse_labels[2])**2)*(map('x',inverse_labels[7])**2 + map('x',inverse_labels[8])**2)
    H_el-=(map('x',inverse_labels[1])**2 + map('x',inverse_labels[2])**2)*(map('x',inverse_labels[3])**2 + map('x',inverse_labels[4])**2)
    H_el-=(map('x',inverse_labels[5])**2 + map('x',inverse_labels[6])**2)*(map('x',inverse_labels[7])**2 + map('x',inverse_labels[8])**2)
    H_el+=(map('x',inverse_labels[5])**2 + map('x',inverse_labels[6])**2)*(map('x',inverse_labels[3])**2 + map('x',inverse_labels[4])**2)
    H_el*=g_1d**2/(4*a)
    
    
    delta_H=0
    #C=(mu*a*g_1d)**2/8
    #D= 2/(a*(g_1d**2))
    for m in range(1,5):
      delta_H+= ((mu*a*g_1d)**2/8)*(map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)**2
      delta_H += (mu*a*g_1d)**2 / (2*a*(2*a*g_1d**2)**2) #this are the corrections, but they depend on g_1d inversely-quadratic
    #print(delta_H)
    return H_b +H_el + delta_H

#Split hamiltonian

def T():  #kinetic term
  labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
  inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
  H_kin=0
  for label in labels:
    H_kin+=(1/2)*(map('p',label)**2)
  return H_kin

def V(g_1d,a,mu): #potential term
  labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
  inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
  H_b=0
  H_b+= (map('x','beta_1')*map('x','alpha_1')-map('x','beta_2')*map('x','alpha_2')+map('x','gamma_1')*map('x','delta_1')-map('x','gamma_2')+map('x','delta_2'))**2     #first term
  H_b+= (map('x','beta_1')*map('x','alpha_2')+map('x','beta_2')*map('x','alpha_1')+map('x','gamma_1')*map('x','delta_2')+map('x','gamma_2')+map('x','delta_1'))**2    #second term
  H_b*=g_1d**2

  H_el=0
  for m in range(1,5):
    H_el+= (map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)**2  #first term
  H_el+=(map('x',inverse_labels[1])**2 + map('x',inverse_labels[2])**2)*(map('x',inverse_labels[7])**2 + map('x',inverse_labels[8])**2)
  H_el-=(map('x',inverse_labels[1])**2 + map('x',inverse_labels[2])**2)*(map('x',inverse_labels[3])**2 + map('x',inverse_labels[4])**2)
  H_el-=(map('x',inverse_labels[5])**2 + map('x',inverse_labels[6])**2)*(map('x',inverse_labels[7])**2 + map('x',inverse_labels[8])**2)
  H_el+=(map('x',inverse_labels[5])**2 + map('x',inverse_labels[6])**2)*(map('x',inverse_labels[3])**2 + map('x',inverse_labels[4])**2)
  H_el*=g_1d**2/(4*a)


  delta_H=0
  #C=(mu*a*g_1d)**2/8
  #D= 2/(a*(g_1d**2))
  for m in range(1,5):
    delta_H+= ((mu*a*g_1d)**2/8)*(map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)**2
    delta_H-=(mu**2*a/4)* (map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)
    delta_H += (mu*a*g_1d)**2 / (2*a*(2*a*g_1d**2)**2)  #this are the corrections, but they depend on g_1d inversely-quadratic
  return H_b + H_el + delta_H


#we diagonalize H total once (exact diagonalization) and store the vector of groundstates for each value of g_1d, so that we can access 
#the values of this vector anytime in the gs_vector

#we define a function that diagonalizes exactly any input hamiltonian (can be used for any operator), maybe dependent on some parameter
#the ground state vector and the vector of ground state energies are returned as output
def exact_diagonalization_and_save(filename,savefig_name,hamiltonian,parameter,a,mu):
  exists_diag=input('is there a file with diagonalization of H?')
  if exists_diag=='False':
    gs_vectors=[]
    gs_energies=[]
    times_vector=[]
    for i in range(len(parameter)):
        time0=time.time()
        H= hamiltonian(parameter[i],a,mu)
        energy, vector = qutip.Qobj.groundstate(H)
        timef=time.time()
        delta_time=timef-time0
        gs_vectors+=[vector]
        gs_energies+=[energy]
        times_vector+=[delta_time]
        print('g value', parameter[i])
        print('H_kin:', qutip.expect(H_kin(parameter[i],a,mu),vector),'H_el:', qutip.expect(H_el(parameter[i],a,mu),vector),'H_b:', qutip.expect(H_b(parameter[i],a,mu),vector),'deltaH:', qutip.expect(deltaH(parameter[i],a,mu),vector))
    result=['parameters:',[parameter, a,mu],times_vector,gs_vectors, gs_energies]
    result_energies=result[-1]
    result_vectors=result[-2]
    result_times_vector=result[-3]
    with open(filename, 'wb') as file:
        pickle.dump(result, file)
  elif exists_diag=='True':
    with open(filename, 'rb') as file:
      result = pickle.load(file)
      result_energies=result[-1]
      result_vectors=result[-2]
      result_times_vector=result[-3]

  beep()
  plt.plot(parameter,result_energies,'r--', label=f'Energy of groundstate of H when a={a}')
  #plt.plot(parameter,result_times_vector,'b--', label='Time to diagonalize H')
  plt.xscale('log')
  plt.xlabel(r'$1/g^2$')
  plt.ylabel('Groundstate energy')
  plt.legend(['Energy of g.s', 'Time to diagonalize'])
  #plt.title(f'Hamiltonian diagonalization when a={a}')
  plt.savefig(savefig_name)
  plt.show()
    # A possible sanity check is to see that H with corrections is positive semidefinite (its lowest eigenvalue is greater than 0)
  return result

def full_diagonalization_and_save(filename,hamiltonian,parameter,a,mu):
  groundstates=[]
  first_excited=[]
  gaps=[]
  for i in range(len(parameter)):
    H= hamiltonian(parameter[i],a,mu)
    energy_gs,energy_exc = qutip.Qobj.eigenenergies(H)[0], qutip.Qobj.eigenenergies(H)[1]
    groundstates+=[energy_gs]
    first_excited+=[energy_exc]
    gaps+=[energy_exc-energy_gs]
    print('g value', parameter[i], 'gap=',energy_exc-energy_gs)
  data=['parameters',['g_vec',parameter,'a:',a,'mu:',mu],groundstates,first_excited,gaps]
  with open(filename, 'wb') as file:
      pickle.dump(data, file)
  beep()
  plt.plot(parameter,gaps,'r--')
  plt.title(f'Gap between first and ground level, a={a}')
  plt.show()
  return data

def plaquette_operator(g,a,N=1): #parameters=[a,N] where N is the number of plaquettes, one in our case
    def P_operator():
        x_alpha_dag=(1/np.sqrt(2))*(map('x','alpha_1')- 1j*map('x','alpha_2'))
        x_beta_dag=(1/np.sqrt(2))*(map('x','beta_1')- 1j*map('x','beta_2'))
        x_delta=(1/np.sqrt(2))*(map('x','delta_1')+ 1j*map('x','delta_2'))
        x_gamma=(1/np.sqrt(2))*(map('x','gamma_1')+ 1j*map('x','gamma_2'))
        return x_alpha_dag*x_beta_dag*x_delta*x_gamma
    def P_dag_operator():
        x_alpha=(1/np.sqrt(2))*(map('x','alpha_1')+ 1j*map('x','alpha_2'))
        x_beta=(1/np.sqrt(2))*(map('x','beta_1')+ 1j*map('x','beta_2'))
        x_delta_dag=(1/np.sqrt(2))*(map('x','delta_1')- 1j*map('x','delta_2'))
        x_gamma_dag=(1/np.sqrt(2))*(map('x','gamma_1')- 1j*map('x','gamma_2'))
        return x_gamma_dag*x_delta_dag*x_beta*x_alpha
    return (1/(2)) *(P_operator()+P_dag_operator())

def expectation_value_on_gs(filename,savefig_name,observable_list,legend_string, hamiltonian, parameter, *args): 
  exists_diag=input('is there a file with diagonalization of H?')
  if exists_diag=='False':
    result= exact_diagonalization_and_save(filename,filename,hamiltonian, parameter, *args)
    result_energies=result[-1]
    result_vectors=result[-2]
    result_times_vector=result[-3]
  elif exists_diag=='True':
    with open(filename, 'rb') as file:
      result = pickle.load(file)
      result_energies=result[-1]
      result_vectors=result[-2]
      result_times_vector=result[-3]
  states=result_vectors
  #here we calculate the expectation value of some arbitrary input observable, on the groundstate of out input hamiltonian
  y_vec_list=[[] for _ in range(len(observable_list))]
  for j in range(len(observable_list)):
    operator=[]
    for i in range(len(parameter)):
      operator+=[observable_list[j](parameter[i],*args)]
      y_vec_list[j]+=[qutip.expect(operator[i],states[i])]
    plt.plot(parameter, y_vec_list[j],linestyle='dashed')
  plt.xscale('log')
  plt.xlabel(r'$1/g^2$')
  plt.ylabel('Energy')
  #plt.title('Expectation value of operators on groundstate of H')
  plt.legend(legend_string,fontsize=16)
  beep()
  plt.savefig(savefig_name)
  plt.show()
  return

#function to tell whether two operators commute or not
def commutator(operator1,operator2,parameter,parameter1=None,parameter2=None):
    answer=True
    for i in range(len(parameter)):
        #print(qutip.commutator(V(g_vec[i],a,mu),H_plaquette(g_vec[i],a,mu),'normal'))
        if qutip.commutator(operator1(parameter[i],parameter1,parameter2),operator2(parameter[i],parameter1,parameter2),'normal') != 0:     
            answer = False
            break
            print('The observables do not commute')
    if answer==True:
       print('commuting observables')
    return answer


def gauss_law_operator(g):
    x_alpha_dag=(1/np.sqrt(2))*(map('x','alpha_1')- 1j*map('x','alpha_2'))
    x_beta_dag=(1/np.sqrt(2))*(map('x','beta_1')- 1j*map('x','beta_2'))
    x_delta=(1/np.sqrt(2))*(map('x','delta_1')+ 1j*map('x','delta_2'))
    x_gamma=(1/np.sqrt(2))*(map('x','gamma_1')+ 1j*map('x','gamma_2'))
    x_alpha=(1/np.sqrt(2))*(map('x','alpha_1')+ 1j*map('x','alpha_2'))
    x_beta=(1/np.sqrt(2))*(map('x','beta_1')+ 1j*map('x','beta_2'))
    x_delta_dag=(1/np.sqrt(2))*(map('x','delta_1')- 1j*map('x','delta_2'))
    x_gamma_dag=(1/np.sqrt(2))*(map('x','gamma_1')- 1j*map('x','gamma_2'))

    p_alpha_dag=(1/np.sqrt(2))*(map('p','alpha_1')- 1j*map('p','alpha_2'))
    p_beta_dag=(1/np.sqrt(2))*(map('p','beta_1')- 1j*map('p','beta_2'))
    p_delta=(1/np.sqrt(2))*(map('p','delta_1')+ 1j*map('p','delta_2'))
    p_gamma=(1/np.sqrt(2))*(map('p','gamma_1')+ 1j*map('p','gamma_2'))
    p_alpha=(1/np.sqrt(2))*(map('p','alpha_1')+ 1j*map('p','alpha_2'))
    p_beta=(1/np.sqrt(2))*(map('p','beta_1')+ 1j*map('p','beta_2'))
    p_delta_dag=(1/np.sqrt(2))*(map('p','delta_1')- 1j*map('p','delta_2'))
    p_gamma_dag=(1/np.sqrt(2))*(map('p','gamma_1')- 1j*map('p','gamma_2'))

    #returns a list of operators of g for each site, numbered 00, 10,01,11 
    return [-x_alpha*p_alpha_dag + p_alpha*x_alpha_dag -x_delta*p_delta_dag +p_delta*x_delta_dag, -x_alpha_dag*p_alpha + p_alpha_dag*x_alpha -x_beta*p_beta_dag +p_beta*x_beta_dag, -x_gamma*p_gamma_dag + p_gamma*x_gamma_dag -x_delta_dag*p_delta +p_delta_dag*x_delta, -x_gamma_dag*p_gamma + p_gamma_dag*x_gamma -x_beta_dag*p_beta +p_beta_dag*x_beta]
    

def plot_energy_gap(filename,hamiltonian,parameter,a,mu):  #if there is no diagonalization then filename=the name how we want to save it
  exists_diag= input('Is there a FULL diagonalization of H in this regime?')
  if exists_diag== 'True':
    with open(filename, 'rb') as file:
      result = pickle.load(file)
      groundstates=result[-3]
      first_excited=result[-2]
      gaps=result[-1]
  if exists_diag== 'False':
    full_diagonalization_and_save(filename,hamiltonian,parameter,a,mu)

  plt.plot(parameter,gaps,'r--')
  #plt.title('Gap between first and ground level')
  plt.xscale('log')
  plt.yscale('log')
  plt.xlabel(r'$1/g^2$')
  plt.ylabel('Energy gap')
  plt.savefig('Energy gap')
  plt.show()
  return


def regime_comparison(filenameH, filename_el, filename_B,filename_kin, filename_delta, savefig_name, parameter,a,mu):
  #before running this function there must be a file with the diagonalisation results of H_el and H_b (uncomment the two lines above:)
  exists_diagonalisation=input('Is there a file with H_total diagonalisation in this regime?')
  if exists_diagonalisation == 'False':
    exact_diagonalization_and_save(filenameH,filenameH,H_plaquette,parameter,a,mu)

  exists_el_diagonalisation=input('Is there a file with H_el diagonalisation in this regime?')
  if exists_el_diagonalisation == 'False':
    exact_diagonalization_and_save(filename_el,filename_el,H_el,parameter,a,mu)

  exists_mag_diagonalisation=input('Is there a file with H_b diagonalisation?')
  if exists_mag_diagonalisation == 'False':
    exact_diagonalization_and_save(filename_B,filename_B,H_b,parameter,a,mu)

  exists_kin_diagonalisation=input('Is there a file with H_kin diagonalisation?')
  if exists_kin_diagonalisation == 'False':
    exact_diagonalization_and_save(filename_kin,filename_kin,H_kin,parameter,a,mu)

  exists_delta_diagonalisation=input('Is there a file with H_delta diagonalisation?')
  if exists_delta_diagonalisation == 'False':
    exact_diagonalization_and_save(filename_delta,filename_delta,deltaH,parameter,a,mu)
  
  #then we recover the five diagonalisations 
  #that of the full hamiltonian
  with open(filenameH, 'rb') as file:
    result = pickle.load(file)
    result_energies=result[-1]
    result_vectors=result[-2]
  with open(filename_el, 'rb') as file_el:
    electric = pickle.load(file_el)
    electric_energies=electric[-1]
    electric_vectors=electric[-2]
     
  with open(filename_B, 'rb') as file_mag:
    magnetic = pickle.load(file_mag)
    magnetic_energies=magnetic[-1]
    magnetic_vectors=magnetic[-2]

  with open(filename_kin, 'rb') as file_kin:
    kinetic = pickle.load(file_kin)
    kinetic_energies=kinetic[-1]
    kinetic_vectors=kinetic[-2]

  with open(filename_delta, 'rb') as file_delta:
    delta = pickle.load(file_delta)
    delta_energies=delta[-1]
    delta_vectors=delta[-2]

  electric_overlaps=[]
  magnetic_overlaps=[]
  kinetic_overlaps=[]
  delta_overlaps=[]
  for i in range(len(parameter)):
    electric_overlaps+=[np.abs(qutip.Qobj.overlap(result_vectors[i],electric_vectors[i]))]
    magnetic_overlaps+=[np.abs(qutip.Qobj.overlap(result_vectors[i],magnetic_vectors[i]))]
    kinetic_overlaps+=[np.abs(qutip.Qobj.overlap(result_vectors[i],kinetic_vectors[i]))]
    delta_overlaps+=[np.abs(qutip.Qobj.overlap(result_vectors[i],delta_vectors[i]))]
  
  plt.plot(parameter,electric_overlaps,'r--', label='overlap with H_el groundstates')
  plt.plot(parameter,magnetic_overlaps,'b--', label='overlap with H_b ground states')
  plt.plot(parameter,kinetic_overlaps,'g--', label='overlap with H_kin ground states')
  plt.plot(parameter,delta_overlaps,color='orange', linestyle='dashed', label='overlap with deltaH ground states')
  plt.xscale('log')
  plt.xlabel(r'$1/g^2$')
  plt.ylabel('Overlap')
  plt.legend([r'$H_{e l}$',r'$H_{B}$',r'$H_{k i n}$', r'$\Delta H$'],fontsize=16)
  #plt.title('Regime comparison')
  plt.savefig(savefig_name)
  plt.show()

  return 

def bipartite_ent_entropy_plot(filename, savefig_name, parameter,*args):
  exists_diag=input('is there a file with diagonalization of H?')
  if exists_diag=='False':
    result= exact_diagonalization_and_save(filename,filename,H_plaquette,parameter, *args)
    result_energies=result[-1]
    result_vectors=result[-2]
    result_times_vector=result[-3]
  elif exists_diag=='True':
    with open(filename, 'rb') as file:
      result = pickle.load(file)
      result_energies=result[-1]
      result_vectors=result[-2]
      result_times_vector=result[-3]
  states=result_vectors
  entropies=[]
  for i in range(len(parameter)):
    rho=states[i] * states[i].dag()
    entropy=entropy_mutual(rho, [0,1,2,3], [4,5,6,7], base=2, sparse=False)
    entropies+=[entropy]
    print(parameter[i],entropy)
  plt.plot(parameter,entropies,'r--')
  #plt.title('Bipartite entanglement entropy of groundstate')
  plt.xscale('log')
  plt.xlabel(r'$1/g^2$')
  plt.ylabel('Entropy')
  plt.savefig(savefig_name)
  plt.show()
  return


def density_plot_plaquette(filename_list,parameter): 
  result=[]
  result_energies=[]
  result_times_vector=[]
  result_vectors=[[],[],[]]
  for i in range(3):
      with open(filename_list[i], 'rb') as file:
        result= pickle.load(file)
        result_energies+=result[-1]
        result_vectors[i]+=result[-2]
        result_times_vector+=result[-3]


  X=parameter
  Y=np.array([1,10,100])
  
  
  X_grid, Y_grid =np.meshgrid(X,Y)
  
  grid= np.vstack([X_grid.ravel(),Y_grid.ravel()]).T 

  W= [[qutip.expect(plaquette_operator(X[j],1,1),result_vectors[i][j])for j in range(len(X))]for i in range(len(Y))]

  plt.plot(X,W[0])
  plt.plot(X,W[1])
  plt.plot(X,W[2])
  plt.xscale('log')
  plt.show()
  fig,ax=plt.subplots(figsize=(10,6))
  # for i in range(len(Y)):
  #   for j in range(len(X)):
  #     ax.text(X_grid[i,j],Y_grid[i,j], f'{X_grid[i,j]:.1},{Y_grid[i,j]}', ha='center', va='center', fontsize=8, color='blue')
  c=ax.pcolormesh(X_grid,Y_grid,W,shading='auto',cmap='viridis')
  fig.colorbar(c,ax=ax)
  ax.set_xlim(X.min(), X.max())
  print(X.max(),X.min())
  ax.set_ylim(Y.min() , Y.max())
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.grid(True, which='both', linestyle='--')
  ax.set_xlabel('1/g^2')
  ax.set_ylabel('mu')
  plt.title('Density plot of groundstate energy wrt mu,g')
  plt.savefig('density plot plaquette operator')
  plt.show()


def exact_diagonalization_and_save_2(filename,hamiltonian,parameter,a,mu):
  groundstates=[]
  first_excited=[]
  gaps=[]
  state_vectors=[]
  for i in range(len(parameter)):
    H= hamiltonian(parameter[i],a,mu)
    energies, states = qutip.Qobj.eigenstates(H,sparse=False, sort='low', eigvals=2, tol=0, maxiter=100000)
    groundstates+=[energies[0]]
    first_excited+=[energies[1]]
    gaps+=[energies[1]-energies[0]]
    state_vectors+=[states]
    print('g value', parameter[i], 'gap=',energies[1]-energies[0])
  data=['parameters',['g_vec',parameter,'a:',a,'mu:',mu],groundstates,first_excited,gaps,state_vectors]
  with open(filename, 'wb') as file:
      pickle.dump(data, file)
  beep()
  plt.plot(parameter,gaps,'r--')
  plt.title(f'Gap between first and ground level, a={a}')
  plt.xscale('log')
  plt.yscale('log')
  plt.savefig('Mass gap')
  plt.show()
  return data


def test_function(parameter): 
  result=[]
  result_energies=[]
  result_times_vector=[]
  result_vectors=[]

  result2=[]
  result_energies2=[]
  result_times_vector2=[]
  result_vectors2=[]

  with open('diagonalisation mu=10.plk', 'rb') as file:
    result= pickle.load(file)
    result_energies+=result[-1]
    result_vectors+=result[-2]
    result_times_vector+=result[-3]

  with open('diagonalisation mu=100.plk', 'rb') as file:
    result2= pickle.load(file)
    result_energies2+=result2[-1]
    result_vectors2+=result2[-2]
    result_times_vector2+=result2[-3]


  X=parameter
  Y1= [qutip.expect(plaquette_operator(X[j],1,1),result_vectors[j])for j in range(len(X))]
  Y2= [qutip.expect(plaquette_operator(X[j],1,1),result_vectors2[j])for j in range(len(X))]

  plt.plot(X,Y1)
  plt.plot(X,Y2)
  
  plt.xscale('log')
  plt.show()
  