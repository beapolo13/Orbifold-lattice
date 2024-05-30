#pip install qutip

import numpy as np
from numpy import transpose, real, sqrt, sin, cos, linalg, cosh, sinh
import scipy
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
from qutip import tensor, destroy, create, identity
from qutip import *
#from qutip.tensor import tensor
import os

n_modes=8
cutoff=3

def beep():
    os.system("afplay /System/Library/Sounds/Ping.aiff")

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
  print('H_kin', H_kin)

  H_b=0
  H_b+= (map('x','beta_1')*map('x','alpha_1')-map('x','beta_2')*map('x','alpha_2')+map('x','gamma_1')*map('x','delta_1')-map('x','gamma_2')*map('x','delta_2'))**2     #first term
  H_b+= (map('x','beta_1')*map('x','alpha_2')+map('x','beta_2')*map('x','alpha_1')+map('x','gamma_1')*map('x','delta_2')+map('x','gamma_2')*map('x','delta_1'))**2    #second term
  H_b*=(g_1d**2)
  print('H_b',H_b)

  H_el=0
  for m in range(1,5):
    H_el+= (map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)**2  #first term
  H_el+=(map('x',inverse_labels[1])**2 + map('x',inverse_labels[2])**2)*(map('x',inverse_labels[7])**2 + map('x',inverse_labels[8])**2)
  H_el-=(map('x',inverse_labels[1])**2 + map('x',inverse_labels[2])**2)*(map('x',inverse_labels[3])**2 + map('x',inverse_labels[4])**2)
  H_el-=(map('x',inverse_labels[5])**2 + map('x',inverse_labels[6])**2)*(map('x',inverse_labels[7])**2 + map('x',inverse_labels[8])**2)
  H_el+=(map('x',inverse_labels[5])**2 + map('x',inverse_labels[6])**2)*(map('x',inverse_labels[3])**2 + map('x',inverse_labels[4])**2)
  H_el*=g_1d**2/(4*a)
  print('H_el',H_el)

  delta_H=0
  #C=(mu*a*g_1d)**2/8
  #D= 2/(a*(g_1d**2))
  for m in range(1,5):
    delta_H+= ((mu*a*g_1d)**2/8)*(map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)**2
    delta_H-=(mu**2*a/4)* (map('x',inverse_labels[2*m-1])**2 + map('x',inverse_labels[2*m])**2)
    delta_H += (mu*a*g_1d)**2 / (2*(2*a*g_1d**2)**2) #this are the corrections, but they depend on g_1d inversely-quadratic
  print('delta H', delta_H)

  return H_kin + H_b + H_el + delta_H

#i have commented this whole function because i have added the corrections to the hamiltonian above in delta_H
# def H_plaquette_with_corrections(g_1d,a,mu):
#     labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
#     inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
#     return H_plaquette(g_1d,a,mu) + 4*(mu*a*g_1d)**2 / (2*a*(2*a*g_1d**2)**2) 

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
    #delta_H += (mu*a*g_1d)**2 / (2*a*(2*a*g_1d**2)**2) #this are the corrections, but they depend on g_1d inversely-quadratic
    #print(delta_H)
    return H_b +H_el + delta_H

def Hint_with_hc(g_1d,a,mu):
    labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
    inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
    return Hint(g_1d,a,mu) + 4*(mu*a*g_1d)**2 / (2*a*(2*a*g_1d**2)**2)

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
    #delta_H += (mu*a*g_1d)**2 / (2*a*(2*a*g_1d**2)**2)  #this are the corrections, but they depend on g_1d inversely-quadratic
  return H_b + H_el + delta_H

def V_with_hc(g_1d,a,mu):
    labels={'alpha_1':1,'alpha_2':2,'beta_1':3,'beta_2':4,'gamma_1':5,'gamma_2':6,'delta_1':7,'delta_2':8}
    inverse_labels = {value: key for key, value in labels.items()}  #this will be useful for the definition of H_el and delta_H
    return V(g_1d,a,mu) + 4*(mu*a*g_1d)**2 / (2*a*(2*a*g_1d**2)**2)


#we diagonalize H total once (exact diagonalization) and store the vector of groundstates for each value of g_1d, so that we can access 
#the values of this vector anytime in the gs_vector

#we define a function that diagonalizes exactly any input hamiltonian (can be used for any operator), maybe dependent on some parameter
#the ground state vector and the vector of ground state energies are returned as output
def exact_diagonalization(hamiltonian,parameter,parameter1=None,parameter2=None):
    gs_vectors=[]
    gs_energies=[]
    for i in range(len(parameter)):
        H= hamiltonian(parameter[i],parameter1,parameter2)
        energy, vector = qutip.Qobj.groundstate(H)
        gs_vectors+=[vector]
        gs_energies+=[energy]
    gs=[gs_vectors, gs_energies]
    beep()
    plt.plot(parameter,gs_energies,'r--', label='Energy of groundstate of H')
    plt.title('Hamiltonian diagonalization')
    plt.show()
    # A possible sanity check is to see that H with corrections is positive semidefinite (its lowest eigenvalue is greater than 0)
    return gs

def plaquette_operator(g,a,N): #N is the number of plaquettes, one in our case
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
    return (1/(2*N)) *(P_operator()+P_dag_operator())

def expectation_value_on_gs(observable, hamiltonian, parameter,cutoff, parameter1=None,parameter2=None): 
    #here we calculate the expectation value of some arbitrary input observable, on the groundstate of out input hamiltonian
    operator=[]
    y_vec=[]
    x_vec=[]
    for value in parameter:
        x_vec+=[1/value**2]
    states=exact_diagonalization(hamiltonian, parameter,cutoff,parameter1,parameter2)[0]
    for i in range(len(parameter)):
       operator+=[observable(parameter[i],parameter1,parameter2)]
       y_vec+=[[qutip.expect(operator,states[i])]]
    plt.plot(x_vec, y_vec, 'r--')
    plt.title('Expectation value of operator on groundstate of H')
    beep()
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


def gauss_law_operator(site,cut=cutoff):
    x_alpha_dag=(1/np.sqrt(2))*(map('x','alpha_1',cut)- 1j*map('x','alpha_2',cut))
    x_beta_dag=(1/np.sqrt(2))*(map('x','beta_1',cut)- 1j*map('x','beta_2',cut))
    x_delta=(1/np.sqrt(2))*(map('x','delta_1',cut)+ 1j*map('x','delta_2',cut))
    x_gamma=(1/np.sqrt(2))*(map('x','gamma_1',cut)+ 1j*map('x','gamma_2',cut))
    x_alpha=(1/np.sqrt(2))*(map('x','alpha_1',cut)+ 1j*map('x','alpha_2',cut))
    x_beta=(1/np.sqrt(2))*(map('x','beta_1',cut)+ 1j*map('x','beta_2',cut))
    x_delta_dag=(1/np.sqrt(2))*(map('x','delta_1',cut)- 1j*map('x','delta_2',cut))
    x_gamma_dag=(1/np.sqrt(2))*(map('x','gamma_1',cut)- 1j*map('x','gamma_2',cut))

    p_alpha_dag=(1/np.sqrt(2))*(map('p','alpha_1',cut)- 1j*map('p','alpha_2',cut))
    p_beta_dag=(1/np.sqrt(2))*(map('p','beta_1',cut)- 1j*map('p','beta_2',cut))
    p_delta=(1/np.sqrt(2))*(map('p','delta_1',cut)+ 1j*map('p','delta_2',cut))
    p_gamma=(1/np.sqrt(2))*(map('p','gamma_1',cut)+ 1j*map('p','gamma_2',cut))
    p_alpha=(1/np.sqrt(2))*(map('p','alpha_1',cut)+ 1j*map('p','alpha_2',cut))
    p_beta=(1/np.sqrt(2))*(map('p','beta_1',cut)+ 1j*map('p','beta_2',cut))
    p_delta_dag=(1/np.sqrt(2))*(map('p','delta_1',cut)- 1j*map('p','delta_2',cut))
    p_gamma_dag=(1/np.sqrt(2))*(map('p','gamma_1',cut)- 1j*map('p','gamma_2',cut))

    if site=='00':
        return -x_alpha*p_alpha_dag + p_alpha*x_alpha_dag -x_delta*p_delta_dag +p_delta*x_delta_dag
    if site=='10':
        return -x_alpha_dag*p_alpha + p_alpha_dag*x_alpha -x_beta*p_beta_dag +p_beta*x_beta_dag
    if site == '01':
        return -x_gamma*p_gamma_dag + p_gamma*x_gamma_dag -x_delta_dag*p_delta +p_delta_dag*x_delta
    if site == '11':
        return -x_gamma_dag*p_gamma + p_gamma_dag*x_gamma -x_beta_dag*p_beta +p_beta_dag*x_beta
       
   
   
