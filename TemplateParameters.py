#Python Modules
from math import sqrt, pi, pow
import os
import numpy as np
import binary_sequences

#####################################################################################################
master_control = True
run_seed = 99.0
Total_runs = 1     #Number of experimental runs - used for cluster computing. For testing purposes set Total_runs = 1

"""System Parameters & Output Specifications"""
t_0 = 0
tau = 0.0                   #initial time
tau_max = 1000          #max sim time
t_frequent = tau_max*0.0001
t_infrequent = tau_max*0.001              #total number of Gillespie steps between infrequent outputs

#output_all = 0              #If = 0 no polymer concentrations output, if = 1 polymer concentrations output for all polymers at each time step
#print_functional = 0        #Outputs concentrations of all sequences with a functional moeity

  
"""User defined Monomer Species & Statistics"""
monomer_species = ['0', '1']
M_N = [500, 500]                 #Initial number of each monomeric species at start of simulation!
m = len(monomer_species)
mono_IDs = []

''' Monomer Chemistry'''
#stability  1s contribute to stability
f_a=0.5     #Three parameters for stability sigmoid function
f_b=2.0    # These parameters should be set so that sigmoid(f_a, f_b, f_c, n) decreases with larger n
f_c=10.0

#replication    0s contribute to replication
r_a= 0.5   #Three parameters for replication sigmoid function
r_b= 2.0
r_c=10.0 

"""Microscopic Reaction Rates"""
monomer_flux = 0.0
# kp = 0.0005                 #Polymerization Rate
# kh = 0.5                  #degradation rate
# kr = 0.005                  #Replication rate
km = 0.0                    #Mutation Rate
kd = 0.00                  # Death rate

"""Functional Motifs"""
NonTrivial=False #If this is true the system will initialize non-trivial replicator motifs
NT_len=4  #The length of non-trivial replicator motifs
NT_N = 3  #total number of non-trivial replicator functional motifs

Functional= False #If this is true the system will initialize functional motifs
F_len = 4   #length of functional motifs
F_N = 5     # Number of different functional motifs

F_motifs = []  #List of T-motifs
NT_motifs = [] #List of NT-motifs


#####################################################################################################
### Do not modify parameters below this line!
#####################################################################################################

"""Global variables not to be changed (program will modify)"""

#Sequences should be made into a list of dictionaries, where the list is indexed by length (smaller search space) 

sequences = []           #List of all sequences present in the system
seq_dict = {}
max_ID = 0

Npoly = 0                 # Total number of polymers in system (summed over lattice sites)
Nmono = 0
Ntot = 0                 # Total number of molecules (monomers + polymers) in the system
tot_species = 0          # Total number of replicator species that have ever existed in sim


Atot = 0.0                                        #Total global propensity for all events
Ap_p = 0.0
Ap_h = 0.0
Ap_r = 0.0
Ap_d = 0.0 



polymerization_events = 0            # Tracks number of replication events
hydrolysis_events = 0            # Tracks number of degredation events
replication_events = 0 
death_events = 0
null_replication = 0
polymerized_to_replicator = 0.0
replicated_mass = 0.0

M = sum(M_N)

max_length = 0.0 
max_length_ID = 0
max_population = 0.0 
max_pop_ID = 0

# if monomer_flux != 0.0: 
#     if Functional == True and NonTrivial == False:
#         dirname = ('data/FL%i_FN%i_kp%.4f_kh%.4f_kr%4f_kd%.4f_Mf%.4f' % (F_len, F_N, kp, kh, kr, kd, monomer_flux))

#         if not os.path.exists(dirname):
#             os.makedirs(dirname)

#     elif Functional == True and NonTrivial == True:
#         dirname = ('data/NTL%i_NTN%i_FL%i_FN%i_kp%.4f_kd%.4f_kr%.4f_kd%.4f_Mf%.4f' % (NT_len, NT_N, F_len, F_N, kp, kh, kr, kd, monomer_flux))

#         if not os.path.exists(dirname):
#             os.makedirs(dirname)

#     elif NonTrivial== True:
#         dirname = ('data/NTL%i_NTN%i_kp%.4f_kh%.4f_kr%4f_kd%4f_Mf%4f' % (NT_len, NT_N, kp, kh, kr, kd, monomer_flux))

#         if not os.path.exists(dirname):
#             os.makedirs(dirname)

#     else:
#         dirname = ('data/kp%.4f_kh%.4f_kr%.4f_kd%.4f_Mf%.4f' % (kp, kh, kr, kd, monomer_flux))

#         if not os.path.exists(dirname):
#             os.makedirs(dirname)


# else: 
#     if Functional == True and NonTrivial == False:
#         dirname = ('data/FL%i_FN%i_kp%.4f_kh%.4f_kr%4f_kd%.4f_M%.1f' % (F_len, F_N, kp, kh, kr, kd, sum(M_N)))

#         if not os.path.exists(dirname):
#             os.makedirs(dirname)

#     elif Functional == True and NonTrivial == True:
#         dirname = ('data/NTL%i_NTN%i_FL%i_FN%i_kp%.4f_kd%.4f_kr%.4f_kd%.4f_M%.1f' % (NT_len, NT_N, F_len, F_N, kp, kh, kr, kd, sum(M_N)))

#         if not os.path.exists(dirname):
#             os.makedirs(dirname)

#     elif NonTrivial== True:
#         dirname = ('data/NTL%i_NTN%i_kp%.4f_kh%.4f_kr%4f_kd%4f_M%.1f' % (NT_len, NT_N, kp, kh, kr, kd, sum(M_N)))

#         if not os.path.exists(dirname):
#             os.makedirs(dirname)

#     else:
#         dirname = ('data/kp%.4f_kh%.4f_kr%.4f_kd%.4f_M%.1f' % (kp, kh, kr, kd, sum(M_N)))

#         if not os.path.exists(dirname):
#             os.makedirs(dirname)
