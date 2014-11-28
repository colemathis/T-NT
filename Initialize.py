# copyright Sara Imari Walker 2011
#Edited By Cole Mathis 2014
import  Parameters
from  Polymers import *
import numpy as np
import math 

################################################################################
def initialize(sequences):
    """This function initializes the first m ID's with the m-monomer species present in the model"""

    for i in range(0, ( Parameters.m)):
        #Loop over monomer species in system
        ID =  Parameters.tot_species
        
        species =  Parameters.monomer_species[i]
        concentration=np.zeros( Parameters.m)
        concentration[i]=1
        sequences.extend([ Polymer(ID,  Parameters.M_N[i], 1, species,concentration)])
        
        Parameters.Ntot +=  Parameters.M_N[i]
        Parameters.Nmono +=  Parameters.M_N[i]
        Parameters.tot_species += 1
        Parameters.seq_dict[species] = ID
        Parameters.mono_IDs.append(ID)
        
################################################################################ 
def update_propensities(sequences, tau_step):
    """updates all reaction propensities"""
    import  Parameters
    from  Parameters import m, kp, kh, kr, Ntot, Nmono, F_motifs, kd
    from  Parameters import Ap_p, Ap_h, Ap_r
    """Compute and Update Reaction Propensities"""

    Ap_p = 0.0
    Ap_h = 0.0
    Ap_r = 0.0
    Ap_d = 0.0
    
    if Ntot < 1:
        print "Ntot < 1"
        exit()           

    for seq in sequences:
        if seq.pop != 0.0:
            """Calculate propensities for each polymer"""
            if seq.len ==1 :       
                seq.Ap_p = kp*seq.pop*(Nmono-1)
                seq.Ap_p1 = kp*seq.pop*sequences[1].pop # Propensity of adding 1
                seq.Ap_p0 = kp*seq.pop*sequences[0].pop # Propensity of adding 0
                seq.Ap_h = 0
                seq.Ap_r = 0
                seq.Ap_d = kd*seq.pop
            
            elif seq.len <= 6:
                seq.Ap_p = kp*seq.pop*(Nmono)
                seq.Ap_p1 = kp*seq.pop*sequences[1].pop # Propensity of adding 1
                seq.Ap_p0 = kp*seq.pop*sequences[0].pop # Propensity of adding 0
                seq.Ap_h = seq.Kh*seq.pop*(seq.len-1)
                seq.Ap_h1 = kh*seq.pop
                seq.Ap_r = 0
                seq.Ap_d = kd*seq.pop
          
            else:
                seq.Ap_p = kp*seq.pop*Nmono #propensity of polymerization is proportional to rate, the number of polymers, and the number of monomers
                seq.Ap_p1 = kp*seq.pop*sequences[1].pop # Propensity of adding 1
                seq.Ap_p0 = kp*seq.pop*sequences[0].pop # Propensity of adding 0
                seq.Ap_h = seq.Kh*(seq.len-1)*seq.pop # propensity of hydrolosis is proportional to rate, length of sequence and the number of polymers
                seq.Ap_h1 = seq.Kh*seq.pop

                resource_depedence = sequence_dependence(seq.seq_dep)
                seq.Ap_r = seq.Kr*seq.pop*resource_depedence
                seq.Ap_r0 = seq.Kr*seq.pop*(float(sequences[0].pop/(1+seq.con[0])))
                seq.Ap_r1 = seq.Kr*seq.pop*(float(sequences[1].pop/(1+seq.con[1])))
                seq.Ap_d = kd*seq.pop 

            Ap_p += seq.Ap_p
            Ap_h += seq.Ap_h
            Ap_r += seq.Ap_r
            Ap_d += seq.Ap_d   
            seq.TimePop_Long += tau_step*float(seq.pop)
            seq.TimePop += tau_step*float(seq.pop)
            seq.SumPop += seq.pop
            time_difference = (Parameters.tau - seq.t_discovery)
            if seq.stable == False and time_difference > 5.0:
                mean_population = seq.TimePop_Long/ (Parameters.tau - seq.t_discovery)
                if mean_population >= 1.0:
                    seq.stable =True
                else:
                    seq.t_discovery = Parameters.tau
        else:
            seq.Ap_p = 0.0
            seq.Ap_h = 0.0
            seq.Ap_r = 0.0
            seq.Ap_d = 0.0

        


    Parameters.Ap_p=Ap_p
    Parameters.Ap_h=Ap_h 
    Parameters.Ap_r=Ap_r 
    Parameters.Ap_d=Ap_d  
    Parameters.Atot=Ap_p+Ap_h + Ap_d +Ap_r    

    
################################################################################
def initialize_motifs():
    import  Parameters
    import binary_sequences
    if  Parameters.NonTrivial == True:
         Parameters.NT_motifs = binary_sequences.binary_primes(Parameters.NT_N,  Parameters.NT_len)
        
    if  Parameters.Functional == True : 
         Parameters.F_motifs = binary_sequences.generate_binaries(Parameters.F_N,  Parameters.F_len)

################################################################################
def sequence_dependence(bonds):
    import Parameters
    dependence = 0
    for bond in bonds:
        if bond == 0:
            dependence += Parameters.sequences[0].pop*Parameters.sequences[0].pop
        elif bond == 1:
            dependence += Parameters.sequences[1].pop*Parameters.sequences[0].pop
        elif bond == 2:
            dependence += Parameters.sequences[1].pop*Parameters.sequences[1].pop
    return dependence
