# This File contains all the class objects for simulating "The Race up Mount Improbable"
# copyright Sara Imari Walker 2011
# Edited By Cole Mathis 2014



import sys, os
from time import clock
from  Polymers import *
from  Initialize import * 
from  Reactions import *
from  Output import *
#import matplotlib.pylab as plt 

import numpy as np
import random
import math




def main():
    import Parameters
    for exp in range(1): # This allows the code to be run many times in a row, generating seperate outputs each time.
        if Parameters.master_control == False:
            originalDirectory = os.getcwd()
            if Parameters.monomer_flux != 0.0: 
                dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_kd%.4f_km%.2f_Mf%.4f/%i' % (Parameters.kp ,Parameters.kh, Parameters.kr, Parameters.kd, Parameters.km, Parameters.monomer_flux, exp))

                if not os.path.exists(originalDirectory+ dirname):
                    os.makedirs(originalDirectory+ dirname+ '/Landscapes')


            else: 
                dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (Parameters.kp ,Parameters.kh, Parameters.kr, Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))

                if not os.path.exists(originalDirectory+ dirname):
                    os.makedirs(originalDirectory+ dirname + '/Landscapes')

            newdir = originalDirectory +dirname
            os.chdir(newdir)


        start_time = clock()
        #Parameters.current_exp = exp    

        """Clear variables for each new experimental run"""

        del  Parameters.sequences[0:len( Parameters.sequences)]
        Parameters.seq_dict.clear()
        Parameters.sequences = []
        Parameters.tot_species = 0
        Parameters.Ntot = 0
        Parameters.Nmono= 0
    


        """Initialize random number generator for each run"""
        Parameters.run_seed = 100*Parameters.current_exp +  random.random()*Parameters.run_seed      
        random.seed( Parameters.run_seed)

       
        initialize(Parameters.sequences)
        initialize_motifs()
        update_propensities(Parameters.sequences, 0.0)
        
        
        # print  Parameters.F_motifs
        # print  Parameters.NT_motifs
        # print 'Intialization complete ...'
 
        

        Parameters.tau = 0.0
        tau_max =  Parameters.tau_max
        freq_counter = 0.0
        infreq_counter = 0.0

        monomer_flux_count = 0
     
       
        while Parameters.tau < tau_max:
            """Main time evolution loop"""
    
            """Choose Reaction"""
            reaction =[]
            dice_roll = random.random()*Parameters.Atot
            #print "Reaction dice_roll: " +repr(dice_roll)
            
            
            if(dice_roll <  Parameters.Ap_p):
                
                #print 'Initiating polymerization ...'
                reaction = polymerization(Parameters.sequences)
                Parameters.polymerization_events +=1
                last_reaction = "polymerization"
                
            # elif (dice_roll <  Parameters.Ap_p +  Parameters.Ap_d):
            #     reaction = death(Parameters.sequences)
            #     Parameters.death_events +=1
            #     last_reaction = 'death'
                
            elif (dice_roll <  Parameters.Ap_p +  Parameters.Ap_d +  Parameters.Ap_h):
                #print 'Initiating degradation ...'
                reaction = degradation(Parameters.sequences)
                Parameters.hydrolysis_events +=1
                last_reaction = 'Hydrolysis'
                
            elif (dice_roll <  Parameters.Ap_p +  Parameters.Ap_d +  Parameters.Ap_h +  Parameters.Ap_r):
                #print 'Initiating Replication ...'
                reaction = replication(Parameters.sequences)
                Parameters.replication_events +=1
                last_reaction = 'replication'
                  
            
            
           #  ############################################################
           # #"""Check Mass Conservation - If violated evolution loop will exit"""
           # #''' This is not useful if monomer_flux > 0 '''
            # num_conserv = 0

            mass = 0
   
            for seq in Parameters.sequences:
              
           
            #    num_conserv += seq.pop

               mass += seq.pop*seq.len
               
               
            # if  Parameters.Ntot != num_conserv:

            #    print 'Total particle number calculated incorrectly'#
            #    break
               
            if sum(Parameters.M_N) != mass:
               print 'Conservation of mass violated'
               print "Calculated Mass: " +repr(mass)
               print "Last event : "
               
               break
            
            ############################################################

        
            """ 3. Adjust Time """
            if  Parameters.Ntot == 1:
                print 'All mass has converged on a single polymer ... '
                break
            if  Parameters.Atot== 0:
                break

            # Random Time step forward.
            dice_roll = random.random()
            tau_step = -math.log(dice_roll)/ Parameters.Atot;
                
            """Increase monomer abundance """ #Contineous flow of monomers
            monomer_flux_count += Parameters.monomer_flux*tau_step
            while (monomer_flux_count >= 1):
                for ID in range(0, Parameters.m):
                     Parameters.sequences[ID].pop += 1
                     Parameters.Ntot +=1
                     Parameters.Nmono +=1
                monomer_flux_count -= 1
            print_all_pop(Parameters.current_exp, Parameters.tau, Parameters.sequences)
            update_propensities(Parameters.sequences, tau_step)

            Parameters.tau += tau_step            
            #Output data
            if freq_counter <= Parameters.tau:
                output_data(Parameters.current_exp,Parameters.tau,Parameters.sequences) 
                freq_counter+=Parameters.t_frequent

                if infreq_counter < Parameters.tau:
                    infreq_output(Parameters.current_exp,Parameters.tau, Parameters.sequences)
                    infreq_counter += Parameters.t_infrequent
                    #print tau
            #print Parameters.tau
            Parameters.tau += tau_step
            
        # Plot data
        print_sequences(Parameters.sequences, Parameters.current_exp)
        #plot_data(Parameters.current_exp,Parameters.tau)
        
        runtime = clock() - start_time
        
        # print 'polymerization_events: ' +repr(Parameters.polymerization_events)
        # print 'hydrolysis_events: ' + repr(Parameters.hydrolysis_events)
        # print 'Replication Events: ' +repr(Parameters.replication_events)
        # print 'Null Replication: ' +repr(Parameters.null_replication)
        # print 'Total Events: ' +repr(Parameters.polymerization_events + Parameters.hydrolysis_events + Parameters.replication_events)

        # print "Runtime: " +repr(runtime)
        # print('\a')
    
# End Function Main ############################################################



if __name__=="__main__":
    main()
    
