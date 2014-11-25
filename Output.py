#Copyright Cole Mathis 2014
import Parameters
import Polymers
import numpy as np
import os
#import matplotlib.pylab as plt 
################################################################################
def output_data(exp, tau, sequences):
    import Parameters

    output_monomers(exp, tau, sequences)
    num_species(exp, tau, sequences)
    output_events(exp, tau)
    replicative_mass(exp, tau, sequences)
    population_diversity(exp,tau,sequences)
    output_propensities(exp, tau, sequences)
    average_length(exp,tau,sequences)
    resource_flow(exp, tau, sequences)
    heterogenaity(exp, tau, sequences)
    particle_number(exp, tau, sequences)
    #average_population(exp, tau, sequences)
    
    #average_concentration(exp,tau,sequences,1)
    #average_concentration(exp,tau,sequences,0)

    #above_averagePop_concentration(exp, tau, sequences, 0)
    #above_averageLen_concentration(exp, tau, sequences, 1)
################################################################################
def infreq_output(exp,tau, sequences):
    
    average_sequence_diversity(exp,tau,sequences)
    
    print_length_distro(exp, tau, sequences)
    print_mass_distro(exp, tau, sequences)

    #output_landscape(sequences, exp, tau)
################################################################################
def particle_number(exp,tau, sequences):
    total_particles = 0.0
    num_replicators = 0.0
    for seq in sequences:
        if seq.pop != 0:
            total_particles += seq.pop
            if seq.len >= 7.0:
                num_replicators += seq.pop
    filename = ('%iparticle_count.dat' % (exp))
    if(tau == 0):
            file = open(filename, 'w')
    else:
            file = open(filename, 'a')
    s = str(tau)+ '     '+str(total_particles)+ '   '+str(num_replicators)
    file.write(s)
    file.write('\n')
    file.close()
################################################################################
def print_length_distro(exp, tau, sequences):
    '''Plot the time averaged monomer-space landscape'''
    import numpy as np
    
    lengths = []
    for seq in sequences:
        if seq.pop != 0:
            for i in range(1,seq.pop):
                lengths.append(seq.len)

    np.savetxt('Landscapes/%iLengths%.0f.dat' %(exp,tau), lengths)
################################################################################
def print_mass_distro(exp, tau, sequences):
    '''Plot the time averaged monomer-space landscape'''
    import numpy as np
    
    masses = []
    lengths =[]
    for seq in sequences:
        if seq.pop != 0:
            if seq.len in lengths:
                index = lengths.index(seq.len)
                masses[index] += seq.len*seq.pop
            else:
                lengths.append(seq.len)
                masses.append(seq.len*seq.pop)
            

    np.savetxt('Landscapes/%iMasses%.0f.dat' %(exp,tau), [lengths, masses])
################################################################################
def num_species(exp,tau, sequences):
    total_species = 0.0
    living_species = 0.0
    max_length = 0.0
    for seq in sequences:
        if seq.pop != 0:
            living_species +=1
        if seq.len > max_length:
            max_length =seq.len
    total_species = len(sequences)
    filename = ('%ispeciescount.dat' % (exp))
    if(tau == 0):
            file = open(filename, 'w')
    else:
            file = open(filename, 'a')
    s = str(tau)+ '     '+str(living_species) + '    '+ str(total_species)+ '   '+ str(max_length)
    file.write(s)
    file.write('\n')
    file.close()
################################################################################
def heterogenaity(exp,tau, sequences):
    total_pop = 0.0
    hetero = 0.0
    for seq in sequences:
        if seq.pop != 0:
            hetero += seq.pop*(float(seq.len - max(seq.con))/float(seq.len) )
            total_pop += seq.pop   

    filename = ('%iMean_Heterogenaity.dat' % (exp))
    if(tau == 0):
            file = open(filename, 'w')
    else:
            file = open(filename, 'a')
    hetero = float(hetero)/float(total_pop)
    s = str(tau)+ '     '+str(hetero)
    file.write(s)
    file.write('\n')
    file.close()
################################################################################
def resource_flow(exp,tau, sequences):

    filename = ('%iResourceFlow.dat' % (exp))
    if(tau == 0):
            file = open(filename, 'w')
    else:
            file = open(filename, 'a')
    s = str(tau)+ '     '+str(Parameters.polymerized_to_replicator) + '    '+ str(Parameters.replicated_mass)
    file.write(s)
    file.write('\n')
    file.close()

################################################################################      
def resource_distro(exp,tau, sequences):
    total_species = 0.0
    living_species = 0.0
    max_length = 0.0
    for seq in sequences:
        if seq.pop != 0:
            living_species +=1
        if seq.len > max_length:
            max_length =seq.len
    total_species = len(sequences)
    filename = ('%ispeciescount.dat' % (exp))
    if(tau == 0):
            file = open(filename, 'w')
    else:
            file = open(filename, 'a')
    s = str(tau)+ '     '+str(living_species) + '    '+ str(total_species)+ '   '+ str(max_length)
    file.write(s)
    file.write('\n')
    file.close()      
 ################################################################################
def replicative_mass(exp,tau, sequences):
    rep_mass = 0.0
    zero_mass = 0.0
    one_mass = 0.0
    for seq in sequences:
        if seq.len >= 7:
            rep_mass += seq.len*seq.pop 
            zero_mass += seq.con[0]*seq.pop 
            one_mass  += seq.con[1]*seq.pop
    filename = ('%iRep_Mass.dat' % (exp))
    if(tau == 0):
            file = open(filename, 'w')
    else:
            file = open(filename, 'a')
    s = str(tau)+ '     '+str(rep_mass) + '    '+ str(zero_mass) + '     '+ str(one_mass) 
    file.write(s)
    file.write('\n')
    file.close()   
#################################################################################
def output_events(exp, tau):
    import Parameters
    filename = ('%iEvents.dat' % (exp))
    if(tau == 0):
            file = open(filename, 'w')
    else:
            file = open(filename, 'a')
    s = str(tau)+ '     '+str(Parameters.polymerization_events) + '    '+ str(Parameters.hydrolysis_events) + '     '+ str(Parameters.replication_events) 
    file.write(s)
    file.write('\n')
    file.close()   

################################################################################
def output_propensities(exp, tau, sequences):
    import Parameters
    filename = ('%iPropensities.dat' % (exp))
    if(tau == 0):
            file = open(filename, 'w')
    else:
            file = open(filename, 'a')
    s = str(tau)+ '     '+str(Parameters.Ap_p) + '    '+ str(Parameters.Ap_h) + '     '+ str(Parameters.Ap_r) 
    file.write(s)
    file.write('\n')
    file.close()
################################################################################
def output_monomers(exp, t, sequences):
    """Print time-dependent monomer concentrations"""

    import Parameters

    for ID in range(0,Parameters.m):

        filename = ('%i_monomer_%i.dat' % ( exp, ID))
    
        if(t == 0):
            file = open(filename, 'w')
        else:
            file = open(filename, 'a')
            
        s = str(t) + '      ' + str(sequences[ID].pop)
        file.write(s)
        file.write('\n')
        file.close()
################################################################################
def print_all_pop(exp, t, sequences):
    """Print time-dependent polymer concentrations"""
    import Parameters
    dirname = ('%iPopulations' %exp)
    if not os.path.exists(dirname):
            os.makedirs(dirname, mode =0777)

    for seq in sequences:

        filename = ('%iPopulations/%s.dat' % (exp, seq.seq))
        
        if(t == 0):
            file = open(filename, 'w')
        else:
            file = open(filename, 'a')
            
        s= str(t) +'      ' + str(seq.pop)
        file.write(s)
        file.write('\n')
        file.close()
################################################################################
def population_diversity(exp, t, sequences):
    """Print time-dependent Shannon index diversity"""
    import math
    import Parameters


    H = 0.0
    if Parameters.Ntot != 0:
        for seq in sequences:
            if (seq.pop != 0):
                p_i = float(seq.pop)/float(Parameters.Ntot)
                H -= p_i * math.log(p_i,2)
        filename = ('%i_PopDiversity.dat' % (exp))
        
        if(t == 0):
            file = open(filename, 'w')
        else:
            file = open(filename, 'a')
                
        s = str(t) + '      ' + str(float(H))
        file.write(s)
        file.write('\n')
        file.close()
################################################################################
def average_sequence_diversity(exp,t, sequences):
    import math
    import Parameters
    aveH = 0.0
    total_population = 0.0
    if Parameters.Ntot != 0:
        for seq in sequences:
            total_population += seq.pop
            p1 = float(seq.con[1])/float(seq.len)
            p0 = float(seq.con[0])/float(seq.len)
            if p1 != 0:
                aveH += float(seq.con[1])*p1*math.log(p1,2)*seq.pop
            if p0 != 0:
                aveH += float(seq.con[0])*p0*math.log(p0,2)*seq.pop
        aveH = -aveH/float(total_population)
        filename = ('%i_AveSeqDiversity.dat' % (exp))
        
        if(t == 0):
            file = open(filename, 'w')
        else:
            file = open(filename, 'a')
                
        s = str(t) + '      ' + str(float(aveH))
        file.write(s)
        file.write('\n')
        file.close()
################################################################################
def average_length(exp, t, sequences):
    import math
    import Parameters

    ave_len=00.0
    total_population = 0.0
    stdev = 0.0
    sigma = 0.0
    if Parameters.Ntot != 0:
        for seq in sequences:
            ave_len += seq.pop*seq.len
            total_population += seq.pop
        
    ave_len = float(ave_len)/float(total_population)
    if Parameters.Ntot != 0:
        for seq in sequences:
            sigma += seq.pop*((seq.len - ave_len)**2)
    sigma = sigma/total_population
    stdev = math.sqrt(sigma)
    filename = ('%i_aveLen.dat' % (exp))
        
    if(t == 0):
        file = open(filename, 'w')
    else:
        file = open(filename, 'a')
                
    s = str(t) + '      ' + str(ave_len) + '    ' + str(stdev)
    file.write(s)
    file.write('\n')
    file.close()
################################################################################
def average_population(exp, t, sequences):
    population = 0.0
    species = 0.0
    ave_pop = 0.0

    for seq in sequences:
        if seq.pop > 0 and seq.len != 1:
            population += seq.pop
            species += 1
    ave_pop = float(population)/float(species)
    filename = ('%i_avePop.dat' % (exp))
        
    if(t == 0):
        file = open(filename, 'w')
    else:
        file = open(filename, 'a')
                
    s = str(t) + '      ' + str(float(ave_pop))
    file.write(s)
    file.write('\n')
    file.close()
################################################################################
def output_landscape(sequences,exp, tau):
    '''Plot the time averaged monomer-space landscape'''
    import numpy as np
    #import matplotlib.pyplot as plt
    ones =[]
    zeros=[]
    AvePop  =[]
    #AveTime =[]
    for seq in sequences:
            ones.append(seq.con[1])
            zeros.append(seq.con[0])
            AvePop.append(float(seq.TimePop/tau))
            #AveTime.append(float(seq.TimePop_Long/seq.SumPop))
            seq.TimePop = 0.0
    for i in range(0,len(ones)-1):
        for j in range((i+1),len(ones)-1):
            if (ones[i]==ones[j]) and (zeros[i]==zeros[j]):
                AvePop[i] += AvePop[j]
                AvePop[j] = 0.0
                #AveTime[i] += AveTime[j]
                #AveTime[j] = 0.0
                ones[j] = 0.0
                zeros[j] = 0.0
    # fig= plt.figure(figsize=(12,12), dpi = 200)
    # ax1 =fig.add_subplot(111)
    # ax1.scatter(ones, zeros, c=AveTime, s=AvePop)
    # x0,x1 = ax1.get_xlim()
    # y0,y1 = ax1.get_ylim()
    # plt.xlabel('Stability Monomers')
    # plt.ylabel('Replication Monomers')
    # if x1>= y1:
    #     plt.axis([-1,x1, -1,x1])
    # else: 
    #     plt.axis([-1,y1, -1,y1])

    #plt.savefig('Landscape.png')
    np.savetxt('Landscapes/%iLandscape%.0f.dat' %(exp,tau) ,[zeros, ones, AvePop] )
    #return [zeros, ones, AvePop]
# ################################################################################
# def average_concentration(exp,t,sequences, monomer):
#     import math
#     import Parameters
#     ave_con = 0.0 
#     total_population = 0.0
#     for seq in sequences:
#         if seq.pop != 0:
#             ave_con += float(seq.con[monomer]*seq.pop)/float(seq.len) 
#             total_population += seq.pop
        
#     ave_con = ave_con/float(total_population)

#     filename = ('%i_aveCon%i.dat' % (exp, monomer))
        
#     if(t == 0):
#         file = open(filename, 'w')
#     else:
#         file = open(filename, 'a')
                
#     s = str(t) + '      ' + str(float(ave_con))
#     file.write(s)
#     file.write('\n')
#     file.close()
# ################################################################################
# def average_concentration_length1(exp, t, sequences, monomer):
#     import math
#     import Parameters
#     ave_con = 0.0 
#     total_population = 0.0
#     for seq in sequences:
#         if seq.pop != 0:
#             ave_con += float(seq.con[monomer]*seq.pop)/float(seq.len) 
        
#     ave_con = ave_con/float(total_population)

#     filename = ('%i_aveConL%i.dat' % (exp, monomer))
        
#     if(t == 0):
#         file = open(filename, 'w')
#     else:
#         file = open(filename, 'a')
                
#     s = str(t) + '      ' + str(float(ave_con))
#     file.write(s)
#     file.write('\n')
#     file.close()
# ################################################################################
# def average_concentration_pop0(exp, t, sequences, monomer):
#     import math
#     import Parameters
#     ave_con = 0.0 
    
#     for seq in sequences:
#         if seq.pop != 0:
#             ave_con += float(seq.con[monomer]*seq.pop)/float(seq.len) 
        
#     ave_con = ave_con/float(Parameters.Ntot)

#     filename = ('%s/%i_aveConP%i.dat' % (Parameters.dirname, exp, monomer))
        
#     if(t == 0):
#         file = open(filename, 'w')
#     else:
#         file = open(filename, 'a')
                
#     s = str(t) + '      ' + str(float(ave_con))
#     file.write(s)
#     file.write('\n')
#     file.close()
# ################################################################################
# def above_averageLen_concentration(exp, t, sequences, monomer):
#     ave_con = 0.0 
#     total_population = 0.0 
#     ave_len = 0.0
#     large_population = 0.0
#     for seq in sequences:
#         ave_len += seq.len*seq.pop
#         total_population += seq.pop
#     ave_len = float(ave_len)/float(total_population)
#     for seq in sequences:
#         if seq.len > ave_len:
#             ave_con += float(seq.con[monomer]*seq.pop)/float(seq.len) 
#             large_population += seq.pop
        
#     ave_con = ave_con/float(large_population)
    
#     filename = ('%s/%i_AaveConL%i.dat' % (Parameters.dirname, exp, monomer))
        
#     if(t == 0):
#         file = open(filename, 'w')
#     else:
#         file = open(filename, 'a')
                
#     s = str(t) + '      ' + str(float(ave_con))
#     file.write(s)
#     file.write('\n')
#     file.close() 
# ################################################################################
# def above_averagePop_concentration(exp, t, sequences, monomer):
#     ave_con = 0.0 
#     total_population = 0.0 
#     ave_pop = 0.0
#     large_population = 0.0
#     for seq in sequences:
#         ave_pop += seq.pop
#         total_population += seq.pop
#     ave_pop = float(ave_pop)/float(len(sequences))
#     for seq in sequences:
#         if seq.pop > ave_pop:
#             ave_con += float(seq.con[monomer]*seq.pop)/float(seq.len)
#             large_population += seq.pop
        
#     ave_con = ave_con/float(large_population)
    
#     filename = ('%s/%i_AaveConP%i.dat' % (Parameters.dirname, exp, monomer))
        
#     if(t == 0):
#         file = open(filename, 'w')
#     else:
#         file = open(filename, 'a')
                
#     s = str(t) + '      ' + str(float(ave_con))
#     file.write(s)
#     file.write('\n')
#     file.close()      
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
def print_sequences(sequences, exp):
# This function takes the LIST sequences as its argument, it will agrrange the 
# name, length and population of each sequence into an array and print that arr-
# ay to a file
#This function is intended to be used after the simulation is completed.
    filename = "%isequencedata.txt" % exp
    fout=open(filename, 'w')
    fout.write('%-15s  %-15s %-15s %s \n' %('ID', 'Sequence', 'Discovery Time', 'length'))
    for ID in range(0,len(sequences)):
            fout.write('%-15s %-15s  %-15s %s\n' % (ID, sequences[ID].seq, sequences[ID].t_discovery, sequences[ID].len)) 
            
    fout.close()             
# ################################################################################
# def print_NT_sequences(sequences):
    
#     fout=open("data\NT_sequences.txt", 'w')
#     fout.write('%-15s  %-15s %-15s %s %s \n' %('ID', 'Sequence', 'Population', 'NT motifs', Parameters.NT_motifs))
    
#     for seq in sequences:
#         if seq.NT == True:
#             fout.write('%-15s %-15s %-15s  %s \n' %(seq.ID, seq.seq, seq.pop, seq.NT_motifs))
            
#     fout.close()  
# ################################################################################
# ################################################################################
# def plot_monomers(exp):
#     fig= plt.figure()
#     t,zeros = np.loadtxt(Parameters.dirname+('/%i_monomer_0.dat' % (exp)), unpack = True)
#     t,ones = np.loadtxt(Parameters.dirname+('/%i_monomer_1.dat' % (exp)), unpack = True)
#     ax2= fig.add_subplot(111)
#     ax2.scatter(t, zeros, c='b', marker = 's', label ='zeros')
#     ax2.scatter(t, ones, c='r', marker = 'o', label ='ones')
#     x0,x1 = ax2.get_xlim()
#     y0,y1 = ax2.get_ylim()
#     plt.legend(loc='upper right')
#     plt.xlabel('time')
#     plt.ylabel('Monomers')
#     plt.axis([0,x1, 0,y1])
#     s= Parameters.dirname[5:]+'M.png'
#     plt.savefig(s)
################################################################################
def plotxy(datafile, x_label, y_label, title):
    import numpy as np
    #import matplotlib.pyplot as plt

    plt.plotfile(datafile, delimiter=' ', cols=(0, 1), 
             names=(x_label, y_label), marker='o')
    s= title+'.png'
    plt.savefig(s)
################################################################################
def plot_data(exp,tau):
    import numpy as np
    #import matplotlib.pyplot as plt

    Landscapes=output_landscape(Parameters.sequences, Parameters.current_exp, tau)

    #plotxy(('%s/%i_aveCon1.dat' % (Parameters.dirname, exp)),'time','concentration', '1concentration')
    #plotxy(('%s/%i_aveCon0.dat' % (Parameters.dirname, exp)),'time','concentration', '0concentration')
    #plotxy(('%s/%i_AaveConL1.dat' % (Parameters.dirname, exp)),'time','concentration', 'AALenConcentration1')
    #plotxy(('%s/%i_AaveConP0.dat' % (Parameters.dirname, exp)),'time','concentration', 'AAPopConcentration0')
    plotxy(('%i_monomer_0.dat' % (exp)), 'time','0s',('%iOvsTime' % exp))
    plotxy(('%i_monomer_1.dat' % (exp)), 'time','1s',('%i1vsTime' % exp))
    # plotxy(('%i_AveSeqDiversity.dat' % ( exp)),'time','Entropy', 'SequneceDiversity')
    plotxy(('%i_PopDiversity.dat' % ( exp)),'time','Entropy', ('%iPopulation Diversity' % exp))
    plotxy(('%i_aveLen.dat' % ( exp)),'time','Average Length', ('%iAveLength' % exp))
    #plotxy(('%s/%i_avePop.dat' % (Parameters.dirname, exp)),'time','Average Pop', 'AvePop')

    # # This will make a .png with 4 plots, the monomer-space landscape, Monomers vs T, Average Length vs time and Diversity vs Time
    # fig= plt.figure(figsize=(12,12), dpi = 200)

    # ax1 =fig.add_subplot(221)
    # ax1.scatter(Landscapes[1], Landscapes[0],c=Landscapes[2], s=Landscapes[3])
    # x0,x1 = ax1.get_xlim()
    # y0,y1 = ax1.get_ylim()
    # plt.xlabel('Stability Monomers')
    # plt.ylabel('Replication Monomers')
    # if x1>= y1:
    #     plt.axis([-1,x1, -1,x1])
    # else: 
    #     plt.axis([-1,y1, -1,y1])
    

    # t,zeros = np.loadtxt(('%i_monomer_0.dat' % (exp)), unpack = True)
    # t,ones = np.loadtxt(('%i_monomer_1.dat' % (exp)), unpack = True)
    # ax2= fig.add_subplot(222)
    # ax2.scatter(t, zeros, c='b', marker = 's', label ='zeros')
    # ax2.scatter(t, ones, c='r', marker = 'o', label ='ones')
    # x0,x1 = ax2.get_xlim()
    # y0,y1 = ax2.get_ylim()
    # plt.legend(loc='upper right')
    # plt.xlabel('time')
    # plt.ylabel('Monomers')
    # plt.axis([0,x1, 0,y1])
    

    # t,Ap_p, Ap_h, Ap_r = np.loadtxt(('%iPropensities.dat' %exp), unpack= True)
    # ax3 = fig.add_subplot(223)
    # ax3.scatter(t, Ap_p, c='b' , label ='Ap_p')
    # ax3.scatter(t,Ap_h, c='r', label ='Ap_h')
    # ax3.scatter(t,Ap_r, c = 'g', label= 'Ap_r')
    # plt.legend(loc= 'upper right')
    # plt.xlabel('time')
    # plt.ylabel('Propensities')

    # # t,length, stdev = np.loadtxt(('%i_aveLen.dat' % (exp)), unpack = True)
    # # ax3= fig.add_subplot(223)
    # # ax3.scatter(t, length)
    # # x0,x1 = ax3.get_xlim()
    # # y0,y1 = ax3.get_ylim()
    # # plt.xlabel('time')
    # # plt.ylabel('Average Length')
    # # plt.axis([0,x1, 0,y1])
    

    # t,diversity = np.loadtxt(('%i_PopDiversity.dat' % (exp)), unpack = True)
    # ax4= fig.add_subplot(224)
    # ax4.scatter(t, diversity)
    # x0,x1 = ax4.get_xlim()
    # y0,y1 = ax4.get_ylim()
    # plt.xlabel('time')
    # plt.ylabel('Population Diversity (Entropy)')
    # plt.axis([0,x1, 0,y1])

    # plt.savefig(('%ikp%.4f_kh%.4f_kr%.4f_km%.2f_M%.4f.png' % (exp, Parameters.kp, Parameters.kh, Parameters.kr, Parameters.km, sum(Parameters.M_N))))

    
    # plt.savefig('%iPropensities.png' %exp)

   