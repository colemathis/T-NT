#!/usr/bin/python
#########################################################################################################
def probability_distributions_sets():
    import numpy as np
    import random
    import os
    import Parameters

    from time import clock
    start_time = clock()

    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [1.0]
    KR = [0.005]


    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):

                """ Generate Probability distros """
                CompleteResource = []
                CompleteRep = []
                CompleteJoint =[]
                for exp in range(2):
                    os.chdir(originalDirectory)
                    dirname = ('/data10.6/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)

                    t,ones = np.loadtxt('%i_monomer_1.dat'%exp, unpack = True, usecols = (0,1))
                    t,zeros = np.loadtxt('%i_monomer_0.dat'%exp, unpack = True, usecols = (0,1))
                    monomers =np.add(ones, zeros)
                    resource_ratio = np.divide(ones, monomers)

                    rep_mass, rep_ones = np.loadtxt('%iRep_Mass.dat'%exp, unpack = True, usecols = (1,3))

                    rep_ratio = []
                    for n in range(len(rep_mass)):
                        if rep_mass[n]== 0:
                            rep_mass[n] = 1.0
                    rep_ratio = np.divide(rep_ones, rep_mass)
                    CompleteJoint.extend(zip(resource_ratio, rep_ratio))
                    CompleteRep.extend(rep_ratio)
                    CompleteResource.extend(resource_ratio)
                print "data collected"
                os.chdir(originalDirectory)
                dirname = ('/data10.6/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                os.chdir(originalDirectory+dirname)
                
                # CompleteResource = list(np.around(CompleteResource))
                # CompleteRep = list(np.around(CompleteRep, decimals =4))
                
                Joint = zip(CompleteResource,CompleteRep)
                Set_Resources = set(CompleteResource)
                Set_Rep  = set(CompleteRep)
                Set_Joint = set(Joint)

                Resource_Probability = {}
                Rep_Probability = {}
                Joint_Probability = {}
                print 'Generating Resource Distro'
                length = len(CompleteResource)

                for resource in Set_Resources:
                    count = 0
                    i = 0
                    while i < (len(CompleteResource)):
                        if resource == CompleteResource[i]:
                            count +=1
                            del(CompleteResource[i])
                        else:
                            i +=1
                    Probability = float(count)/float(length)

                    Resource_Probability[resource] = Probability

                print "Generating Rep Distro"
                length = len(CompleteRep)
                for rep in Set_Rep:
                    count = 0
                    i = 0
                    while i <(len(CompleteRep)):
                        if rep == CompleteRep[i]:
                            count +=1
                            del(CompleteRep[i])
                        else:
                            i +=1
                    Probability = float(count)/float(length)

                    Rep_Probability[rep] = Probability
                
                print "Generating Joint Distro"
                length = len(Joint)
                for condition in Set_Joint:
                    count = 0
                    i = 0
                    while i < (len(Joint)):
                        if condition == Joint[i]:
                            count +=1
                            del(Joint[i])
                        else:
                            i +=1
                    Probability = float(count)/float(length)

                    Joint_Probability[condition] = Probability
                # print "Saving to File"
                # filename = 'Resource_Probabilities.dat'
                # file = open(filename, 'w')
                # for resource in Set_Resources:
                #     s= str(resource)+'  '+str(Resource_Probability[resource])
                #     file.write(s)
                #     file.write('\n')
                # file.close

                # filename = 'Rep_Probabilities.dat'
                # file = open(filename, 'w')
                # for Rep in Set_Rep:
                #     s= str(Rep)+'  '+str(Rep_Probability[Rep])
                #     file.write(s)
                #     file.write('\n')
                # file.close

                # filename = 'Joint_Probabilities.dat'
                # file = open(filename, 'w')
                # for condition in Set_Joint:
                #     s= str(condition[0])+'  '+str(condition[1])+'  '+str(Joint_Probability[condition])
                #     file.write(s)
                #     file.write('\n')
                # file.close

                os.chdir(originalDirectory)
    runtime = clock() - start_time
    print runtime
#########################################################################################################
def probability_distributions():
    import numpy as np
    import random
    import os
    import Parameters

    from time import clock
    start_time = clock()
    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [0.001, 0.005, 0.009]
    KR = [0.005,0.05]
    
    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):

                """ Generate Probability distros """

                Resource_Probability = {}
                Rep_Probability = {}
                Joint_Probability = {}
                length = 0
                for exp in range(100):
                    os.chdir(originalDirectory)
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)


                    t,ones = np.loadtxt('%i_monomer_1.dat'%0, unpack = True, usecols = (0,1))
                    t,zeros = np.loadtxt('%i_monomer_0.dat'%0, unpack = True, usecols = (0,1))
                    monomers =np.add(ones, zeros)
                    rep_mass, rep_ones = np.loadtxt('%iRep_Mass.dat'%0, unpack = True, usecols = (1,3))
                    print exp
                    resource_ratio = np.divide(ones, monomers)
                    
                    rep_ratio = []
                    for n in range(len(rep_mass)):
                        if rep_mass[n]== 0:
                            rep_mass[n] = 1.0
                    rep_ratio = np.divide(rep_ones, rep_mass)

                    Joint  = zip(resource_ratio, rep_ratio)

                    for resource in resource_ratio:
                        if resource not in Resource_Probability:
                            Resource_Probability[resource] = float(1.0)
                        else:
                            Resource_Probability[resource] += float(1.0)

                    for rep in rep_ratio:
                        if rep not in Rep_Probability:
                            Rep_Probability[rep] = float(1.0)
                        else:
                            Rep_Probability[rep] += float(1.0)
                    
                    for condition in Joint:
                        if condition not in Joint_Probability:
                            Joint_Probability[condition] = float(1.0)
                        else:
                            Joint_Probability[condition] += float(1.0)
                    length += len(Joint)
                    print (2**30) - len(Resource_Probability)
                    print (2**30) - len(Rep_Probability)
                    print (2**30) - len(Joint_Probability)
                print length
                for condition in Joint_Probability:
                    Joint_Probability[condition] /= float(length)
                for Rep in Rep_Probability:
                    Rep_Probability[Rep] /= float(length)
                for resource in Resource_Probability:
                    Resource_Probability[resource] /= float(length)


                print "data collected"
                os.chdir(originalDirectory)
                dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                os.chdir(originalDirectory+dirname)
                
                # CompleteResource = list(np.around(CompleteResource))
                # CompleteRep = list(np.around(CompleteRep, decimals =4))
                
                # Set_Resources = set(CompleteResource)
                # Set_Rep  = set(CompleteRep)
                # Set_Joint = set(Joint)

                
               
                print "Saving to File"
                filename = 'Resource_Probabilities.dat'
                file = open(filename, 'w')
                for resource in Resource_Probability:
                    s= str(resource)+'  '+str(Resource_Probability[resource])
                    file.write(s)
                    file.write('\n')
                file.close

                filename = 'Rep_Probabilities.dat'
                file = open(filename, 'w')
                for Rep in Rep_Probability:
                    s= str(Rep)+'  '+str(Rep_Probability[Rep])
                    file.write(s)
                    file.write('\n')
                file.close

                filename = 'Joint_Probabilities.dat'
                file = open(filename, 'w')
                for condition in Joint_Probability:
                    s= str(condition[0])+'  '+str(condition[1])+'  '+str(Joint_Probability[condition])
                    file.write(s)
                    file.write('\n')
                file.close

                os.chdir(originalDirectory)

    runtime = clock() - start_time
    print runtime
#########################################################################################################                     
def mutualInfoTimeSeries():
    import numpy as np
    import random
    import os
    import Parameters
    import matplotlib.pylab as plt


    
    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [1.0, 0.75, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    KR = [0.005]


    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):
                
                os.chdir(originalDirectory)
                dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                os.chdir(originalDirectory+dirname)

                Rep_values, Rep_probability = np.loadtxt('Rep_Probabilities.dat',  unpack = True, dtype = float)
                Resource_values, Resource_probability = np.loadtxt('Resource_Probabilities.dat',  unpack = True, dtype = float)
                Joint1_values, Joint2_values, Joint_probability = np.loadtxt('Joint_Probabilities.dat', unpack = True, dtype = float)
                # Rep_values = np.around(Rep_values, decimals = 5)
                # Rep_probability = np.around(Rep_probability, decimals = 5)
                # Resource_values = np.around(Resource_values, decimals = 5)
                # Resource_probability = np.around(Resource_probability, decimals = 5)
                # Joint1_values = np.around(Joint1_values, decimals = 5)
                # Joint2_values = np.around(Joint2_values, decimals = 5)
                # Joint_probability = np.around(Joint_probability, decimals = 5)
                print "data Loaded"
                rep_distro = {}
                resource_distro ={}
                joint_distro = {}
                

                for n in range(len(Rep_values)-1):
                    rep_distro[Rep_values[n]] = Rep_probability[n]
                for n in range(len(Resource_values)-1):
                    resource_distro[Resource_values[n]] = Resource_probability[n]
                for n in range(len(Joint1_values)-1):
                    joint_distro[(Joint1_values[n], Joint2_values[n])] = Joint_probability[n]


                print "Distributions prepared"

                for exp in range(100):                
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)

                    t, zeros = np.loadtxt('%i_monomer_0.dat' %0, unpack = True, dtype = float)
                    t, ones = np.loadtxt('%i_monomer_1.dat' %0, unpack = True, dtype = float)

                    t, Rep, Rzeros, Rones = np.loadtxt('%iRep_Mass.dat' %0, unpack = True, dtype = float)

                    monomers = np.add(zeros, ones)
                    for m in range(len(Rep)):
                        if Rep[m] == 0:
                            Rep[m] = 1.0
                    resource_ratio = np.divide(ones, monomers)
                    rep_ratio = np.divide(Rones, Rep)
                    #print rep_ratio, rep_distro
                    window_size = 100
                    mutualInfo = []
                    for q in range(len(rep_ratio)-window_size-1):
                        mutual = 0
                        mutual = CalcMutualInfo(resource_ratio[q:q+window_size], rep_ratio[q:q+window_size],resource_distro, rep_distro, joint_distro) 
                        #mutual =  entropy(rep_ratio[i:i+window_size], rep_distro)#- jointEntropy(resource_ratio[i:i+window_size], rep_ratio[i:i+window_size], joint_distro)
                        mutualInfo.append(mutual)
                    plt.plot(mutualInfo)
                    plt.savefig('MItimeSeries.png')
                    plt.close()
                    os.chdir(originalDirectory)

                    #entropy(resource_ratio[i:i+window_size], resource_distro) +
#########################################################################################################
def probability_distributions_ints():
    import numpy as np
    import random
    import os
    import Parameters

    from time import clock
    start_time = clock()
    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [0.001, 0.005, 0.009]
    KR = [0.005,0.05]
    

    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):

                """ Generate Probability distros """

                Resource_Probability = {}
                Rep_Probability = {}
                Joint_Probability = {}
                length = 0
                for exp in range(100):
                    ones = []
                    zeros = []
                    rep_zeros= []
                    rep_ones = []
                    replicatos = []
                    resources = []
                    os.chdir(originalDirectory)
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)


                    t,ones = np.loadtxt('%i_monomer_1.dat'%0, unpack = True, usecols = (0,1))
                    t,zeros = np.loadtxt('%i_monomer_0.dat'%0, unpack = True, usecols = (0,1))
                    
                    rep_zeros, rep_ones = np.loadtxt('%iRep_Mass.dat'%0, unpack = True, usecols = (2,3))
                    resources = zip(zeros, ones)
                    replicators = zip(rep_zeros, rep_ones)
                    print exp
                    
                    Joint  = zip(zeros, ones, rep_zeros, rep_ones)

                    for resource in resources:
                        if resource not in Resource_Probability:
                            Resource_Probability[resource] = float(1.0)
                        else:
                            Resource_Probability[resource] += float(1.0)

                    for rep in replicators:
                        if rep not in Rep_Probability:
                            Rep_Probability[rep] = float(1.0)
                        else:
                            Rep_Probability[rep] += float(1.0)
                    
                    for condition in Joint:
                        if condition not in Joint_Probability:
                            Joint_Probability[condition] = float(1.0)
                        else:
                            Joint_Probability[condition] += float(1.0)
                    length += len(Joint)
                    
                print length
                for condition in Joint_Probability:
                    Joint_Probability[condition] /= float(length)
                for Rep in Rep_Probability:
                    Rep_Probability[Rep] /= float(length)
                for resource in Resource_Probability:
                    Resource_Probability[resource] /= float(length)


                print "data collected"
                os.chdir(originalDirectory)
                dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                os.chdir(originalDirectory+dirname)
                
                # CompleteResource = list(np.around(CompleteResource))
                # CompleteRep = list(np.around(CompleteRep, decimals =4))
                
                # Set_Resources = set(CompleteResource)
                # Set_Rep  = set(CompleteRep)
                # Set_Joint = set(Joint)

                
               
                print "Saving to File"
                filename = 'Resource_Probabilities.dat'
                file = open(filename, 'w')
                for resource in Resource_Probability:
                    s= str(resource[0])+'  '+str(resource[1])+'  '+str(Resource_Probability[resource])
                    file.write(s)
                    file.write('\n')
                file.close

                filename = 'Rep_Probabilities.dat'
                file = open(filename, 'w')
                for Rep in Rep_Probability:
                    s= str(Rep[0])+'  '+str(Rep[1])+'  '+str(Rep_Probability[Rep])
                    file.write(s)
                    file.write('\n')
                file.close

                filename = 'Joint_Probabilities.dat'
                file = open(filename, 'w')
                for condition in Joint_Probability:
                    s= str(condition[0])+ '    ' +str(condition[1])+ '    '+str(condition[2])+ '    '+str(condition[3])+ '    '+str(Joint_Probability[condition])
                    file.write(s)
                    file.write('\n')
                file.close

                os.chdir(originalDirectory)

    runtime = clock() - start_time
    print runtime
    print len(Resource_Probability)
#########################################################################################################                     
def mutualInfoTimeSeries_ints():
    import numpy as np
    import random
    import os
    import Parameters
    import matplotlib.pylab as plt


    
    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [0.001, 0.005, 0.009]
    KR = [0.005,0.05]
    
    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):
                
                os.chdir(originalDirectory)
                dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                os.chdir(originalDirectory+dirname)

                Rep_values1, Rep_values2, Rep_Probabilities = np.loadtxt('Rep_Probabilities.dat', usecols =(0,1,2),  unpack = True, dtype = float)
                #Rep_Probabilities = np.genfromtxt('Rep_Probabilities.dat', usecols =(2),  unpack = True, dtype = float)
                Resource_values1, Resource_values2, Resource_Probabilities = np.loadtxt('Resource_Probabilities.dat', usecols = (0,1,2),  unpack = True, dtype = float)
                #Resource_Probabilities = np.genfromtxt('Resource_Probabilities.dat', usecols= (2), unpack = True, dtype = float)
                Joint_values1, Joint_values2, Joint_values3, Joint_values4, Joint_Probabilities = np.loadtxt('Joint_Probabilities.dat', usecols= (0,1,2,3,4), unpack = True, dtype = float)
                #Joint_Probabilities = np.genfromtxt('Joint_Probabilities.dat', usecols = (4), unpack = True, dtype = float)
                Rep_values = zip(Rep_values1, Rep_values2)
                Resource_values = zip(Resource_values1, Resource_values2)
                Joint_values = zip(Joint_values1, Joint_values2, Joint_values3, Joint_values4)
                
                #print type(Joint_values), type(Joint_values[1])
                # Rep_values = np.around(Rep_values, decimals = 5)
                # Rep_probability = np.around(Rep_probability, decimals = 5)
                # Resource_values = np.around(Resource_values, decimals = 5)
                # Resource_probability = np.around(Resource_probability, decimals = 5)
                # Joint1_values = np.around(Joint1_values, decimals = 5)
                # Joint2_values = np.around(Joint2_values, decimals = 5)
                # Joint_probability = np.around(Joint_probability, decimals = 5)
                print "data Loaded"
                

                rep_distro = {}
                resource_distro ={}
                joint_distro = {}
                

                for n in range(len(Rep_values)):
                    rep_distro[Rep_values[n]] = Rep_Probabilities[n]
                for n in range(len(Resource_values)):
                    resource_distro[Resource_values[n]] = Resource_Probabilities[n]
                for n in range(len(Joint_values)):
                    joint_distro[Joint_values[n]] = Joint_Probabilities[n]
                print len(Resource_Probabilities)
                


                print "Distributions prepared"
                
                for exp in range(100):
                    print exp
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)

                    t,ones = np.loadtxt('%i_monomer_1.dat'%0, unpack = True, usecols = (0,1))
                    t,zeros = np.loadtxt('%i_monomer_0.dat'%0, unpack = True, usecols = (0,1))
                    rep_zeros, rep_ones = np.loadtxt('%iRep_Mass.dat'%0, unpack = True, usecols = (2,3))
                    
                    resources = zip(zeros, ones)
                    replicators = zip(rep_zeros, rep_ones)

                    print exp
                    
                    #print rep_ratio, rep_distro
                    window_size = 100
                    mutualInfo = []
                    for q in range(len(replicators)-window_size):
                        mutual = 0
                        mutual = CalcMutualInfo(resources[q:q+window_size], replicators[q:q+window_size],resource_distro, rep_distro, joint_distro) 
                        #mutual =  entropy(rep_ratio[i:i+window_size], rep_distro)#- jointEntropy(resource_ratio[i:i+window_size], rep_ratio[i:i+window_size], joint_distro)
                        mutualInfo.append(mutual)
                    plt.plot(mutualInfo)
                    np.savetxt('%iMutualInfo.dat' %exp, mutualInfo)
                    plt.savefig('MItimeSeries.png')
                    plt.close()
                    os.chdir(originalDirectory)

                    #entropy(resource_ratio[i:i+window_size], resource_distro) +
#########################################################################################################
def EA_Mutual_Info():
    import numpy as np
    import random
    import os
    import Parameters
    import matplotlib.pylab as plt


    
    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [0.001, 0.005, 0.009]
    KR = [0.005,0.05]
    


    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):
                
                os.chdir(originalDirectory)
                dirname1 = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                os.chdir(originalDirectory+dirname1)
                mean_Mutual_Info = []

                for exp in range(100):
                    print exp
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)

                    mutuals = np.loadtxt('%iMutualInfo.dat' %exp)
                    print len(mutuals)
                    if exp == 0:
                        print exp
                        mean_Mutual_Info.extend(mutuals)
                        print "This should be the first file loaded, do the numbers below equal each other?"
                        print len(mean_Mutual_Info), len(mutuals)
                        
                    else:

                        mean_Mutual_Info =np.add(mean_Mutual_Info[0:len(mutuals)], mutuals[0:len(mean_Mutual_Info)])
                mean_Mutual_Info = np.divide(mean_Mutual_Info, 100.0)

                os.chdir(originalDirectory+dirname1)
                plt.plot(mean_Mutual_Info)
                plt.show()
                plt.plot(mean_Mutual_Info)
                plt.savefig('EA_Mutual_Info.png')
                plt.close()
                np.savetxt('EA_Mutual_Info.dat', mean_Mutual_Info)
                os.chdir(originalDirectory)
#########################################################################################################################
def determine_transition_times():
    import numpy as np
    import random
    import os
    import Parameters
    import matplotlib.pylab as plt


    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [1.0]
    KR = [0.005]


    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):               
                os.chdir(originalDirectory)
                dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                os.chdir(originalDirectory+dirname)

                RefMI = np.loadtxt('RefMutualInfo.dat')
                RefMI = np.mean(RefMI)
                print RefMI
                raw_input("press enter to continue...")


                window_size = 500
                trans_times = []
                for exp in range(100):
                    tau = 0
                    Transition = False
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)

                    print exp
                    mutual_info = np.loadtxt('%iMutualInfo.dat' %exp)

                    for q in range(len(mutual_info)-window_size):
                        mean = np.mean(mutual_info[q:q+window_size])
                        if mean < (0.7)*RefMI:
                            Transition = True
                            tau = q
                            trans_times.append(tau)
                            break


                dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                os.chdir(originalDirectory+dirname)

                n, bins, patches = plt.hist(trans_times, 10,cumulative= True,  histtype='stepfilled')
                plt.show()
                print np.mean(trans_times)
                n, bins, patches = plt.hist(trans_times, 10, cumulative= True,  histtype='stepfilled')
                plt.savefig('TransTimes.png')
                os.chdir(originalDirectory)
#########################################################################################################################
def EA_Length_Distro():
    import numpy as np
    import random
    import os
    import Parameters
    import matplotlib.pylab as plt


    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [1.0]
    KR = [0.005]

    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):               
                os.chdir(originalDirectory)
                dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                os.chdir(originalDirectory+dirname)

                RefMI = np.loadtxt('RefMutualInfo.dat')
                RefMI = np.mean(RefMI)
                print RefMI
                raw_input("press enter to continue...")


                window_size = 500
                trans_times = []
                lengths_before = []
                lengths_after =  []
                for exp in range(100):
                    tau = 0
                    Transition = False
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)

                    print exp
                    mutual_info = np.loadtxt('%iMutualInfo.dat' %exp)

                    for q in range(len(mutual_info)-window_size):
                        mean = np.mean(mutual_info[q:q+window_size])
                        if mean < (0.5)*RefMI:
                            Transition = True
                            tau = q
                            trans_times.append(tau)
                            break
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i/Landscapes' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)

                    # Pre Transition
                    
                    for q in range(0,tau, 100):
                        loaded_lengths =np.loadtxt('%iLengths%i.dat' %(0, q))
                        lengths_before.extend(loaded_lengths)
                    for q in range(np.around(q, decimals = -2), Parameters.tau_max, 100):
                        loaded_lengths =np.loadtxt('%iLengths%i.dat' %(0, q))
                        lengths_after.extend(loaded_lengths)
                
                dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                os.chdir(originalDirectory+dirname)

                n, bins, patches = plt.hist(lengths_before, histtype='stepfilled')
                plt.savefig('%iPreTrans_Length.png' %exp)
                plt.show()
                plt.close()

                n, bins, patches = plt.hist(lengths_after, histtype='stepfilled')
                plt.savefig('%iPostTrans_Length.png' %exp)
                plt.show()
                plt.close()
                

                # n, bins, patches = plt.hist(trans_times, 10,cumulative= True,  histtype='stepfilled')
                # plt.show()
                # print np.mean(trans_times)
                # n, bins, patches = plt.hist(trans_times, 10, cumulative= True,  histtype='stepfilled')
                # plt.savefig('TransTimes.png')


#########################################################################################################################
def EA_Mass_Distro():
    import numpy as np
    import random
    import os
    import Parameters
    import matplotlib.pylab as plt


    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [1.0, 0.75, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    KR = [0.005]

    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):               
                os.chdir(originalDirectory)
                dirname = ('/data11.22/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                os.chdir(originalDirectory+dirname)

                
                RefMI = 0.010

                


                window_size = 100
                trans_times = []

                masses_before = {}
                masses_after =  {}
                for exp in range(100):
                    tau = 0
                    Transition = False
                    dirname = ('/data11.22/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)

                    print exp
                    mutual_info = np.loadtxt('%iMutualInfo.dat' %exp)
                    print len(mutual_info)

                    for q in range(len(mutual_info)-window_size):
                        mean = np.mean(mutual_info[q:q+window_size])
                        if mean < RefMI:
                            Transition = True
                            tau = q
                            trans_times.append(tau)
                            break
                    dirname = ('/data11.22/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i/Landscapes' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)

                    # Pre Transition
                    print tau
                    mass = 0
                    for q in range(0,np.around(tau, decimals = -1), 10):
                        lengths, masses =np.loadtxt('%iMasses%i.dat' %(0, q))
                        for p in range(len(lengths)):
                            if lengths[p] in masses_before:
                                masses_before[lengths[p]] += float(masses[p])
                            else: 
                                masses_before[lengths[p]] = float(masses[p])
                        mass +=masses[p]
                    if Transition == True and np.around(tau, decimals = -1)< Parameters.tau_max:
                        print np.around(tau, decimals = -1)
                        mass = 0
                        for q in range(np.around(tau, decimals = -1), Parameters.tau_max-1, 10):
                            lengths, masses=np.loadtxt('%iMasses%i.dat' %(0, q))

                            for p in range(len(lengths)):
                                if lengths[p] in masses_after:
                                    masses_after[lengths[p]] += float(masses[p])
                                else: 
                                    masses_after[lengths[p]] = float(masses[p])
                                mass +=masses[p]
                
                dirname = ('/data11.22/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                os.chdir(originalDirectory+dirname)
                lengths = []
                masses = []
                for key in masses_before:
                    lengths.append(key)
                    masses.append(masses_before[key])


                plt.bar(lengths, masses)
                plt.savefig('PreTrans_Mass.png')
                plt.show()
                plt.close()
            
                lengths = []
                masses = []
                for key in masses_after:
                    lengths.append(key)
                    masses.append(masses_after[key])
                plt.bar(lengths, masses)
                plt.savefig('PostTrans_Mass.png')
                plt.show()
                plt.close()
#########################################################################################################
def transferEntropy():
    import numpy as np
    import random
    import os
    import Parameters


    
    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [1.0]
    KR = [0.005]


    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):
                for exp in range(100):
                    os.chdir(originalDirectory)
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)
##########################################################################################################
def plot_hetero():
    import numpy as np
    import random
    import os
    import Parameters

    import matplotlib.pylab as plt
    
    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [1.0, 0.5]
    KR = [0.005]


    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):
                for exp in range(100):
                    os.chdir(originalDirectory)
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)

                    t, hetero = np.loadtxt('%iMean_Heterogenaity.dat' % exp, unpack =True )

                    os.chdir(originalDirectory+'/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f'% (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                    plt.plot(t, hetero)
                    plt.savefig('%iHeteroVStime.png' %exp)
                    plt.close()
##########################################################################################################
def plot_length():
    import numpy as np
    import random
    import os
    import Parameters

    import matplotlib.pylab as plt
    
    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [1.0]
    KR = [0.005]


    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):
                for exp in range(100):
                    os.chdir(originalDirectory)
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)

                    t, length = np.loadtxt('%i_aveLen.dat' % exp, usecols = (0,1), unpack =True )

                    os.chdir(originalDirectory+'/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f'% (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
                    plt.plot(t, length)
                    plt.savefig('%iLengthVsTime.png' %exp)
                    plt.close()
#########################################################################################################
def plot_length_distro():
    import numpy as np
    import random
    import os
    import Parameters

    import matplotlib.pylab as plt
    
    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [1.0]
    KR = [0.005]


    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):
                for exp in range(1):
                    os.chdir(originalDirectory)
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i/Landscapes' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)
                    for i in range(0,Parameters.tau_max, 100):
                        lengths =np.loadtxt('%iLengths%i.dat' %(exp, i))
                        
                        n, bins, patches = plt.hist(lengths, 50, histtype='stepfilled')
                        plt.savefig('%iLength.png' %i)
                        plt.close()
                        #i +=100
    os.chdir(originalDirectory)
#########################################################################################################
def mass_flow_chart():
    import numpy as np
    import random
    import os
    import Parameters

    import networkx as nx
    import matplotlib.pylab as plt
    
    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [1.0]
    KR = [0.005]
    monomers = ['0', '1']

    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):
                for exp in range(1):
                    os.chdir(originalDirectory)
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)
                    time, ones = np.loadtxt('Populations/1.dat', unpack = True)
                    time, zeros = np.loadtxt('Populations/0.dat', unpack = True)
                    time_series_length = len(ones)

                    IDs, sequences  = np.loadtxt('0sequencedata.txt', unpack =True, usecols = (0,1,3))

                    for seq in sequences:
                        # Load population time series
                        time, seq_pop = np.loadtxt('Populations/%s.dat' %seq, unpack = True)
                        # ReShape with zeros to match total time series
                        time_series_difference = time_series_length - len(seq_pop)
                        seq_pop = np.pad(seq_pop, (time_series_difference,0 ), 'constant', constant_values = 0 )


    os.chdir(originalDirectory)
#########################################################################################################
def plot_mass_distro():
    import numpy as np
    import random
    import os
    import Parameters

    import matplotlib.pylab as plt
    
    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [0.5]
    KR = [0.005]


    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):
                for exp in range(1):
                    os.chdir(originalDirectory)
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i/Landscapes' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)
                    for i in range(0,Parameters.tau_max, 100):
                        lengths, masses =np.loadtxt('%iMasses%i.dat' %(exp, i))
                        
                        plt.bar(lengths, masses)
                        plt.savefig('%iMass.png' %i)
                        plt.close()
                        #i +=100
##########################################################################################################
def entropy(sample, distro):
    import numpy as np
    H= 0
    for value in sample:
        p = distro[value]

        H -= p*np.log(p)
    return H
#########################################################################################################
def jointEntropy(sample1, sample2, distro):
    import numpy as np
    H= 0
    sample = zip(sample1, sample2)
    for value in sample:
        p = distro[value]

        H -= p*np.log(p)
    return H
#########################################################################################################    
def CalcMutualInfo(sample1,sample2, distro1, distro2, jointDistro):
    import numpy as np
    I = 0
    joint = []
    for j in range(len(sample1)-1):
        joint.append((sample1[j][0], sample1[j][1], sample2[j][0], sample2[j][1]))
    for j in range(len(sample1)-1):
        if sample1[j] not in distro1:
            print "Distro1 does not contain this key"
            print type(sample1[j])
            print sample1[j]
            raw_input('Press Enter to Continue')
            continue
        if sample2[j] not in distro2:
            print "Distro2 does not contain this key"
            continue
        p1 = distro1[sample1[j]]
        p2 = distro2[sample2[j]]
    
        pJoint = jointDistro[joint[j]]

        if p1 != 0 and p2 != 0:
            I += pJoint*np.log(pJoint/(p1*p2))
    return I
#########################################################################################################
def fixation_times():
    import numpy as np
    import random
    import os
    import Parameters


    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [0.50, 5.0, 0.1, 0.05, 0.01, 1.0]
    KR = [0.05 , 0.010, 0.005, 0.001, 0.50, 0.10]

    zeros = '0000000'
    ones  = '1111111'

    filename = 'fixation_times.dat'
    file= open(filename, 'a')
    s = 'KH     KR      Time to Zeros       Time to Ones' 
    file.write(s)
    file.write('\n')
    file.close() 
    
    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):
                FIX_Times_Zeros = []
                FIX_Times_Ones = []
                for exp in range(100):
                    os.chdir(originalDirectory)
                    dirname = ('/data10.6/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)

                    seqs, d_times = np.loadtxt('%isequencedata.txt'%exp, unpack =True, usecols= (1,2), dtype= str)
                
                    for n in range(len(seqs)):
                        if seqs[n] == zeros:
                            FIX_Times_Zeros.append(float(d_times[n]))
                        elif seqs[n] == ones:
                            FIX_Times_Ones.append(float(d_times[n]))

                Mean_Time_Ones = np.mean(FIX_Times_Ones)
                Mean_Time_Zeros = np.mean(FIX_Times_Zeros)
                
                os.chdir(originalDirectory)

                filename = 'fixation_times.dat'

                file =open(filename, 'a')
                s = str(KH[j])+'    '+str(KR[k]) + '    '+str(Mean_Time_Zeros)+'    '+str(Mean_Time_Ones)
                file.write(s)
                file.write('\n')
                file.close() 
#########################################################################################################
def symmetric_probabilities():
    import numpy as np
    import random
    import os
    import Parameters


    
    originalDirectory = os.getcwd()
    KP = [0.0005]
    KH = [1.0]
    KR = [0.005]


    for i in range(len(KP)):
        for j in range(len(KH)):
            for k in range(len(KR)):
                for exp in range(100):
                    os.chdir(originalDirectory)
                    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], Parameters.km, Parameters.tau_max, sum(Parameters.M_N), exp))
                    os.chdir(originalDirectory+dirname)
#########################################################################################################
def species_count():
    import numpy as np
    import random
    import os
    import Parameters

    import matplotlib.pylab as plt

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
    plt.rc('font', **font)
    exp = 17
    t, living, ever = np.loadtxt('%ispeciescount.dat'%exp, unpack =True, usecols= (0,1,2), dtype= float)
    fig = plt.figure(figsize= (10,10))
    plt.plot(t,living, label = 'Extant')
    plt.plot(t,ever, label = 'Explored')
    plt.axis([0,10000, 0, 600])
    plt.xlabel('Time')
    plt.ylabel('Number of Distinct Species')
    plt.legend(loc = 'upper left')
    plt.savefig('species_count.png')
#########################################################################################################
def plot_fixation_times():
    import numpy as np
    import random
    import os
    import Parameters
    import matplotlib.pylab as plt

    originalDirectory = os.getcwd()
    kh, kr, tZ, tO = np.loadtxt('fixation_times.dat', unpack = True, dtype =float, skiprows= 1)
    colors = ['r', 'g', 'b', 'm', 'y', 'c']

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

    plt.rc('font', **font)
    zips = zip(kh, kr, tZ, tO)
    zips = sorted(zips, key=lambda zip: zip[1])
    kh, kr, tZ, tO = zip(*zips)
    fig = plt.figure(figsize= (10,10))
    Kr = sorted(list(set(kr)))
    color_count = 0
    for KR in Kr:
        times= []
        hydrolysis = []
        for i in range(len(kh)):
            if kr[i]== KR:
                times.append(tZ[i])
                hydrolysis.append(np.log(kh[i]))

        pairs = zip(times, hydrolysis)
        pairs = sorted(pairs, key=lambda pair: pair[1])
        times, hydrolysis = zip(*pairs)
        plt.scatter(hydrolysis,times, color = colors[color_count])                    
        plt.plot(hydrolysis,times, label = 'kr = '+str(KR), color = colors[color_count])
        color_count +=1
    plt.legend(ncol = 2)
    plt.ylabel('Mean Fixation time for Best Replicator')
    plt.xlabel('ln(kH)')
    plt.axis([-5,0.1, 0, 6000])
    plt.savefig('MeanfixationtimeVSKH.png')
    plt.show()
#########################################################################################################
def print_polymer_Populations(IDs, seqs,exp):
    #import matplotlib.pylab as plt
    import numpy as np
    from string import split
    import random
    from operator import add
    allpops =[[], [], [], [], [], [], [], []]
    maxtime = 0
    #print allpops
    #print seqs
    
    alltime = []
    for i in range(len(IDs)-1):
        ID = IDs[i]
        seq= seqs[i]
        #print seq
        concentrations = determine_monomer_abundances(seq)
        
        length = len(str(seq))
        index= concentrations[0]

        t,Pop = np.loadtxt(('%iPopulations/%i.dat' %(exp, ID)), unpack = True)
        allpops[index].append(Pop)
        if len(t) > maxtime:
            maxtime= len(t)
            alltime[:] =t 
    combinedPops = []
    for i in range(8):
        combinedPops.append([0.0]*maxtime)
        for j in range(len(allpops[i])-1):
            padding = [0.0]* (maxtime -len(allpops[i][j]) )
            padding.extend(allpops[i][j])
            
            combinedPops[i] = map(add, padding, combinedPops[i])
            if len(combinedPops[i]) != maxtime:
                print "Fucked up"
        print len(combinedPops[i])
    print 'Done reducing degeneracy of populations'
    fig = plt.figure(figsize=(10,10), dpi = 200)
    ax1 = fig.add_subplot(111)
    for i in range(len(combinedPops)):
        color = (float(i)/7.0, 0, float(7-i)/7.0)
        ax1.scatter(alltime, combinedPops[i], c = color, label = ('%i Zeros' %i))
    plt.legend(loc = 'upper right')    
    plt.savefig('%iLen7Polymers.png'%exp)
########################################################################################################
def determine_monomer_abundances(sequence):
    """ Takes a string of M monomer types, returns a list of the monomers present """
    
    monomers=['0', '1']
    m=2
    
    #Intialize list of concentrations
    concentrations=[0]*m
    
    
    #for each monomer in the sequence, identify the monomer
    for n in range(1,len(sequence)+1):
        for i in range(0, m):
            if sequence[(n-1):n] == monomers[i]:
                concentrations[i] += 1
    return concentrations  
########################################################################################################
def polymer_pop_plot():
    import Parameters
    import os
    import shutil
    # import multiprocessing
    #import matplotlib.pylab as plt
    import numpy as np
    originalDirectory = os.getcwd()
    Runs=[[0.0005, 0.50, 0.0050, 0.01]]
    for i in range(len(Runs)):
        os.chdir(originalDirectory)

        Parameters.kp =Runs[0][0]
        Parameters.kh =Runs[0][1]
        Parameters.kr =Runs[0][2]
        Parameters.km =Runs[0][3]
        dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (Parameters.kp, Parameters.kh, Parameters.kr, Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
        newdir = originalDirectory +dirname
        os.chdir(newdir)
        for exp in range(256):
            print 'Analyzing Experiment %i' %exp
            printing_IDs = []
            printing_seqs = []
            IDs, seqs = np.loadtxt(('%isequencedata.txt' %exp), unpack = True, dtype = str, usecols= (0,1), skiprows = 1)
            #print exp
            #print len(seqs) - len(set(seqs))
            for i in range(len(seqs)-1):
                length =len(str(seqs[i]))
                ID = int(IDs[i])
                if length == 7:
                    printing_IDs.append(ID)
                    printing_seqs.append(seqs[i])
            print "Done aquiring len 7 IDs"

            print_polymer_Populations(printing_IDs, printing_seqs,exp)
            print "Done printing population graph"

            t, Ap_p, Ap_h, Ap_r =  np.loadtxt(('%iPropensities.dat' %exp), unpack = True)
            fig = plt.figure(figsize=(10,10), dpi = 100)
            ax1 = fig.add_subplot(111)
            ax1.scatter(t, Ap_p, c = 'b', label= 'Ap_p')
            ax1.scatter(t, Ap_h, c = 'r', label= 'Ap_h')
            ax1.scatter(t, Ap_r, c = 'g', label= 'Ap_r')
            plt.legend(loc = 'upper right')
            plt.savefig('%iPropensities.png' %exp)
            print exp
##############################################################################################################################
def print_transition_reps(): 
    import Parameters
    import os
    import shutil
    
    #import matplotlib.pylab as plt
    import numpy as np
    originalDirectory = os.getcwd()
    Runs=[[0.0005, 0.50, 0.0050, 0.01]]
    for i in range(len(Runs)):
        os.chdir(originalDirectory)

        Parameters.kp =Runs[0][0]
        Parameters.kh =Runs[0][1]
        Parameters.kr =Runs[0][2]
        Parameters.km =Runs[0][3]
        dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (Parameters.kp, Parameters.kh, Parameters.kr, Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
        newdir = originalDirectory +dirname
        os.chdir(newdir)

        #Use Reference
        filename = '%iPropensities.dat' % 35
        t, Div = np.loadtxt(filename, unpack =True, usecols =(0,3))
        reference  = max(Div)
        # reference = 3.0
        print reference
        print sum(Div)
        raw_input('Press Enter to Continue...')
        

        important_lengths = [1,3,5,6,7,8,9,11]
        fig = plt.figure()
        num_lengths = len(important_lengths)
        for investigating_length in important_lengths:
            os.chdir(newdir)
            fig = plt.figure()
            transition_replicators =[[], [], []]
            for exp in range(256):
                print "Working on exp: " +str(exp)
                filename = '%iPropensities.dat' % exp
                t, Div = np.loadtxt(filename, unpack =True, usecols=(0,3))

                # Find the phase change times
                phase_change = False
                for m in range(len(t)):
                    if Div[m] > reference:
                        change = Div[m]
                        phase_change = True
                        tau_crit = t[m] 
                        break


                #Find all the replicating length polymers
                printing_IDs = []
                printing_seqs = []
                IDs, lengths = np.loadtxt(('%isequencedata.txt' %exp), unpack = True, dtype = str, usecols= (0,3), skiprows = 1)
                print len(lengths)
                #print len(seqs) - len(set(seqs))
                for i in range(len(lengths)-1):
                    length = int(lengths[i])
                    # print length
                    ID = int(IDs[i])
                    if length == investigating_length:
                        printing_IDs.append(ID)
                        #printing_seqs.append(seqs[i])
                

                replicators = [0.0, 0.0, 0.0] 

                for i in range(len(printing_IDs)-1):
                    
                    ID = printing_IDs[i]
                    
                    t,Pop = np.loadtxt(('%iPopulations/%i.dat' %(exp, ID)), unpack = True)
                    if type(t) == np.float64:
                        if t == tau_crit:

                            replicators +=Pop[j]
                            break
                    else:
                        for j in range(len(t)):
                            if t[j] == tau_crit:
                                replicators[0] +=Pop[j-5]
                                replicators[1] +=Pop[j]
                                replicators[2] +=Pop[j+5]
                                break
                

                transition_replicators[0].append(replicators[0])
                transition_replicators[1].append(replicators[1])
                transition_replicators[2].append(replicators[2])
            print transition_replicators
            os.chdir(originalDirectory)


            ax = fig.add_subplot(111) 
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            plt.title('Length %i Polymers present' % investigating_length, y=1.05)


            ax1 = fig.add_subplot(311)
            ax1.set_title('Before Transtion')
            # the histogram of the data
            n, ObsBins, patches = ax1.hist(transition_replicators[0], 25, facecolor='green')
            x0_1,xmax_1 = ax1.get_xlim()

            ax2 = fig.add_subplot(312)
            ax2.set_title('At Transtion')
            n, ObsBins, patches = ax2.hist(transition_replicators[1], 25, facecolor='green')
            x0_2,xmax_2 = ax2.get_xlim()

            ax3 = fig.add_subplot(313)
            ax3.set_title('After Transtion')
            n, ObsBins, patches = ax3.hist(transition_replicators[2], 25, facecolor='green')
            x0_3,xmax_3 = ax3.get_xlim()

            xMax = max([xmax_3, xmax_2, xmax_1])
            ax1.set_xlim([0,xMax])
            ax2.set_xlim([0,xMax])
            ax3.set_xlim([0,xMax])
            
            fig.tight_layout()
            plt.savefig('L%iTransitionHist.png' %investigating_length)
##############################################################################################################################
def main():  #discovery_time
    import Parameters
    import os
    import shutil
    
    #import matplotlib.pylab as plt
    import numpy as np
    originalDirectory = os.getcwd()
    Runs=[[0.0005, 0.50, 0.0050, 0.01]]
    for i in range(len(Runs)):
        os.chdir(originalDirectory)

        Parameters.kp =Runs[0][0]
        Parameters.kh =Runs[0][1]
        Parameters.kr =Runs[0][2]
        Parameters.km =Runs[0][3]
        dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (Parameters.kp, Parameters.kh, Parameters.kr, Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
        
        #Use Reference
        filename = 'RefPropensities.dat'
        t, Div = np.loadtxt(filename, unpack =True, usecols =(0,3))
        reference  = max(Div)



        newdir = originalDirectory +dirname
        os.chdir(newdir)
        OneSeq = '1111111'
        ZeroSeq= '0000000'
        time_to_zeros = []
        time_to_ones = []
        transition_to_zeros = []
        transition_to_ones = []

        for exp in range(256):
            print "Working on exp: " +str(exp)

            filename = '%iPropensities.dat' % exp
            t, Div = np.loadtxt(filename, unpack =True, usecols=(0,3))

            # Find the phase change times
            phase_change = False
            for m in range(len(t)):
                if Div[m] > reference:
                    change = Div[m]
                    phase_change = True
                    tau_crit = t[m] 
                    break

            #Find all the replicating length polymers
            zero_ID = 0
            one_ID = 0
            IDs, seqs = np.loadtxt(('%isequencedata.txt' %exp), unpack = True, dtype = str, usecols= (0,1), skiprows = 1)
            for i  in range(len(seqs)):
                if seqs[i] ==OneSeq:
                    one_ID = i

                elif seqs[i] == ZeroSeq:
                    zero_ID = i 

                if one_ID != 0 and zero_ID != 0:
                    break

                
                
            t1,Pop1 = np.loadtxt(('%iPopulations/%i.dat' %(exp, one_ID)), unpack = True)
            t0,Pop0 = np.loadtxt(('%iPopulations/%i.dat' %(exp, zero_ID)), unpack = True)
            if type(t1) == np.float64:
                  transition_to_ones.append((t1-tau_crit))
            if type(t0) == np.float64:
                transition_to_zeros.append((t0-tau_crit))
            else:
                transition_to_ones.append((t1[0]-tau_crit))
                transition_to_zeros.append((t0[0]-tau_crit))
        os.chdir(originalDirectory)
        np.savetxt('TransitionToOnes.dat', transition_to_ones)
        np.savetxt('TransitiontoZeros.dat', transition_to_zeros)
##############################################################################################################################
def rep_mass_at_transition(): 
    import Parameters
    import os
    import shutil
    
    #import matplotlib.pylab as plt
    import numpy as np
    originalDirectory = os.getcwd()
    Runs=[[0.0005, 0.50, 0.0050, 0.01]]
    for i in range(len(Runs)):
        os.chdir(originalDirectory)

        Parameters.kp =Runs[0][0]
        Parameters.kh =Runs[0][1]
        Parameters.kr =Runs[0][2]
        Parameters.km =Runs[0][3]
        dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (Parameters.kp, Parameters.kh, Parameters.kr, Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
        newdir = originalDirectory +dirname
       

        #Use Reference
        filename = 'RefPropensities.dat'
        t, Ap_r = np.loadtxt(filename, unpack =True, usecols =(0,3))
        reference  = max(Ap_r)
        os.chdir(newdir)
        # reference = 3.0
        print reference
        raw_input('Press Enter to Continue...')

        transition_mass =[[], [], []]

        for exp in range(256):
            replicator_mass = [0.0, 0.0, 0.0]
            print "Working on exp: " +str(exp)
            filename = '%iPropensities.dat' % exp
            t, Ap_r = np.loadtxt(filename, unpack =True, usecols=(0,3))

            # Find the phase change times
            phase_change = False
            for m in range(len(t)):
                if Ap_r[m] > reference:
                    change = Ap_r[m]
                    phase_change = True
                    tau_crit = t[m] 
                    break

            t,Mass, Ones, Zeros = np.loadtxt(('%iRep_Mass.dat' %(exp)), unpack = True)
            
            for j in range(len(t)):
                if t[j] == tau_crit:
                    replicator_mass[0] +=Mass[j-2]
                    replicator_mass[1] +=Mass[j]
                    replicator_mass[2] +=Mass[j+2]
                    break
            

            transition_mass[0].append(replicator_mass[0])
            transition_mass[1].append(replicator_mass[1])
            transition_mass[2].append(replicator_mass[2])
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        # the histogram of the data
        n, ObsBins, patches = ax1.hist(transition_replicators[0], 25, facecolor='green')

        ax2 = fig.add_subplot(312)
        n, ObsBins, patches = ax2.hist(transition_replicators[1], 25, facecolor='green')
        ax3 = fig.add_subplot(313)
        n, ObsBins, patches = ax3.hist(transition_replicators[2], 25, facecolor='green')
        plt.savefig('MassTransitionHist.png')
        np.savetxt('transition_replicator_mass.dat', transition_mass)          
##############################################################################################################################
def main2(): 
    import Parameters
    import os
    import shutil
    
    #import matplotlib.pylab as plt
    import numpy as np
    from operator import add
    originalDirectory = os.getcwd()
    Runs=[[0.0005, 0.50, 0.0050, 0.01]]
    for i in range(len(Runs)):
        os.chdir(originalDirectory)

        Parameters.kp =Runs[0][0]
        Parameters.kh =Runs[0][1]
        Parameters.kr =Runs[0][2]
        Parameters.km =Runs[0][3]
        dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (Parameters.kp, Parameters.kh, Parameters.kr, Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
        newdir = originalDirectory +dirname
       

        #Use Reference
        filename = 'RefPropensities.dat'
        t, Ap_r = np.loadtxt(filename, unpack =True, usecols =(0,3))
        reference  = max(Ap_r)
        os.chdir(newdir)
        # reference = 3.0
        print reference
        raw_input('Press Enter to Continue...')

        preTransitionResource = []
        postTransitionResouce =[]
        CompleteResource = []
        preTransitionRep = []
        postTransitionRep =[]
        CompleteRep = []

        for exp in range(256):
            
            print "Working on exp: " +str(exp)
            filename = '%iPropensities.dat' % exp
            t, Ap_r = np.loadtxt(filename, unpack =True, usecols=(0,3))

            # Find the phase change times
            phase_change = False
            for m in range(len(t)):
                if Ap_r[m] > reference:
                    change = Ap_r[m]
                    phase_change = True
                    tau_crit = t[m] 
                    break

            onearr = np.loadtxt('%i_monomer_1.dat' %exp)
            ones = onearr[:,1]
            zeroarr = np.loadtxt('%i_monomer_0.dat' % exp)
            zeros = zeroarr[:,1]
            monomers = map(add, ones, zeros)
            repMassArr = np.loadtxt('%iRep_Mass.dat' % exp)
            repMass = repMassArr[:,1]
            repOnes = repMassArr[:,3]
            onemersRatio = []
            oneRepMassRatio = []
            for i in range(len(ones)):
                ratio = float(ones[i])/float(monomers[i])
                onemersRatio.append(ratio)
                if repMass[i] == 0:
                    oneRepMassRatio.append(0)
                else:
                    ratio = float(repOnes[i])/float(repMass[i])
                    oneRepMassRatio.append(ratio)

            preTransitionResource.extend(onemersRatio[:m ])
            postTransitionResouce.extend(onemersRatio[m:])
            CompleteResource.extend(onemersRatio)
            preTransitionRep.extend(oneRepMassRatio[:m])
            postTransitionRep.extend(oneRepMassRatio[m:])
            CompleteRep.extend(oneRepMassRatio)
        os.chdir(originalDirectory)
        np.savetxt('preTransitionRep.dat', preTransitionRep)
        np.savetxt('preTransitionResource.dat', preTransitionResource)
        np.savetxt('postTransitionRep.dat', postTransitionRep)
        np.savetxt('postTransitionResouce.dat', postTransitionResouce)
        np.savetxt('CompleteRep.dat', CompleteRep)
        np.savetxt('CompleteResource.dat', CompleteResource)       
##############################################################################################################################
def main1(): 
    import Parameters
    import os
    import shutil
    from operator import sub
    
    #import matplotlib.pylab as plt
    import numpy as np
    originalDirectory = os.getcwd()
    Runs=[[0.0005, 0.50, 0.0050, 0.01]]
    for i in range(len(Runs)):
        os.chdir(originalDirectory)

        Parameters.kp =Runs[0][0]
        Parameters.kh =Runs[0][1]
        Parameters.kr =Runs[0][2]
        Parameters.km =Runs[0][3]
        dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (Parameters.kp, Parameters.kh, Parameters.kr, Parameters.km, Parameters.tau_max, sum(Parameters.M_N)))
        newdir = originalDirectory +dirname
        

        #Use Reference
        filename = 'RefPropensities.dat'
        t, Ap_r = np.loadtxt(filename, unpack =True, usecols =(0,3))
        reference  = max(Ap_r)
        os.chdir(newdir)
        # reference = 3.0
        print reference
        raw_input('Press Enter to Continue...')

        transition_ratio =[[]]#, []]

        for exp in range(256):
            replicator_ratio = [0.0, 0.0]
            monomer_ratio = [0.0, 0.0]
            print "Working on exp: " +str(exp)
            filename = '%iPropensities.dat' % exp
            t, Ap_r = np.loadtxt(filename, unpack =True, usecols=(0,3))

            # Find the phase change times
            phase_change = False
            for m in range(len(t)):
                if Ap_r[m] > reference:
                    change = Ap_r[m]
                    phase_change = True
                    tau_crit = t[m] 
                    break

            t,Mass, Rep_Zeros, Rep_Ones = np.loadtxt(('%iRep_Mass.dat' %(exp)), unpack = True)
            t0, Zeros =np.loadtxt(('%i_monomer_0.dat' %(exp)), unpack = True)
            t1, Ones =np.loadtxt(('%i_monomer_1.dat' %(exp)), unpack = True)

            for j in range(len(t)):
                if t[j] == tau_crit and Mass[j] != 0.0:

                    #replicator_ratio[0] +=float(Rep_Zeros[j-5]/(Rep_Zeros[j-5]+Rep_Ones[j-5]))
                    replicator_ratio[0] +=float(Rep_Zeros[j]/(Rep_Zeros[j]+Rep_Ones[j]))
                    #replicator_ratio[2] +=float(Rep_Zeros[j+2]/(Rep_Zeros[j+2]+Rep_Ones[j+2]))
                    break
            for i in range(len(t0)):
                if t0[i] == tau_crit and Zeros[i]+Ones[i] !=0:
                    #monomer_ratio[0] += float(Zeros[i-5]/(Zeros[i-5] +Ones[i-5]))
                    monomer_ratio[0] += float(Zeros[i]/(Zeros[i] +Ones[i]))
                    #monomer_ratio[2] += float(Zeros[i+2]/(Zeros[i+2] +Ones[i+2]))

            ratio_difference = map(sub, monomer_ratio, replicator_ratio)
        
            

            #transition_ratio[0].append(ratio_difference[0])
            transition_ratio[0].append(ratio_difference[0])
           # transition_ratio[2].append(ratio_difference[2])
        os.chdir(originalDirectory)
        np.savetxt('transition_replicator_ratio.dat', transition_ratio)
            # fig = plt.figure()
            # ax1 = fig.add_subplot(311)
            # # the histogram of the data
            # n, ObsBins, patches = ax1.hist(transition_replicators[0], 25, facecolor='green')

            # ax2 = fig.add_subplot(312)
            # n, ObsBins, patches = ax2.hist(transition_replicators[1], 25, facecolor='green')
            # ax3 = fig.add_subplot(313)
            # n, ObsBins, patches = ax3.hist(transition_replicators[2], 25, facecolor='green')
            # plt.savefig('L%iTransitionHist.png' %investigating_length)
##############################################################################################################################
def k_test():
    import os
    import numpy as np 
    import matplotlib.pylab as plt
    from scipy import stats

    Runs=[[0.0005, 0.50, 0.0050, 0.01]]
    originalDirectory = os.getcwd()
    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f' % (Runs[0][0], Runs[0][1], Runs[0][2], Runs[0][3], 1000.0,1000))
    newdir = originalDirectory +dirname
    os.chdir(newdir)

    change_times =np.loadtxt('Phase_change_Times.dat')
    expoential = np.random.exponential(250 ,size= len(change_times))
    for x in change_times:
        x = x/max(change_times)
    for x in expoential:
        x = x/max(expoential)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # the histogram of the data

    n, ObsBins, patches = ax1.hist(change_times, 25, facecolor='green', cumulative=True)
    # for i in range(len(ObsBins)):
    #     ObsBins[i] = sum[ObsBins[:i-1]]
    n, TestBins,patches = ax2.hist(expoential, 25, facecolor = 'blue',cumulative=True)
    # for i in range(len(TestBins)):
    #     TestBins[i] = sum[TestBins[:i-1]]
    

    # fig2 = plt.figure()
    # ax1 = fig2.add_subplot(211)
    # ax2 = fig2.add_subplot(212)


    # plt.savefig('CummChangeTimes.png')
   

    D, p = stats.ks_2samp(change_times,expoential)
    print (D,p)
    plt.title('KS Statistic: D= %f     p= %f' %(D,p))

    plt.savefig('ChangeTimes.png')
#########################################################################################################################
if __name__=="__main__":
    #probability_distributions_ints()
    
    #plot_mass_distro()
    #mutualInfoTimeSeries_ints()
    
    #EA_Mutual_Info()
    EA_Mass_Distro()
    # determine_transition_times()
    # EA_Length_Distro()
    #fixation_times()
    #plot_fixation_times()
    #species_count()
    # plot_hetero()
    # plot_length()
    
			            
