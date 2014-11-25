#!/usr/bin/python

import os
import shutil
import multiprocessing
from subprocess import call
import time
##############################################################################################################################################
def master():
	from Main import main
	print "Number of CPUs on machine:", multiprocessing.cpu_count()
	
	KP = [0.0005]
	KH = [0.5]
	KR = [0.005]
	
	jobs = []
	job_count = 0
	originalDirectory = os.getcwd()
	#pool = multiprocessing.Pool()


	for i in range(len(KP)):
	    for j in range(len(KH)):
	        for k in range(len(KR)):
	        	
		        	for exp in range(100):
						os.chdir(originalDirectory)
						
						import TemplateParameters as TParameters

						if TParameters.monomer_flux != 0.0: 
							dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_kd%.4f_km%.2f_Mf%.4f/%i' % (KP[i],KH[j], KR[k], TParameters.kd, TParameters.km, TParameters.monomer_flux, exp))

							if not os.path.exists(originalDirectory+ dirname):
								os.makedirs(originalDirectory+ dirname+ '/Landscapes')


						else: 
						    dirname = ('/data/kp%.4f_kh%.4f_kr%.4f_km%.2f_T%.1f_M%.0f/%i' % (KP[i],KH[j], KR[k], TParameters.km, TParameters.tau_max, 1000, exp))

						    if not os.path.exists(originalDirectory+ dirname):
						    	os.makedirs(originalDirectory+ dirname + '/Landscapes')

						newdir = originalDirectory +dirname
						shutil.copy(originalDirectory+'/Main.py', newdir)
						shutil.copy(originalDirectory+'/Reactions.py', newdir+'/Reactions.py')
						shutil.copy(originalDirectory+'/Initialize.py', newdir+'/Initialize.py')
						shutil.copy(originalDirectory+'/TemplateParameters.py', newdir+'/Parameters.py')
						shutil.copy(originalDirectory+'/Output.py', newdir+'/Output.py')
						shutil.copy(originalDirectory+'/Polymers.py', newdir+'/Polymers.py')
						shutil.copy(originalDirectory+'/binary_sequences.py', newdir+'/binary_sequences.py')
						#shutil.copy(originalDirectory+'/NonTrivial.pbs', newdir+'/NonTrivial.pbs')
						
						
						os.chdir(newdir)
						file= open("Parameters.py", 'a')
						s = '\n\ncurrent_exp = %i \nkp = %f \nkh = %f \nkr = %f ' % (exp, KP[i], KH[j], KR[k])
						print s
						file.write(s)
						file.close()
						
						from Main import main
						from Parameters import kp, kh, kr, current_exp
						print kp, kh, kr, current_exp
						p = multiprocessing.Process(target =main)
						jobs.append(p)
						p.start()
						
						while len(jobs) > 11:
							for job in jobs:
								if job.is_alive() == False:
									jobs.remove(job)
									print "Job's done."
									job_count +=1 
									print job_count, 'Completed Runs'


##################################################################################################
def frange(start, stop, size):
	values = []
	i = start
	while i< stop:
		values.append(i)
		i += size
	return values


# End Function Main ############################################################

master()
    

