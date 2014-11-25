#########################################################################################################
def moments():
	import numpy as np
	import random
	import os
	import Parameters
	import matplotlib.pylab as plt

	t, particles, replicators = np.loadtxt('0particle_count.dat', unpack =True)
	t, Rep_Mass = np.loadtxt('0Rep_Mass.dat', unpack =True, usecols=(0,1))
	t, zeros = np.loadtxt('0_monomer_0.dat', unpack = True)
	t, ones = np.loadtxt('0_monomer_1.dat', unpack= True)

	m1_m0 = np.subtract(1000, particles)
	x1 = np.add(zeros, ones)
	m1r_m0r = np.subtract(Rep_Mass,replicators)


	thermo_x1 = float(Parameters.kh/Parameters.kp)*np.divide(m1_m0,particles)
	kinetic_x1 = float(Parameters.kh/Parameters.kr)*np.divide(m1_m0,m1r_m0r)
	plt.plot(t,x1,t, thermo_x1, t, kinetic_x1)
	plt.show()

	barrier = float(Parameters.kr/Parameters.kp)*np.divide(m1r_m0r, particles)
	plt.plot(t, barrier)
	plt.show()
	

#########################################################################################################
#########################################################################################################################
if __name__=="__main__":
	moments()
