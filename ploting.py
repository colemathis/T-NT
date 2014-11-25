import matplotlib.pylab as plt
import numpy as np
from string import split
import random

polymersPlotted = 8
with open('0_Populations.dat', 'r') as f:
	lines= f.readlines()
m =len(lines)
n = len(split(lines[m-1]))

Populations = np.zeros([m,n])

for i in range(m):
	floats = map(float, lines[i].split())
	for j in range(len(floats)):
		Populations[i,j] = floats[j]
print Populations[:,2]
fig= plt.figure(figsize=(12,12), dpi = 200)
ax1= fig.add_subplot(111)
colors= [0]*polymersPlotted
print len(colors)
for i in range (polymersPlotted):
	colors[i] = (random.random(), random.random(), random.random())
for i in range(polymersPlotted):
	ax1.scatter(Populations[:,0], Populations[:,i+1], color = colors[i], label =str(i-1))
plt.legend(loc='upper right')
plt.savefig('AllPopulations.png')