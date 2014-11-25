#!/usr/bin/python

import os
import shutil
import multiprocessing
from subprocess import call
import time

KP = [0.0005]
KH = [0.50, 5.0, 0.1, 0.05, 0.01, 1.0]
KR = [0.05 , 0.010, 0.005, 0.001, 0.50, 0.10]
jobs = []
job = 0
originalDirectory = os.getcwd()
pool = multiprocessing.Pool()



for exp in range(100):
    shutil.copy(originalDirectory+'/TemplateParameters.py', originalDirectory+'/Parameters.py')
    file= open("Parameters.py", 'a')
    s = '\n\ncurrent_exp = %i' % job
    file.write(s)
    file.close()
    call(['qsub', 'NonTrivial.pbs'])
    #call(['python', 'master.py'])
    job +=1
    time.sleep(5)
        
