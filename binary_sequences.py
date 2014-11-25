#Copyright Cole Mathis 2014
import numpy as np
import random

################################################################################
def generate_binaries(N, L):
    """ Returns a list of N  distinct binary strings of length L """
    # This fails for lengths greater than 31
    
    sample=random.sample(xrange((2**L)-1),N)
    
    for i in range(0,len(sample)):
        sample[i]=bin(sample[i])[2:]
        sample[i]=sample[i].zfill(L)    
    
    return sample
    
################################################################################
def binary_primes(N,L):
    primes= primes_below(2**L)
    if len(primes) < N:
        print "Not enough distinct primes for this length"
    sample= random.sample(primes, N)
    
    for i in range(0,len(sample)):
        sample[i]=bin(sample[i])[2:]
        sample[i]=sample[i].zfill(L)    
    
    return sample

################################################################################
def primes_below(maxNumber):
    # This method was developed by Stack Overflow user Daniel G. (http://stackoverflow.com/users/207432/daniel-g)
    """
    Given a number, returns all numbers less than or equal to
    that number which are prime.
    """
    allNumbers = range(3, maxNumber+1, 2)
    for mIndex, number in enumerate(xrange(3, maxNumber+1, 2)):
        if allNumbers[mIndex] == 0:
            continue
        # now set all multiples to 0
        for index in xrange(mIndex+number, (maxNumber-3)/2+1, number):
            allNumbers[index] = 0
    return [2] + filter(lambda n: n!=0, allNumbers)
################################################################################