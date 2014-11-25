# This File contains all the class objects for simulating "The Race up Mount Improbable"
# copyright Sara Imari Walker 2011
#Edited By Cole Mathis 2013

from numpy import*
import Parameters


###################################################################################################

class Polymer(object):
    """A Polymeric Sequence"""
    
    def __init__(self, ID, population, length, sequence, concentrations):
        #print("A new polymer has been born!")
        self.ID = ID
       	self.pop = population
        self.t_discovery = 0.0
	self.len = length
	self.nb = (length-1)
        self.seq = sequence
        self.con= concentrations #LIST!
        self.TimePop= 0.0
        self.TimePop_Long = 0.0
        self.SumPop = 0.0
        self.stable = False

        self.Kh = 0.0
        self.Kr = 0.0 
        self.Ap_p = 0.0
        self.Ap_h = 0.0
        self.Ap_r = 0.0
        self.Ap_d = 0.0

        self.F = False    #if TRUE, sequence contains trivial replicator
        self.F_motifs = []
        #self.T_frags = [[-1]*2 for i in range(self.nb)]
        self.NT = False   #if TRUE, sequences contains nontrivial replicator
        self.NT_motifs = []
        #self.NT_frags = [[-1]*2 for i in range(self.nb)]
        
        


###################################################################################################
def determine_monomer_abundances(sequence):
    """ Takes a string of M monomer types, returns a list of the monomers present """
    from Parameters import monomer_speices
    concentrations=zeros(m)
    monomers=Parameters.monomer_speices
    for n in range(1,len(sequence)):
        for i in range(0, m):
            if sequence[(n-1):n] == monomers[i]:
                concetrations[i] += 1
    return concentrations
        