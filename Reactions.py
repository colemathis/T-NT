# copyright Sara Imari Walker 2011
# Edited By Cole Mathis 2014

from Polymers import*
from numpy import*
import Parameters
import random

#####################################################################################################################################
def polymerization(sequences):
    """Takes a LIST of sequences (including monomers), polymerizes one sequence and updates populations  """
#This function chooses a sequence out of the LIST sequences, it will then add a monomer from the pool of monomers to this sequence. 
#After adding a monomer to the sequence, this function will update the populations of the monomer and the sequence.


#'1. Pick Sequence'
    original_sequence_ID = pick_sequence(sequences, 'polymerization')
    
#'2. reduce Original_sequence population'
    sequences[original_sequence_ID].pop -= 1
    if sequences[original_sequence_ID].len == 1:
        Parameters.Nmono -=1
         
#'3. Pick Monomer'
    added_monomer_ID = pick_monomer(sequences)
    Parameters.Nmono -= 1
 # Reduce added_monomer population
    sequences[added_monomer_ID].pop -= 1
   
#'4. Add monomer to sequence'
    new_sequence = sequences[original_sequence_ID].seq + sequences[added_monomer_ID].seq

#'5. identify new sequence'
    New_ID = identify_sequence(new_sequence, sequences) #identify_sequence will return the ID of the new sequence, if the new sequence does not exist, it will return False

#'6. Update New Sequence population or created new sequence population'

 # If the new sequence does not exist in the population yet, create it.
    if type(New_ID) == str: 
        make_new_polymer(new_sequence, sequences)
        New_ID = len(sequences)-1
        
  # If the new sequences does exist in the population, increase its population by one.
    elif type(New_ID)== int:
        sequences[New_ID].pop +=1
        if sequences[New_ID].pop > Parameters.max_population:
            Parameters.max_population = sequences[New_ID].pop 
            Parameters.max_pop_ID = New_ID
    if sequences[New_ID].len >= 7:
      Parameters.polymerized_to_replicator  += 1
    Parameters.Ntot -=1         
    return [sequences[original_sequence_ID], sequences[New_ID]]

######################################################################################################################################
def death(sequences):
    """This function will remove a sequence from the list of sequences"""
    sequence_ID= pick_sequence(sequences, 'death')
    sequences[sequence_ID].pop -= 1
    Parameters.Ntot -= 1
    if sequences[sequence_ID].len == 1:
      Parameters.Nmono -=1
    if sequences[sequence_ID].pop < 0:
        print "negative population from death"
        exit()
    return [sequences[sequence_ID]]
    
######################################################################################################################################
def degradation(sequences):
    """ This function chooses an original sequence out of the list sequences, it will then break a point on this sequence """
#' Then this function will break the original sequence at that point, creating two new sequences.
#' After breaking the original sequence, it will update the population of the original sequence, and the two new sequences created.
#'1. Pick Sequence'
    original_sequence_ID = pick_sequence(sequences, 'degradation')
   
#'2. reduce Original_sequence population'
    sequences[original_sequence_ID].pop -= 1
    
#'3. Break sequence'
    new_sequences=hydrolize(sequences[original_sequence_ID].seq)
    
#'4. Identify two new sequences'
    New_ID1= identify_sequence(new_sequences[0], sequences) 
    New_ID2= identify_sequence(new_sequences[1], sequences)
    
#'5. Update New sequences population or create new sequence population'
    # Hydrolysis events create two new sequences.
    "This could be generalized to update the population or create new sequence population of N sequences" 
   
    # If the sequences are the same handle them at the same time
    if new_sequences[0] == new_sequences[1]:
        
        #If the new sequence does not exist, extend the list of sequences
        if type(New_ID1) == str:    
            New_ID1= len(sequences)
            make_new_polymer(new_sequences[0], sequences) 
            sequences[New_ID1].pop +=1
            
        # If the new sequence does exist increase its population
        else:            
            sequences[New_ID1].pop +=2
            if sequences[New_ID1].len == 1:
              Parameters.Nmono+=2
            
    # If the sequences are different handle them one at a time.
    else:
        # First sequence
        if type(New_ID1) == str: #If the new sequence does not exist, extend the list of sequences 
            
            make_new_polymer(new_sequences[0], sequences)
            New_ID1 = len(sequences)-1
        else: # If the new sequence does exist increase its population
            sequences[New_ID1].pop +=1  
            if sequences[New_ID1].len == 1:
              Parameters.Nmono +=1
            if sequences[New_ID1].pop > Parameters.max_population:
                Parameters.max_population = sequences[New_ID1].pop 
                Parameters.max_pop_ID = New_ID1 
            
        # Second Sequence
        if type(New_ID2) == str:   #If the new sequence does not exist, extend the list of sequences 
            make_new_polymer(new_sequences[1], sequences)
            New_ID2 = len(sequences)-1
        else:               #   If the new sequence does exist increase its population
            sequences[New_ID2].pop +=1
            if sequences[New_ID2].len == 1:
              Parameters.Nmono+=1
            if sequences[New_ID2].pop > Parameters.max_population:
                Parameters.max_population = sequences[New_ID2].pop 
                Parameters.max_pop_ID = New_ID2
    Parameters.Ntot +=1             
    return [sequences[original_sequence_ID], sequences[New_ID1], sequences[New_ID2]]

######################################################################################################################################
def replication(sequences):
    
#"This function chooses a sequence out of the list of 'sequences,' it will replicate this sequence by taking all of the constitutant 
#'monomers out of the existing monomer pool. There is noise injected into the replication process, so therefore the new sequence may not
#'be identical to the original sequence. The after replication the population of the new sequence will be updated, and the monomer populations
#'will also be updated. 

#'1. Pick Sequence'
    replicating_ID = pick_sequence(sequences, 'replication')
   
#'2. Identify Replication Parameters'
    if sequences[replicating_ID].len > Parameters.Nmono:
        #Replication not possible
        #print "Replication not possible not enough Monomers"
        return []
        
#'3. Replicate w/  Noise'
    #Determine number of mutations
     # Full 
    N_mutations=0
    for n in range (0,sequences[replicating_ID].len):
        dice_roll=random.random()
        if dice_roll < Parameters.km:
            N_mutations += 1
    
    
    #Mutate original sequence
    mutant=mutate(sequences[replicating_ID].seq, N_mutations)
    required_resources = determine_monomer_abundances(mutant)
    if (required_resources[0] > sequences[0].pop) or (required_resources[1] > sequences[1].pop):
      #print "Replication Not Possible not enough monomers"
      return []
    #Make sure monomer population can support creation of mutant
    else:         
  #'4. Identify new sequence'
      mutant_ID = identify_sequence(mutant, sequences)
      
  #'5. Update New sequence population'
      if type(mutant_ID) == str:
          make_new_polymer(mutant,sequences)
          mutant_ID = len(sequences)-1
      else:
          sequences[mutant_ID].pop +=1
          if sequences[mutant_ID].pop > Parameters.max_population:
              Parameters.max_population = sequences[mutant_ID].pop 
              Parameters.max_pop_ID = mutant_ID
      Parameters.Ntot +=(1 - len(mutant))
      Parameters.Nmono -= len(mutant)
      sequences[0].pop -= required_resources[0]
      sequences[1].pop -= required_resources[1]
      Parameters.replicated_mass += sum(required_resources)
      return [sequences[replicating_ID], sequences[mutant_ID]]    

        
#########################################################################################################################################
#########################################################################################################################################
def pick_sequence(sequences, reaction_type):
 #"This function will choose a sequence out of the LIST of sequences' and returns the ID for that sequence
 # The arguements for this are the LIST sequences, and the STRING reaction type
 # There are 3 reaction types: polymerization, degradation, replication
    
    
    # If The reaction is polymerization
    if reaction_type == 'polymerization':
        
       # Assign a random number between 0 and the total propensity of polymerization
       dice_roll = random.random()*Parameters.Ap_p
       
       # Dummy Variable to index through each sequences propensity of death
       checkpoint = 0.0
       ID=0
       for seq in sequences:
          checkpoint += seq.Ap_p
          if checkpoint > dice_roll:
            ID =seq.ID
            break 

       return ID #Return a found sequence ID
        
    elif reaction_type == 'death':
        # Assign a random number between 0 and the total propensity of death
       dice_roll = random.random()*Parameters.Ap_d
       
       # Dummy Variable to index through each sequences propensity of death
       checkpoint = 0.0
       ID=0
       for seq in sequences:
          checkpoint += seq.Ap_d
          if checkpoint > dice_roll:
            ID =seq.ID
            break 

       return ID #Return a found sequence ID
    
    # If The reaction is degradation
    elif reaction_type == 'degradation':
        
       # Assign a random number between 0 and the total propensity 
       dice_roll = random.random()*Parameters.Ap_h
       # Dummy Variable to index through each sequences propensity 
       checkpoint = 0.0
       ID=0
       for seq in sequences:
          checkpoint += seq.Ap_h
          if checkpoint > dice_roll:
            ID =seq.ID
            break 
       return ID #Return a found sequence ID

    # If The reaction is replication
    elif reaction_type == 'replication':
        
       # Assign a random number between 0 and the total propensity 
       dice_roll = random.random()*Parameters.Ap_r
       # Dummy Variable to index through each sequences propensity
       checkpoint = 0.0
       ID=0
       for seq in sequences:
          checkpoint += seq.Ap_r
          if checkpoint > dice_roll:
            ID =seq.ID
            break 
       return ID #Return a found sequence ID

####################################################################################################################################### 
def pick_monomer(sequences):
# This  function will select a monomer from the pool of all monomers based on their populations, and return that monomers ID
# This function takes as its arguement the LIST of all existing sequences  

  # Pick a random number on the interval of 0 to monomer_population     
  dice_roll = random.random()*Parameters.Nmono
 
    # Dummy Variables to index over
  checkpoint = 0.0
  ID=0
  for seq in sequences:
      checkpoint += seq.pop
      if checkpoint > dice_roll:
        ID =seq.ID
        if seq.len != 1:
          print "pick_monomer not picking monomers!"
          exit()
        break 
  return ID
        
#####################################################################################################################################        
def identify_sequence(new_sequence, sequences):
# This function takes a _sequence (a string), and compares it to the list of existing sequences (objects, Polymers),
#  if this _sequence exists this function will return the ID of the new_sequence. If it does not exist it will return False.
  #import os
  'This should be generalized to accept a list of sequences and return a list of IDs, use sets to find intersections'
  Found = 'False' 
  #print 'I am looking for ' +repr(new_sequence)
  if new_sequence in Parameters.seq_dict:
    #print 'I found ' + repr(new_sequence) + ', its ID is ' 
    Found = Parameters.seq_dict[new_sequence]
    return Found
    #raw_input("Press Enter to continue...")
  else:
    Found = 'False'
    return Found
    #print 'I didnt find it, it must be new'
    #raw_input("Press Enter to continue...")
 # for sequence in sequences:
 #        if len(new_sequence) == sequence.len and new_sequence == sequence.seq:
 #            Found = sequence.ID
            #break 
    
  
 
####################################################################################################################################
def make_new_polymer(new_sequence, sequences):
    """ Takes a new sequence and the list sequences, identifies any functional 
    motifs in the new_sequence and initializes it in the population of sequences"""
    import Parameters
    from Parameters import f_a, f_b, f_c, r_a, r_b, r_c
    # Give it an ID
    New_ID= len(sequences)
    Parameters.max_ID = New_ID
    if len(new_sequence) > Parameters.max_length:
        Parameters.max_length_ID =New_ID
        Parameters.max_length = len(new_sequence)
    Parameters.seq_dict[new_sequence] = New_ID

    # Determine the chemical concentration
    concentrations= determine_monomer_abundances(new_sequence)
    
    # Intialize it in the population of sequences
    sequences.extend([Polymer(New_ID, 1, len(new_sequence), new_sequence, concentrations)])
    sequences[New_ID].Kr = float(Parameters.kr*(1.0+(sigmoid(r_a,r_b,r_c, sequences[New_ID].con[0]))) )
    sequences[New_ID].Kh = float(Parameters.kh*(1.0-(sigmoid(f_a, f_b, f_c, sequences[New_ID].con[1]))))
    sequences[New_ID].t_discovery = Parameters.tau
    # Look for Non-trivial replication motif
    NT_presence=identify_motifs(new_sequence, Parameters.NT_motifs)
    
    # If there is a non-trivial replication motif initialize that and identify the motif
    if sum(NT_presence) > 0:
        sequences[New_ID].NT=True
        sequences[New_ID].NT_motifs= NT_presence
        
        # Look for functional sequences in the non-trivial replicator
        Functional_presence= identify_motifs(new_sequence, Parameters.F_motifs)
        
        if sum(Functional_presence) > 0 :
            sequences[New_ID].F_motifs= Functional_presence
       
####################################################################################################################################
def hydrolize(original_sequence_string):
# This function takes as its arguments the STRING of a sequence, and it will break it into two new STRINGS which it will return in a LIST
    
    # Pick a place to break the string, pick a random number on the interval 0 to len(sequence) round and cast as an int
    dice_roll=int(random.random()*(len(original_sequence_string)-1))
    
    #Create List of new sequences
    new_sequences=[original_sequence_string[:dice_roll+1], original_sequence_string[dice_roll+1:]]
    
    #Return the list of new sequences
    return new_sequences
  
#####################################################################################################################################
def determine_monomer_abundances(sequence):
    """ Takes a string of M monomer types, returns a list of the monomers present """
    
    monomers=Parameters.monomer_species
    m=Parameters.m
    
    #Intialize list of concentrations
    concentrations=[0]*m
    
    
    #for each monomer in the sequence, identify the monomer
    for n in range(1,len(sequence)+1):
        for i in range(0, m):
            if sequence[(n-1):n] == monomers[i]:
                concentrations[i] += 1
    return concentrations  

####################################################################################################################################
def identify_motifs(sequence, motifs):
    ''' Takes a sequence and searches for a list of motifs in that sequence, 
    returns array of integers, corresponding to presence of motifs '''
    
    # Initialize the return list
    motifs_present=[0]*len(motifs)
    
     # Start at the beginning of the sequence 
     # i indicates where to  begin looking at the sequence
    
    
        
        # For all the possible motifs
    for motif in random.sample(motifs, len(motifs)):
        #Length of the current motif
        L=len(motif)
        i=0
        while i < len(sequence)-L:       
            #If the current sequence length is a motif
            if sequence[i:i+L] == motif:
                #print "Sequence piece is motif!"
                n= motifs.index(motif)
                # Record the presence of that motif
                motifs_present[n] += 1
                sequence= sequence[:i] +' '*L +sequence[i+L:]
                i += (L-1)
                # Don't compare the rest of the motifs to that sequence piece
                break 
                
            else:
                 i += 1 
    #print len(sequence)
      
    return motifs_present   
      
#####################################################################################################################################     
def mutate(sequence, N_mutations):
    """Takes a sequence returns a sequence with N mutations"""
    monomer_species=Parameters.monomer_species
    mutation_points = random.sample(range(len(sequence)), N_mutations)
    
    for n in range(0,len(sequence)):
        s=sequence[n]
        intersection= set([n])&set(mutation_points)
        if intersection ==[]:
            if s == monomer_species[0]:
                replacement=monomer_species[0]
                
            elif s == monomer_species[1]:
                replacement=monomer_species[1]
                
        else:
            if s == monomer_species[0]:
                replacement=monomer_species[1]
                

            elif s == monomer_species[1]:
                replacement=monomer_species[0]
                
            
        sequence = sequence[:n]+ replacement +sequence[n+1:]
      
    return sequence
    
    
################################################################################
def sigmoid(alpha,beta,gamma,n):
    """Simple sigmoid function, f(n), returns n for a given set of {alpha,beta,gamma} """
    value=0.0
    value= float(alpha + ((1-alpha)*(n**beta)/(gamma+ float(n)**beta)))
    return value



