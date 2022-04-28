import random
import string


def randomStringGenerator(N):
    t = ''.join(random.choices(string.ascii_uppercase, k = N))      # generates a random string of size N
    return t

  
# Initial Population
def initialPopulation(sizePop):                                     # size of population => initial population size [Number of strings]
    population = []
    
    sizeString = 10                                                 # size of string = 10 (e.g. :-  ACSOIKNSNV)
        
    for i in range(sizePop):
        candSol = randomStringGenerator(sizeString)                 # generating all strings for the initial population
        population.append(candSol)
        
    return population


  
# Calculate Fitness
def Fitness(population):
    fitness = []                                                    # stores fitness of the entire population
    
      
    
    for candSol in population:
        ascii_values = [ord(character) for character in candSol]    # e.g. :- [65, 67, 83, 79, 73, 75, 78, 83, 78, 86] for ACSOIKNSNV
        sumDiffAscii = 0
        
        maxv = 25*9                                                 # maximum difference possible between each adjacent positions = ord(Z)-ord(A)
                                                                    # i.e., 90-65 = 25 
                                                                    # for a string of size 10 - there can be 9 adjacent pairs 
                                                                    # i.e, total maximum difference possible = 25*9 [e.g. - AZAZAZAZAZ]

        minv = 0                                                    # minimum difference possible between each adjacent position [when both are same]
                                                                    # e.g - ord(A)-ord(A) = 0
                                                                    # i.e, total minimum difference possible = 0*9 = 0 [e.g. - AAAAAAAAAA]
        
        
        # Calculating the fitness (in our case, the total difference for a string) 
        for i in range(len(candSol)-1):
            diff = abs( ascii_values[i] - ascii_values[i+1] )
            sumDiffAscii += diff
        
        # calculating the normalized fitness  
        sumDiffAscii = abs(sumDiffAscii - 100)
        fitness_val = 1-((sumDiffAscii - minv)/(maxv - minv))      # fitness = |(difference - min) / (max - min)|
                                                                   # fitness_val ∈ [0,1]  (some value equal to or between 0 and 1)
        fitness.append(fitness_val)
        
    return fitness


def Fitn(population):
    fitness = []                                                    # stores fitness of the entire population
    
      
    
    for candSol in population:
        ascii_values = [ord(character) for character in candSol]    # e.g. :- [65, 67, 83, 79, 73, 75, 78, 83, 78, 86] for ACSOIKNSNV
        sumDiffAscii = 0
        
        maxv = 25*9                                                 # maximum difference possible between each adjacent positions = ord(Z)-ord(A)
                                                                    # i.e., 90-65 = 25 
                                                                    # for a string of size 10 - there can be 9 adjacent pairs 
                                                                    # i.e, total maximum difference possible = 25*9 [e.g. - AZAZAZAZAZ]

        minv = 0                                                    # minimum difference possible between each adjacent position [when both are same]
                                                                    # e.g - ord(A)-ord(A) = 0
                                                                    # i.e, total minimum difference possible = 0*9 = 0 [e.g. - AAAAAAAAAA]
        
        
        # Calculating the fitness (in our case, the total difference for a string) 
        for i in range(len(candSol)-1):
            diff = abs( ascii_values[i] - ascii_values[i+1] )
            sumDiffAscii += diff
        
        # calculating the normalized fitness  
        sumDiffAscii = abs(sumDiffAscii - 100)
        fitness_val = 1-((sumDiffAscii - minv)/(maxv - minv))      # fitness = |(difference - min) / (max - min)|
                                                                   # fitness_val ∈ [0,1]  (some value equal to or between 0 and 1)
        
        
    return fitness_val
    
"""# Function to generate the new seeds in each generation
# Based on their previous fitness
def generate_population(population):
        new_seeds = [] # Initiate list of new seeds
        fitness = [] # Initiate to store the value of fitness for population in the function
        maxfit = 0.55
        # Based on how many new children each previous fitness parent produces
        
        fitness=Fitn(population)
        for candSol in population: # this loop will run for each individual in previous population
            #we generate the new seeds in a way that the individuals in pop above a suitable fitness can only produce seeds
            if(fitness[i]>maxfit):          
               # Append new seeds to the population
               new_seeds[i].append((candSol))
           #to population , adding the off spring
            population = population + new_seeds #(need to calculate this one's fitness)
         Return new seeds from the function
        return population"""

# Selection
def Selection(population):
    return population[:10]                                          # we return the top ten elements or parents as the best population





   
    
    
    
# Taking individual chromosomes to fitness level
from matplotlib import pyplot as plt
def main():
    sizePop = 2
    target = 100   # The difference we wish to acheive
    iterations = 0
    sigma_init = 0.5
    sigma_final = 0.001
    # Step 1: Generating the initial population
    population = initialPopulation(sizePop)
    
    values = []                                                    # Will contain the final set of values
    
    while(iterations < 10):                                       # Maximum Iteration Count = 100
        print("--------------GENERATION "+str(iterations)+"-----------")
        iterations+=1
        sigma_init = 0.5
        sigma_final = 0.001
        sigma = (100-iterations) * (sigma_init - sigma_final)*0.01
        # Step 2: Calculate Fitness
        fitness = Fitness(population)
        
        # sorting population based on fitness
        
        population = [x for y, x in sorted(zip(fitness, population), reverse=True)]
        
        fitness = [x for x, y in sorted(zip(fitness, population), reverse=True)]

        print(fitness[:10])        
        print(population[:10])
        bestcost = fitness[0]
        minseed = 0
        worstcost = fitness[-1]
        maxseed = 100
        q=[]
        l=[]
        for i in range(sizePop):
          ratio = (fitness[i] - worstcost)/(bestcost - worstcost);
          S = round(minseed + ((maxseed - minseed)*ratio))
          for j in range(S):
            NewSolutionPosition = randomStringGenerator(10)
            q.append(NewSolutionPosition)
            #fitness.append(
            l.append(Fitn(NewSolutionPosition))
            #print(fitness)
            #print(population)
        population = q + population
        fitness = fitness + l
        population = [x for y, x in sorted(zip(fitness, population), reverse=True)]
        fitness =  [x for x, y in sorted(zip(fitness, population), reverse = True)]
    if(len(population)>2):
        population = population[:2]
        fitness = fitness[:2]
        print(population)
        print(fitness)  
        
        #ascii_values = [ord(character) for character in population[0]]
        #sumDiffAscii = 0
        #for i in range(9):
         #   diff = abs( ascii_values[i] - ascii_values[i+1] )
          #  sumDiffAscii += diff
        #values.append(fitness[0])
        
        #if(abs(sumDiffAscii-target)==0):
        #    print('Found')
        #    print(population[0])
        #    break
        
        
        
    #print(population[0])
    #ascii_values = [ord(character) for character in population[0]]
    #sumDiffAscii = 0
    #for i in range(len(population[0])-1):
    #    diff = abs( ascii_values[i] - ascii_values[i+1] )
    #    sumDiffAscii += diff   
    #print(sumDiffAscii)
    ascii_values = [ord(character) for character in population[0]]
    sumDiffAscii = 0
    for i in range(9):
        diff = abs( ascii_values[i] - ascii_values[i+1] )
        sumDiffAscii += diff
    values.append(fitness[0])
    
    
    
main()
