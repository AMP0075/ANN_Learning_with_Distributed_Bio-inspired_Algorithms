#💡⚡ TO DO - excluding multithreading 
#
#             1. Generalization of input                                        -  .✅ for any input data as CSV file                                        
#                                                                                  . now optimize it using sklearn - OneHotEncoder or LabelEncoder
#             2. Make sigmoid functions                                         -  rather than hard coding it 
#             3. Choice of activation function                                  -  giving an option to choose the activation function
#             4. Seperate error functions                                       -  rather than hardcoding the Mean Square Error 
#
import math
import random
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class gaAnn():
    def __init__(self, dimensions = (8, 5), initialPopSize = 10, input_values = [[10,20,30,40], [1,2,3,4]], output_values_expected = [[0, 0, 1], [0, 1, 0]], iterations = 10, elicitation_rate = 0.01, mutation_rate=0.0001):
        self.initialPopSize = initialPopSize
        self.allPop_Weights = []
        self.allPopl_Chromosomes = []
        self.allPop_ReceivedOut  = []
        self.allPop_ErrorVal = []
        self.n_iterations = iterations
        self.elicitation_rate = elicitation_rate
        self.mutation_rate = mutation_rate
        self.fitness = []
        
        # input and output
        self.X= input_values[:]
        self.Y = output_values_expected[:]
        
        self.dimensions = dimensions
        self.dimension = [len(self.X[0])]
        
        for i in self.dimensions:
            self.dimension.append(i)
        self.dimension.append(len(self.Y[0]))
        
        #print(self.dimension)
        
        
        

        # ======================================================== #
        # finding initial weights 
            
        self.pop = []                                             # weights
        for g in range(self.initialPopSize):
            W = []
            for i in range(len(self.dimension)-1):
                w = np.random.random((self.dimension[i+1],self.dimension[i]))
                W.append(w)
            self.pop.append(W)
            
        self.init_pop =[]                                          # chromosomes
        for W in self.pop:
            chromosome = []
            for w in W :
                chromosome.extend(w.ravel().tolist())
            self.init_pop.append(chromosome)
        # ======================================================== #
        
        
    def Fitness(self, population):
        # X, Y and pop are used
        self.fitness = []
        for chromo in population:
            # convert c -> m1, m2, ..., mn
            startPos = 0
            m = []
            k1 = 0
            for i in range(len(self.dimension)-1):
                p = self.dimension[i]
                q = self.dimension[i+1]
                k2 = k1 + p * q
                mtemp = chromo[k1:k2]
                m.append(np.reshape(mtemp, (p, q)))
                k1 = k2
                
            for x,y in zip(self.X, self.Y):

                yo = x

                for mCount in range(len(m)):
                    yo = np.dot(yo, m[mCount])
                    for yoCount in range(len(yo)):
                        yo[yoCount] = (1 / (1 + math.exp(-yo[yoCount])))
                
                total_error = 0

                for i in range(len(yo)):
                    total_error += ((yo[i] - y[i])**2)
                    
            self.fitness.append(total_error)      
    
    
    
    
# ======================= GA Part =====================#    
    
    
    def Selection(self, population):
        return population[:10]


    # Crossover
    def Crossover(self, bestPop):
        children = []
        
        for i in range(len(bestPop)):
            
            for j in range(len(bestPop)):
                if(i!=j):
                    child1 = bestPop[i][:]
                    child2 = bestPop[j][:]
                    
                    # Performing Crossover
                    k = random.randint(0,len(bestPop[i]))
                    child1 = child1[:k] + child2[k:]
                    child2 = child1[k:] + child2[:k]
                    children.append(child1)
                    children.append(child2)        
                    
        return children
    
    
    

    # Mutation
    def mutation(self, bestPop):
        mutRate = 1
        temp = self.mutation_rate
        while (int(temp) != 1):
            temp *= 10
            mutRate *= 10
        chance = random.randint(1, int(mutRate) + 1)
        if (chance == mutRate):
            i = random.randint(0, len(bestPop) - 1)    # Selecting a random chromosome from the best population
            # print(i, len(bestPop))
            k = random.randint(0, len(bestPop[i]) - 1) # Selecting a random gene position on the selected chromosome
            t = random.randint(-100, 100)              # Selecting a random weight value - (as per line 178)
            sol = bestPop[i]
            sol[k] = t
        return bestPop

# ==================== GA definition part ends - iteration included in main function ================ #

    def main(self, fileName = "iris"):
        
        # ================== Input dataset and corresponding output ========================= #
        
        fileName += ".csv"
        data = pd.read_csv(fileName)

        classes = []
        output_values_expected = []
        input_values = []

        #manual one hot encoding
        # finding all classes and one hot encoding
        c=0
        for i in data[data.columns[-1]].unique():

            s = len(data[data.columns[-1]].unique())
            k = [0 for i in range(s)]
            if(i in classes):
                b = classes.index(i)
                k[b] = 1
            else:
                k[c] = 1
            classes.append([i,k])
            c+=1

        # output values as integers
        for i in range(len(data[data.columns[-1]])):
            for j in range(len(classes)):
                if(classes[j][0] ==  data[data.columns[-1]][i]):
                    output_values_expected.append(classes[j][1])
                    break

        input_values = []
        for j in range(150):
            b = []
            for i in range(1, len(data.columns)-1):
                b.append(data[data.columns[i]][j])
            input_values.append(b)
            
        
        self.X= input_values[:]
        self.Y = output_values_expected[:]
        
        # ================== GA Methods ========================= #
        
        #Step 1: Initial Population
        population = self.init_pop
        iterations = 0
        values = []
        
        while(iterations < self.n_iterations):                         # Maximum Iteration Count = 100
            print("--------------GENERATION "+str(iterations)+"-----------")
            iterations+=1
            
            # Step 2: Calculate Fitness
            self.Fitness(population)
            
            # sorting population based on fitness
                        
            population = [x for y, x in sorted(zip(self.fitness, population))]
            
            fitness = [x for x, y in sorted(zip(self.fitness, population))]

            #print(fitness[:10])        
            #print(population[:10])
            
            
            # Step 3: Selection
            bestPop = self.Selection(population)
            
            # Step 4: Crossover
            children = self.Crossover(bestPop)
            
            # Step 5: Mutation
            children = self.mutation(children)
            
            # Elitism
            elitRate = self.elicitation_rate
            temp1 = int(elitRate*len(bestPop))
            bestPop = bestPop[:temp1]
            temp2 = int(elitRate*len(bestPop))+len(children)
            population = children + bestPop
            population = population[:temp2]
        
        #print(population[0])
        #print(fitness[0])
        
        
a=gaAnn()

a.main() #say a.main(iris)
