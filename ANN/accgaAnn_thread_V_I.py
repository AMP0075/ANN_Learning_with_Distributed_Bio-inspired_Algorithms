import math
import random
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from scipy.special import expit

from threading import Thread
import time

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        # print(type(self._target))
        # print(self._target)
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class InputData():
    # input and output
        # ================== Input dataset and corresponding output ========================= #
        def __init__(self,fileName = "iris"):
            self.fileName = fileName
            self.fileName += ".csv"
            data = pd.read_csv(self.fileName)

            output_values_expected = []
            input_values = []

            # ~~~~ encoding ~~~~#

            # labelencoder = LabelEncoder()
            # data[data.columns[-1]] = labelencoder.fit_transform(data[data.columns[-1]])

            # one hot encoding - for multi-column
            # enc = OneHotEncoder(handle_unknown='ignore')
            # combinedData = np.vstack((data[data.columns[-2]], data[data.columns[-1]])).T
            # print(combinedData)
            # y = enc.fit_transform(combinedData).toarray()
            # y = OneHotEncoder().fit_transform(combinedData).toarray()


            y = LabelBinarizer().fit_transform(data[data.columns[-1]])
            # print(y)

            # ~~~~ encoding ends~~~~#

            for j in range(150):
                output_values_expected.append(y[j])

            # print(output_values_expected)

            for j in range(150):
                b = []
                for i in range(1, len(data.columns) - 1):
                    b.append(data[data.columns[i]][j])
                input_values.append(b)

            self.X = input_values[:]
            self.Y = output_values_expected[:]
        def main(self):
            return (self.X, self.Y)
    

class gaAnn():

    # ================== Activation Functions ================ #

    # accepts a vector or list and returns a list after performing corresponding function on all elements

    def sigmoid(self, vectorSig):
        '''returns 1/(1+exp(-x)), where the output values lies between zero and one'''
        sig = expit(vectorSig)
        return sig

    def binaryStep(self, x):
        ''' It returns '0' is the input is less then zero otherwise it returns one '''
        return np.heaviside(x, 1)

    def linear(self, x):
        ''' y = f(x) It returns the input as it is'''
        return x

    def tanh(self, x):
        ''' It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1'''
        return np.tanh(x)

    def relu(self, x):  # Rectified Linear Unit
        ''' It returns zero if the input is less than zero otherwise it returns the given input'''
        x1 = []
        for i in x:
            if i < 0:
                x1.append(0)
            else:
                x1.append(i)

        return x1

    def leakyRelu(self, x):
        ''' It returns zero if the input is less than zero otherwise it returns the given input'''
        x1 = []
        for i in x:
            if i < 0:
                x1.append((0.01 * i))
            else:
                x1.append(i)

        return x1

    def parametricRelu(self, a, x):
        ''' It returns zero if the input is less than zero otherwise it returns the given input'''
        x1 = []
        for i in x:
            if i < 0:
                x1.append((a * i))
            else:
                x1.append(i)

        return x1

    def softmax(self, x):
        ''' Compute softmax values for each sets of scores in x'''
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    # ============ Activation Functions Part Ends ============= #

    def __init__(self, dimensions=(8, 5), initialPopSize=100, iterations=10, elicitation_rate=0.01,
                 mutation_rate=0.001, m = 10, bestCount = 50,
                input_values=[] , output_values_expected=[]):
        self.initialPopSize = initialPopSize
        self.allPop_Weights = []
        self.allPopl_Chromosomes = []
        self.allPop_ReceivedOut = []
        self.allPop_ErrorVal = []
        self.n_iterations = iterations
        self.elicitation_rate = elicitation_rate
        self.mutation_rate = mutation_rate
        self.fitness = []
        
        self.distribution_factor = m
        self.selection_var = bestCount

        # input and output

        self.X = input_values[:]
        self.Y = output_values_expected[:]

        self.dimensions = dimensions
        self.dimension = [len(self.X[0])]

        for i in self.dimensions:
            self.dimension.append(i)
        self.dimension.append(len(self.Y[0]))

        # print("Dimension of each layer : ", self.dimension)

        # ================ Finding Initial Weights ================ #

        self.pop = []  # weights
        for g in range(self.initialPopSize):
            W = []
            for i in range(len(self.dimension) - 1):
                w = np.random.randint(-100, 100, (self.dimension[i + 1], self.dimension[i]))
                W.append(w)
            self.pop.append(W)

        self.init_pop = []  # chromosomes
        for W in self.pop:
            chromosome = []
            for w in W:
                chromosome.extend(w.ravel().tolist())
            self.init_pop.append(chromosome)

        # ================ Initial Weights Part Ends ================ #

    def mean_square_error(self, expected, predicted):
        total_error = 0.0
        for i in range(len(predicted)):
            total_error += ((predicted[i] - expected[i]) ** 2)
        return (-total_error)

    def Fitness(self, population, threadcall = 0):
        # X, Y and pop are used
        if(threadcall):
            fitnesss = []
        else:
            self.fitness = []
        for chromo in population:
            # convert c -> m1, m2, ..., mn
            total_error = 0.0
            m = []
            k1 = 0
            for i in range(len(self.dimension) - 1):
                p = self.dimension[i]
                q = self.dimension[i + 1]
                k2 = k1 + p * q
                mtemp = chromo[k1:k2]
                m.append(np.reshape(mtemp, (p, q)))
                k1 = k2

            for x, y in zip(self.X, self.Y):

                yo = x

                for mCount in range(len(m)):
                    yo = np.dot(yo, m[mCount])
                    yo = self.sigmoid(yo)

                total_error += self.mean_square_error(yo, y)

            if(threadcall):
                fitnesss.append(total_error)
            else:
                self.fitness.append(total_error)
        if(threadcall):
            return(fitnesss)
            
        # print(len(self.fitness))
        
    # ========== Threading Fitness Calculation =========== #
    def parallel_fitness(self, population):
        fitness_par = []
        m = self.distribution_factor
        chunk_size = len(population)/m
        result = [0] * m

        threads = []
        start = 0
        end = start + int(chunk_size)

        for i in range(m):
            b = population[start:end]
            process = ThreadWithReturnValue(target=self.Fitness, args=[b, 1])
            process.start()
            threads.append(process)
            start = end
            end = start + int(chunk_size)
        i = 0
        for process in threads:
            result[i] = process.join()
            i += 1
        for i in result:
            fitness_par.extend(i)
        return fitness_par

    # ======================= GA Part =====================#

    
        
    def Selection(self, population):
        return population[:self.selection_var]

    # Crossover
    def Crossover(self, bestPop):
        children = []
        for i in range(len(bestPop)):
            for j in range(len(bestPop)):
                if (i != j):
                    child1 = bestPop[i][:]
                    child2 = bestPop[j][:]

                    # Performing Crossover
                    k = random.randint(0, len(bestPop[i]))
                    child1 = child1[:k] + child2[k:]
                    child2 = child1[k:] + child2[:k]
                    children.append(child1)
                    children.append(child2)
        return children
    
    def parallel_crossover(self, bestPop):
        print("Best Pop : ",len(bestPop))
        children_par = []
        m = self.distribution_factor
        chunk_size = len(bestPop)/m
        results = [0] * m

        threads = []
        start = 0
        end = start + int(chunk_size)
        
        for ip in range(m):
            b = bestPop[start:end]
            processs = ThreadWithReturnValue(target=self.Crossover, args=[b])
            processs.start()
            threads.append(processs)
            start = end
            end = start + int(chunk_size)
        ip = 0
        for processs in threads:
            results[ip] = processs.join()
            ip += 1
        for i in results:
            children_par.extend(i)
        return children_par
        
        
    
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

    def main(self):

        # ================== GA Methods ========================= #

        # Step 1: Initial Population
        population = self.init_pop
        iterations = 0
        fitness = []
        best_per_gen = []
        all_gen_best_weight = []
        
        while (iterations < self.n_iterations):  # Maximum Iteration Count = 100
            print("--------------GENERATION " + str(iterations) + "-----------")
            iterations += 1

            # Step 2: Calculate Fitness
            self.fitness = self.parallel_fitness(population)
            
            #print(len(self.fitness))

            # sorting population based on fitness

            sorted_population = [x for y, x in sorted(zip(self.fitness, population))]

            fitness = [x for x, y in sorted(zip(self.fitness, population))]
            
            self.fitness = fitness[:]
            population = sorted_population[:]
            
            best_per_gen.append(-fitness[-1])
            
            all_gen_best_weight.append(population[-1])
            
            
            #print(len(population))

            # print(fitness[:10])
            # print(population[:10])

            # Step 3: Selection
            bestPop = self.Selection(population)[:]

            # Step 4: Crossover
            children = self.Crossover(bestPop)
            

            # Step 5: Mutation
            children = self.mutation(children)

            # Elitism
            elitRate = self.elicitation_rate
            temp1 = elitRate * len(children)
            temp1 = int(temp1)
            next_gen = sorted_population+children[:temp1]
            population = next_gen[:]
         
            
        # print("BEST SET OF WEIGHTS : \n", population[-1])
        print("Fitness : ", -fitness[-1])
        return (-fitness[-1], best_per_gen, sorted_population[-1], self.dimension, all_gen_best_weight)
        # print(fitness[:101])
