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
from numpy.random import default_rng

from threading import Thread
import time


class InputData():
    # input and output
    # ================== Input dataset and corresponding output ========================= #
    def __init__(self, fileName="iris"):
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


class psoAnn():

    def sigmoid(self, vectorSig):
        '''returns 1/(1+exp(-x)), where the output values lies between zero and one'''
        sig = expit(vectorSig)
        return sig
    def Eucledian(self, cord1, cord2):
        dist = 0.0
        if ((type(cord1) == int and type(cord2) == int) or ((type(cord1) == float and type(cord2) == float))):
            dist = math.pow((cord1 - cord2), 2)
        else:
            for i, j in zip(cord1, cord2):
                dist += math.pow((i - j), 2)
        return math.pow(dist, 0.5)

    # =========== Distance Calculation Part Ends ============== #

    def __init__(self, dimensions=(8, 5),
                 initialPopSize=10, particles = 20, iterations=10, c1 = 0.1, c2 = 0.1, w = 0.8,
                 m = 5, input_values=[], output_values_expected=[]):

        """
        Args:
          allbest
          psobest
          n_particles : number of particles
          xbest : best particle position
          gbest : best group position
          c1, c2 : two positive constants
          r1, r2 : two random parameters within [0,1]
        """

        self.initialPopSize = initialPopSize
        self.allPop_Weights = []
        self.allPopl_Chromosomes = []
        self.allPopl_Velocity = []
        self.allPop_ReceivedOut = []
        self.allPop_ErrorVal = []
        self.n_iterations = iterations

        self.n_particles = particles
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.pbest = []

        self.fitness = []
        self.gBest = float('-inf')

        self.distribution_factor = m
        
        # input and output
        self.X = input_values[:]
        self.Y = output_values_expected[:]

        self.dimensions = dimensions
        self.dimension = [len(self.X[0])]

        for i in self.dimensions:
            self.dimension.append(i)
        self.dimension.append(len(self.Y[0]))

        # print(self.dimension)

        # ======== Finding Initial Weights and Velocity ========= #

        self.pop = []  # weights
        for g in range(self.initialPopSize):
            W = []
            for i in range(len(self.dimension) - 1):
                w = np.random.random((self.dimension[i + 1], self.dimension[i]))
                W.append(w)
            self.pop.append(W)


        self.init_pop = []  # chromosomes or particles
        for W in self.pop:
            chromosome = []
            for w in W:
                chromosome.extend(w.ravel().tolist())
            self.init_pop.append(chromosome)


        velocitys = []  # velocity of particles
        for g in range(self.initialPopSize):
            W = []
            for i in range(len(self.dimension) - 1):
                w = np.random.randint(-100, 100,(self.dimension[i + 1], self.dimension[i]))
                #w=np.dot(w,(0.02 * random.randrange(-1, 2, 2)))
                W.append(w)
            velocitys.append(W)

        self.velocity = []
        for W in velocitys:
            chromosome = []
            for w in W:
                chromosome.extend(w.ravel().tolist())
            self.velocity.append(chromosome)

        # self.pbest = np.copy(self.pop)




    # ================ Initial Weights Part Ends ================ #

    def mean_square_error(self, expected, predicted):
        total_error = 0.0
        for i in range(len(predicted)):
            total_error += ((predicted[i] - expected[i]) ** 2)
        return (-total_error)

    def Fitness(self, population):
        # X, Y and pop are used
        fitness = []
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

            fitness.append(total_error)

        return fitness

        # print(len(self.fitness))

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
            process = ThreadWithReturnValue(target=self.Fitness, args=[b])
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

    # ======================= PSO Part =====================#

    def updateVelocityPosition(self, population, velocity, start):
        r1, r2 = np.random.rand(2)
        velocity = np.array(velocity)
        population = np.array(population)
        self.pBest = np.array(self.pBest)
        self.gBest = np.array(self.gBest)
        for i in range(len(velocity)):
            velocity[i] = np.dot(self.w, velocity[i]) + \
                            np.dot((self.c1 * r1), np.subtract(self.pBest[start+i], population[i])) + \
                            np.dot((self.c2 * r2), np.subtract(self.gBest, population[i]))             
            population[i] = np.add(population[i], velocity[i])
        return (population, velocity)

    # since we use self.pbest and self.gbest so we need to code accordingly
    def parallel_update_velocity_position(self, population, velocity):
        all_population = []
        all_velocity = []
        m = self.distribution_factor
        chunk_size = len(velocity)/m
        result = [0] * m

        threads = []
        start = 0
        end = start + int(chunk_size)
        for i in range(m):
            b = population[start:end][:]
            c = velocity[start:end][:]
            process = ThreadWithReturnValue(target=self.updateVelocityPosition, args=[b, c, start])
            process.start()
            threads.append(process)
            start = end
            end = start + int(chunk_size)
        i = 0
        for process in threads:
            result[i] = process.join()
            i += 1
            
        for i in result:
            all_population.extend(i[0])
            all_velocity.extend(i[1])
        return (all_population, all_velocity)


    def swarmBestUpdate(self, population_x, fitness_x, fitness_g):
        for i in range(len(population_x)):
            if(fitness_x[i] > self.fitness[i]):
                self.pBest[i] = np.copy(population_x[i])
                self.fitness[i] = self.Fitness([self.pBest[i]])
                if(self.fitness[i] > fitness_g):
                    self.gBest = np.copy(self.pBest[i])
                    fitness_g = self.fitness[i]
        return fitness_g

    # ==================== PSO definition part ends - iteration included in main function ================ #

    def main(self):

        # ================== PSO Methods ========================= #

        # Step 1: Initial Population or Particles and pBest
        population = self.init_pop

        self.pBest = np.copy(population)

        iterations = 0
        
        best_per_gen = []
        
        # Step 2: Calculate Fitness and find gBest
        self.fitness = self.parallel_fitness(self.pBest)

        population_g = [x for y, x in sorted(zip(self.fitness, population))]

        fitness_g = [x for x, y in sorted(zip(self.fitness, population))]

        self.gBest = np.copy(population_g[-1])
        fitness_g = fitness_g[-1]

        velocity = self.velocity

        population_x = np.copy(population)
        velocity_x = np.copy(velocity)
        fitness_x = np.copy(self.fitness)


        while (iterations < self.n_iterations):  # Maximum Iteration Count = 100
            print("--------------GENERATION " + str(iterations) + "-----------")
            iterations += 1


            # Step 4: Update Position and Velocity
            population_x, velocity_x = self.parallel_update_velocity_position(population_x, velocity_x)
            #print("pop ", population_x)

            # print(population_x[0][0])
            # print(velocity_x[0][0])        

            # Step 5: Update particle's best known position and swarm's best known position
            fitness_x = self.parallel_fitness(population_x)
            fitness_g = self.swarmBestUpdate(population_x, fitness_x, fitness_g)

            print(fitness_x[-1])
            # print(fitness_g)
            if(type(fitness_g) == type([])):
                best_per_gen.append(-(fitness_g[0]))
            else:
                best_per_gen.append(-fitness_g)



        print("Global : ", -fitness_g[0]) #global best
        #print(self.gBest) #global best weights


        #print("Particle : ", self.fitness)#particle best fitness
        #print(self.pBest) #particle best  weights
        return(fitness_g, best_per_gen, self.gBest, self.dimension)
