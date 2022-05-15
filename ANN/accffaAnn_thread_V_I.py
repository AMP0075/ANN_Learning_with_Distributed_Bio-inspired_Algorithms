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

# noinspection SpellCheckingInspection

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

        for j in range(len(data)):
            output_values_expected.append(y[j])

        # print(output_values_expected)

        for j in range(len(data)):
            b = []
            for i in range(1, len(data.columns) - 1):
                b.append(data[data.columns[i]][j])
            input_values.append(b)

        self.X = input_values[:]
        self.Y = output_values_expected[:]

    def main(self):
        return (self.X, self.Y)


class ffaAnn():

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

    # ================= Distance Calculation ================== #

    def Eucledian(self, cord1, cord2):
        dist = 0.0
        if ((type(cord1) == int and type(cord2) == int) or ((type(cord1) == float and type(cord2) == float))):
            dist = math.pow((cord1 - cord2), 2)
        else:
            for i, j in zip(cord1, cord2):
                dist += math.pow((i - j), 2)
        return math.pow(dist, 0.5)

    def Manhattan(self, cord1, cord2):
        # |x1-y1| + |x2-y2| + |x3-y3| + ...
        dist = 0.0
        if ((type(cord1) == int and type(cord2) == int) or ((type(cord1) == float and type(cord2) == float))):
            dist = math.fabs(cord1 - cord2)
        else:
            for i, j in zip(cord1, cord2):
                dist += math.fabs(i - j)
        return dist

    def MaximumDistance(self, cord1, cord2):
        # max(|x1-y1|, |x2-y2|, |x3-y3|, ...)
        dist = float('-inf')
        if ((type(cord1) == int and type(cord2) == int) or ((type(cord1) == float and type(cord2) == float))):
            dist = math.fabs(cord1 - cord2)
        else:
            for i, j in zip(cord1, cord2):
                tempDist = math.fabs(i - j)
                if (tempDist > dist):
                    dist = tempDist
        return dist

    def MinimumDistance(self, cord1, cord2):
        # min(|x1-y1|, |x2-y2|, |x3-y3|, ...)
        dist = float('inf')
        if ((type(cord1) == int and type(cord2) == int) or ((type(cord1) == float and type(cord2) == float))):
            dist = math.fabs(cord1 - cord2)
        else:
            for i, j in zip(cord1, cord2):
                tempDist = math.fabs(i - j)
                if (tempDist < dist):
                    dist = tempDist
        return dist

    def Chebishev(self, cord1, cord2, exponent_h):
        dist = 0.0
        if ((type(cord1) == int and type(cord2) == int) or ((type(cord1) == float and type(cord2) == float))):
            dist = math.pow((cord1 - cord2), exponent_h)
        else:
            for i, j in zip(cord1, cord2):
                dist += math.pow((i - j), exponent_h)
        dist = math.pow(dist, (1.0 / exponent_h))
        return dist

    # =========== Distance Calculation Part Ends ============== #

    def __init__(self, dimensions=(8, 5),
                 initialPopSize=10, iterations=10, gamma=0.001, beta_base=2.5,
                 alpha=0.2, alpha_damp=0.97, delta=0.05, exponent=2, m=5,
                 input_values=[], output_values_expected=[], seed=None):

        """
        Args:
            n_iterations (int): maximum number of iterations, default = 10
            initialPopSize (int): number of population size, default = 100
            gamma (float): Light Absorption Coefficient, default = 0.001
            beta_base (float): Attraction Coefficient Base Value, default = 2
            alpha (float): Mutation Coefficient, default = 0.2
            alpha_damp (float): Mutation Coefficient Damp Rate, default = 0.97
            delta (float): Mutation Step Size, default = 0.05
            exponent (int): Exponent (m in the paper), default = 2
        """

        self.initialPopSize = initialPopSize
        self.allPop_Weights = []
        self.allPopl_Chromosomes = []
        self.allPop_ReceivedOut = []
        self.allPop_ErrorVal = []
        self.n_iterations = iterations

        self.gamma = gamma  # (0, 1.0)
        self.beta_base = beta_base  # (0, 3.0)
        self.alpha = alpha  # (0, 1.0)
        self.alpha_damp = alpha_damp  # (0, 1.0)
        self.delta = delta  # (0, 1.0)
        self.exponent = exponent  # [2, 4]
        self.rng = default_rng(seed)

        self.fitness = []

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

        # ================ Finding Initial Weights ================ #

        self.pop = []  # weights
        for g in range(self.initialPopSize):
            W = []
            for i in range(len(self.dimension) - 1):
                w = np.random.randint(-90, 90, (self.dimension[i + 1], self.dimension[i]))
                W.append(w)
            self.pop.append(W)

        self.init_pop = []  # chromosomes
        for W in self.pop:
            chromosome = []
            for w in W:
                chromosome.extend(w.ravel().tolist())
            self.init_pop.append(chromosome)

        # ================ Initial Weights Part Ends ================ #

    # For FireFly Algorithm, Fitness can be found out by calculating the BRIGHTNESS (Mean Square Error in our case)

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
            total_error = 0
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

                for i in range(len(yo)):
                    total_error += self.mean_square_error(yo, y)

            fitness.append(total_error)

        return fitness

    def parallel_fitness(self, population):
        fitness_par = []
        m = self.distribution_factor
        chunk_size = len(population) / m
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

    # ======================= FFA Part =====================#

    def UpdatePosition(self, population):
        population = np.array(population)
        # print("population before update - ",population)
        for fireflyCount1 in range(len(population)):
            for fireflyCount2 in range(1 + fireflyCount1, len(population)):
                if (fireflyCount1 != fireflyCount2):
                    if (self.fitness[fireflyCount2] > self.fitness[fireflyCount1]):
                        # case in which 2nd firefly is more brighter
                        # move less brighter firefly towards more brighter one
                        # use the formula for update - x(t+1) = x(t) + beta * e^(-γ * (r^2)) * distance + α * ε

                        distance = self.Eucledian(population[fireflyCount1], population[fireflyCount2])
                        # distance = np.linalg.norm(self._position - better_position)

                        # beta = beta0 * e^(-γ r^m)
                        beta = self.beta_base * math.exp(-self.gamma * (distance))
                        attractiveness = beta * math.exp(-self.gamma * (distance))

                        # print(beta)
                        # print(attractiveness)

                        # attractiveness * (xj - xi)
                        array1 = np.array(population[fireflyCount2])
                        array2 = np.array(population[fireflyCount1])
                        subtracted_array = np.subtract(array1, array2)

                        beta_firefly1 = np.dot(attractiveness, subtracted_array)
                        population[fireflyCount1] = np.add(population[fireflyCount1],
                                                           (beta_firefly1 + self.alpha * (random.random() - 0.5)))
                        break

                    elif (self.fitness[fireflyCount2] == self.fitness[fireflyCount1]):
                        # case in which both fireflies have equal brightness
                        # random walk
                        print("hi")
                        population[fireflyCount1] = np.add(population[fireflyCount1],
                                                           (self.alpha * (random.random() + 0.5)))
                        population[fireflyCount2] = np.add(population[fireflyCount2],
                                                           (self.alpha * (random.random() + 0.5)))
                        break
        # print("population after update- ",population)
        return population

    # if we split the data into multiple parts then fireflies in one chunk cannot be attracted to fireflies in another chunk,
    # so those position updates wont be possible
    # but we can assume that since attractive power decreases with distance so those flies won't be attracted much

    # while running the code multiple times, it was found that the fitness improves on doing so

    def parallel_updatePosition(self, population):
        population_all = []
        m = self.distribution_factor
        chunk_size = len(population) / m
        result = [0] * m

        threads = []
        start = 0
        end = start + int(chunk_size)

        for i in range(m):
            b = population[start:end]
            process = ThreadWithReturnValue(target=self.UpdatePosition, args=[b])
            process.start()
            threads.append(process)
            start = end
            end = start + int(chunk_size)
        i = 0
        for process in threads:
            result[i] = process.join()
            i += 1
        for i in result:
            population_all.extend(i.tolist())
        return population_all

    # ==================== FFA definition part ends - iteration included in main function ================ #

    # ==================== FFA definition part ends - iteration included in main function ================ #

    def main(self):

        # ================== FFA Methods ========================= #

        # Step 1: Initial Population or Fireflies
        population = self.init_pop
        iterations = 0
        fitness = []
        best_per_gen = []
        all_gen_best_weight = []

        while (iterations < self.n_iterations):  # Maximum Iteration Count = 100
            print("--------------GENERATION " + str(iterations) + "-----------")
            iterations += 1

            # Step 2: Calculate Fitness
            self.fitness = self.parallel_fitness(population)[:]

            # sorting population based on fitness

            sorted_population = [x for y, x in sorted(zip(self.fitness, population))]

            fitness = [x for x, y in sorted(zip(self.fitness, population))][:]
            
            if(iterations==1):
                print("Initial worst fitness = ", -fitness[0], "\n\n Initial best fitness = ", -fitness[-1])
            

            self.fitness = fitness[:]

            best_per_gen.append(-self.fitness[-1])
            
            all_gen_best_weight.append(sorted_population[-1])

            # Step 3: Brightness
            population = self.parallel_updatePosition(sorted_population)[:]

            # print(fitness[:10])
            # print(population[:10])

            # Step 4: Changing Alpha for each iteration
            self.alpha *= self.alpha_damp

        sorted_population = [x for y, x in sorted(zip(self.fitness, population))]

        fitness = [x for x, y in sorted(zip(self.fitness, population))]

        # print(sorted_population[-1])
        print("Fitness : ", -fitness[-1])
        return (-fitness[-1], best_per_gen, sorted_population[-1], self.dimension, all_gen_best_weight)
