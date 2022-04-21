# 💡⚡ TODO - excluding multithreading
#
#             1. Generalization of input            -  ✅ for any input data as CSV file
#                                                      ✅ now optimize it using sklearn - OneHotEncoder or LabelBinarizer
#             2. Make sigmoid functions             -  ✅ rather than hard coding it
#             3. Choice of activation function      -  giving an option to choose the activation function
#             4. Seperate error functions           -  rather than hardcoding the Mean Square Error
#

import math
# import random
import numpy as np
from numpy.random import default_rng
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from scipy.special import expit


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
        if (type(cord1) == int and type(cord2) == int):
            dist = math.pow((cord1 - cord2), 2)
        else:
            for i, j in zip(cord1, cord2):
                dist += math.pow((i - j), 2)
        return math.pow(dist, 0.5)

    def Manhattan(self, cord1, cord2):
        # |x1-y1| + |x2-y2| + |x3-y3| + ...
        dist = 0.0
        if (type(cord1) == int and type(cord2) == int):
            dist = math.fabs(cord1 - cord2)
        else:
            for i, j in zip(cord1, cord2):
                dist += math.fabs(i - j)
        return dist

    def MaximumDistance(self, cord1, cord2):
        # max(|x1-y1|, |x2-y2|, |x3-y3|, ...)
        dist = float('-inf')
        if (type(cord1) == int and type(cord2) == int):
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
        if (type(cord1) == int and type(cord2) == int):
            dist = math.fabs(cord1 - cord2)
        else:
            for i, j in zip(cord1, cord2):
                tempDist = math.fabs(i - j)
                if (tempDist < dist):
                    dist = tempDist
        return dist

    def Chebishev(self, cord1, cord2, exponent_h):
        dist = 0.0
        if (type(cord1) == int and type(cord2) == int):
            dist = math.pow((cord1 - cord2), exponent_h)
        else:
            for i, j in zip(cord1, cord2):
                dist += math.pow((i - j), exponent_h)
        dist = math.pow(dist, (1.0 / exponent_h))
        return dist

    # =========== Distance Calculation Part Ends ============== #

    def __init__(self, dimensions=(8, 5), initialPopSize=100, input_values=([10, 20, 30, 40], [1, 2, 3, 4]),
                 output_values_expected=([0, 0, 1], [0, 1, 0]), iterations=10, gamma=0.001, beta_base=2,
                 alpha=0.2, alpha_damp=0.99, delta=0.05, exponent=2, seed=None):

        """
        Args:
          ⚡ problem (dict): The problem dictionary
            n_iterations (int): maximum number of iterations, default = 10
            initialPopSize (int): number of population size, default = 100
            gamma (float): Light Absorption Coefficient, default = 0.001
            beta_base (float): Attraction Coefficient Base Value, default = 2
            alpha (float): Mutation Coefficient, default = 0.2
            alpha_damp (float): Mutation Coefficient Damp Rate, default = 0.99
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
                w = np.random.random((self.dimension[i + 1], self.dimension[i]))
                W.append(w)
            self.pop.append(W)

        self.init_pop = []  # chromosomes or fireflies
        for W in self.pop:
            chromosome = []
            for w in W:
                chromosome.extend(w.ravel().tolist())
            self.init_pop.append(chromosome)

        # ================ Initial Weights Part Ends ================ #

    # For FireFly Algorithm, Fitness can be found out by calculating the BRIGHTNESS (Mean Square Error in our case)

    def Fitness(self, population):
        # X, Y and pop are used
        total_error = 0.0
        self.fitness = []

        for chromo in population:
            # convert c -> m1, m2, ..., mn
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
                    for yoCount in range(len(yo)):
                        yo[yoCount] = (1 / (1 + math.exp(-yo[yoCount])))

                total_error = 0

                for i in range(len(yo)):
                    total_error += ((yo[i] - y[i]) ** 2)

            self.fitness.append(total_error)

    # ======================= FFA Part =====================#

    def UpdatePosition(self, population):
        for fireflyCount1 in range(len(population)):
            for fireflyCount2 in range(len(population)):
                if(fireflyCount1 != fireflyCount2):
                    if (self.fitness[fireflyCount2] > self.fitness[fireflyCount1]):
                        # case in which 2nd firefly is more brighter
                        # move less brighter firefly towards more brighter one
                        # use the formula for update - x(t+1) = x(t) + beta * e^(-γ * (r^2)) * distance + α * ε

                        distance = self.Eucledian(population[fireflyCount1], population[fireflyCount2])
                        # distance = np.linalg.norm(self._position - better_position)

                        # beta = beta0 * e^(-γ r^m)
                        beta = self.beta_base * math.exp(-self.gamma * (distance ** self.exponent))
                        attractiveness = beta * math.exp(-self.gamma * (distance ** 2))

                        # attractiveness * (xj - xi)
                        beta_firefly1 = attractiveness * (population[fireflyCount2] - population[fireflyCount1])
                        population[fireflyCount1] += (beta_firefly1 + self.alpha * (self.rng.random(0, 1) - 0.5))
                        break

                    elif (self.fitness[fireflyCount2] == self.fitness[fireflyCount1]):
                        # case in which both fireflies have equal brightness
                        # random walk
                        population[fireflyCount1] += (self.alpha * (self.rng.random(0, 1) - 0.5))
                        population[fireflyCount2] += (self.alpha * (self.rng.random(0, 1) - 0.5))
                        break

        return population

    # ==================== FFA definition part ends - iteration included in main function ================ #

    def main(self, fileName="iris"):

        # ================== Input dataset and corresponding output ========================= #

        fileName += ".csv"
        data = pd.read_csv(fileName)

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

        #
        y = LabelBinarizer().fit_transform(data[data.columns[-1]])
        # print(y)

        # ~~~~ encoding ends~~~~#

        for j in range(len(data)):
            output_values_expected.append(y[j])

        for j in range(len(data)):
            b = []
            for i in range(1, len(data.columns) - 1):
                b.append(data[data.columns[i]][j])
            input_values.append(b)

        self.X = input_values[:]
        self.Y = output_values_expected[:]

        # ================== Input Ends ========================== #

        # ================== FFA Methods ========================= #

        # Step 1: Initial Population or Fireflies
        population = self.init_pop
        iterations = 0
        fitness = []

        while (iterations < self.n_iterations):  # Maximum Iteration Count = 100
            print("--------------GENERATION " + str(iterations) + "-----------")
            iterations += 1

            # Step 2: Calculate Fitness
            self.Fitness(population)

            # sorting population based on fitness

            population = [x for y, x in sorted(zip(self.fitness, population))]

            fitness = [x for x, y in sorted(zip(self.fitness, population))]

            # Step 3: Brightness
            population = self.UpdatePosition(population)

            print(fitness[:10])
            # print(population[:10])

            # Step 4: Changing Alpha for each iteration
            self.alpha *= self.alpha_damp



        # print(population[0])  # best set of weights
        print(fitness[0])


a = ffaAnn()  # Parameters - dimensions, initialPopSize, input_value, output_values_expected, iterations,
# elicitation_rate, mutation_rate

# e.g. :-  a =  ffaAnn(dimensions = (8, 5), initialPopSize = 10, input_values = [[10,20,30,40], [1,2,3,4]],
# output_values_expected = [[0, 0, 1], [0, 1, 0]], iterations = 10, elicitation_rate = 0.01, mutation_rate=0.0001)

# If you want to use your own input and output values remove "Input dataset and corresponding output till GA Step1"
# part from main method

a.main("../ANN/iris")  # a.main("filename") where filename is the name of csv file or dataset
# or just use a.main()