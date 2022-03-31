import random
import math

def matrix_mul(matrixA, matrixB):
    """
    Function to create a matrix resulting from multiplication of two matrices
    """
    r1=len(matrixA)
    if(type(matrixA[0]) == int or type(matrixA[0]) == float):
        r1=1
        c1=len(matrixA)
    else:
        c1=len(matrixA[0])
    r2=len(matrixB)
    if(type(matrixB[0]) == int or type(matrixB[0]) ==  float):
        r2=1
        c2=len(matrixB)
    else:
        c2=len(matrixB[0])
        
    matrixC=[]
    
    if(r1 == 1):
        lambdaA = []
        for i in matrixA:
            lambdaA.append([i])
        matrixA = lambdaA[:]
        
    if(r2 == 1):
        lambdaB = []
        for i in matrixB:
            lambdaB.append([i])
        matrixB = lambdaB[:]
    
    if(r1 == 1)|(r1 == c2 == 1):  # For (1x3)*(3x1)=(1x1), (1x2)*(2x2)=(1x2)
        for i in range(r1):
            matrixC = []
            for j in range(c2):
                matrixC.append(0)            
            
    else:        
        for i in range(r1):
            row_matrixC = []
            for j in range(c2):
                row_matrixC.append(0)            
            matrixC.append(row_matrixC)

    if (c1 != r2):
        print('Kindly change the order of matrices. For multiplication of matrices, the no. of columns of first matrix must be equal to no. of rows of second matrix.')
    
    elif(r1 == 1)&(c2 == 1):  #(1x3)*(3x1)=(1x1)
        for i in range(r1):
            for j in range(c2):
                for k in range(r2):
                    matrixC[i] += matrixA[k][0] * matrixB[k][j]
    
    elif(r1 == 1)&(r2 == 1):
        for i in range(r1):
            for j in range(c2):
                matrixC[j] += matrixA[i][0] * matrixB[j][0]
                
    elif(r1 == 1)&(c2 != 1): #(1x2)*(2x2)=(1x2)
        for i in range(r1):
            for j in range(c2):
                for k in range(r2):
                    matrixC[j] += matrixA[i][0] * matrixB[k][j]
                        
    
    else: #all other cases
        for i in range(r1):
            for j in range(c2):
                for k in range(r2):
                    matrixC[i][j] += matrixA[i][k] * matrixB[k][j]
                
    return matrixC
                    

def one_D_mat_transpose(m):
    matTransp = []
    for i in m:
        matTransp.append([i])
    return matTransp
  
  
# ============== Actual Code Starts here =============== #
initialPopSize      = 8
allPop_Weights      = []
allPopl_Chromosomes = []
allPop_ReceivedOut  = []
allPop_ErrorVal     = []

input_values        = [[10,20,30,40], [1,2,3,4], [-1,2,3,-4], [3, -21, -1, 1]]
yExp                = [[0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
X                   = input_values[:]
Y                   = yExp[:]


dimensions = (3, 6, 5)
dimension = [len(input_values[0])]
for i in dimensions:
    dimension.append(i)
dimension.append(len(Y[0]))


error_Val           = []


for popSize in range(initialPopSize):
    
# ======================================================== #
    #finding initial weights 
    allWeights = []

    for i in range(len(dimension)-1):
        k = dimension[i]*dimension[i+1]
        t=[]
        currentWeight = []
        count = 0 
        for z in range(k+1):
            if(count == dimension[i]):
                currentWeight.append(t)
                t=[]
                count=0
            t.append(round(random.random(),4))
            count+=1
        allWeights.append(currentWeight)                    # current chromosome weight matrices
        
    allPop_Weights.append(allWeights)                       # will store weight matrices for all chromosomes

# ======================================================== #

    #chromosome
    chromosome = []

    for i in allWeights:
        for j in i:
            for k in j:
                chromosome.append(k)                        # current chromosome 

    allPopl_Chromosomes.append(chromosome)                  # will store all chromosomes

    # ======================================================== #

    # expected output
    yExp = [[0,0,1], [0,1,0]]                               # 2 ouputs corresponding to 2 input samples

    # ======================================================== #

yo = []

for chromosome in allPopl_Chromosomes:
    y = []

    for inp in X:
        t = inp
        t = one_D_mat_transpose(t)
        for wi in allWeights:
            temp = matrix_mul(wi,t)
            t = temp

        for tempk in t:
            if tempk[0] >= 0.0:
                z = math.exp(-tempk[0])
                tempk[0] = 1 / (1 + z)

                # --- added to change values to 1 and 0 --- #
                if(tempk[0] + 0.002 >= 1.0):
                    tempk[0] = 1.0
                else:
                    tempk[0] = 0.0


            else:
                z = math.exp(tempk[0])
                tempk[0] = z / (1 + z)

                # --- added to change values to 1 and 0 --- #
                if(tempk[0] + 0.002 >= 1.0):
                    tempk[0] = 1.0
                else:
                    tempk[0] = 0.0

        y.append(t)                                     # y' = w.x'   is the list of ouputs for all input 
                                                        #             samples for current chromosome

    allPop_ReceivedOut.append(y)                        # yo or allPop_ReceivedOut is list of all received outputs
# ======================================================== #

#print(len(allPop_Weights))

#print(len(allPop_ReceivedOut))

"""
for i in allPop_ReceivedOut:
    for j in i:
        '''
        maxValPos = 0
        for k in range(len(j)):
            if(j[maxValPos][0] < j[k][0]):
                maxValPos = k
        '''
        
        for k in range(len(j)):
            '''
            if(k == maxValPos):
                j[k][0] = 1
            else:
                j[k][0] = 0
            '''
            j[k][0] = (1/(1+ 2.718281828459045 ** (-j[k][0])))
"""

# print(allPop_ReceivedOut)


# ======================================================== #

# error value
allErrorValue = []

for currPop_ReceivedOut_Count in range(len(allPop_ReceivedOut)):
    
    currPop_ReceivedOut = allPop_ReceivedOut[currPop_ReceivedOut_Count]
    curr_error = []
    for piCount in range(len(currPop_ReceivedOut)):
        pi = currPop_ReceivedOut[piCount]
        errorVal = 0
        for yj in range(len(pi)):
            errorVal += ((pi[yj][0] - Y[piCount][yj])**2)
        curr_error.append(errorVal)
    allErrorValue.append(curr_error)

    
# finding minimum error weight
minErrorVal = 10000000
for allErrorCount in range(len(allErrorValue)):
    minPos = len(Y)
    curr_error  = allErrorValue[allErrorCount]
    for currError_Count in range(len(curr_error)):
        if(curr_error[currError_Count] < minErrorVal):
            print(curr_error[currError_Count])
            minErrorVal = curr_error[currError_Count] 
            minPos = currError_Count
            minMainPos = allErrorCount

print(currError_Count)
print(allErrorCount)

print(allPopl_Chromosomes[allErrorCount])

bestChromosome = allPopl_Chromosomes[allErrorCount]
