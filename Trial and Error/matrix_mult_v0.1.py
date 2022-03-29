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
    print(r1,c1,r2,c2)
    
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
                    print(matrixB[k][j])
                    matrixC[j] += matrixA[i][0] * matrixB[k][j]
                        
    
    else: #all other cases
        for i in range(r1):
            for j in range(c2):
                for k in range(r2):
                    matrixC[i][j] += matrixA[i][k] * matrixB[k][j]
                
    return matrixC
                    
matrix_mul([1,2],[[1,2,3],[1,2,3]])
