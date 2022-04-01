import pandas as pd
import matplotlib.pyplot as plt
fileName = input("Enter Dataset Name : ")+".csv"
data = pd.read_csv(fileName)

classes = []
outputValues = []
inputValues = []

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
            outputValues.append(classes[j][1])
            break

inputValues = []
for j in range(150):
    b = []
    for i in range(1, len(data.columns)-1):
        b.append(data[data.columns[i]][j])
    inputValues.append(b)
