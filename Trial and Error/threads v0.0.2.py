# Learning simple threading - testing out

import numpy as np
from threading import Thread
import time

print("\n\n=========Finding maximum element in an array using threads=========\n\n")

a = [(int(10000 * np.random.random())) for i in range(10000000)]
# print("Array : ", a)
start_time = time.time()
b = a[:]
b.sort()
maximum = b[-1]
end_time = time.time()
print("Time taken without threading : ",end_time - start_time)
m = 1000
chunk_size = len(a) / m
result = [0] * m


def maxvalue(arr, result_in, i):
    max_value = -1
    for k in arr:
        if (k > max_value):
            max_value = k
    result_in[i] = max_value


start_time = time.time()
threads = []
start = 0
end = start + int(chunk_size)
for i in range(m):
    b = a[start:end]
    process = Thread(target=maxvalue, args=[b, result, i])
    process.start()
    threads.append(process)
    start = end
    end = start + int(chunk_size)
for process in threads:
    process.join()
a = result[:]
m = 1
chunk_size = len(a) / m
result = [0] * m
threads = []
start = 0
end = start + int(chunk_size)
for i in range(m):
    b = a[start:end]
    process = Thread(target=maxvalue, args=[b, result, i])
    process.start()
    threads.append(process)
    start = end
    end = start + int(chunk_size)
for process in threads:
    process.join()
print("Maximum Value : ", result)
end_time = time.time()
print("Time taken using threads : ", end_time - start_time)

if (maximum == result[0]):
    print("Successful")
else:
    print("This will not be printed :)")
