# Learning simple threading

import numpy as np
from threading import Thread

a = [(int(10000 * np.random.random())) for i in range(10000)]
b = a[:]
b.sort()
maximum = b[-1]
m = 10
chunk_size = len(a) / m
result = [0] * m


def maxvalue(arr, result_in, i):
    max_value = -1
    for k in arr:
        if (k > max_value):
            max_value = k
    result_in[i] = max_value


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
print(a)
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
print(result)

if(maximum == result[0]):
    print("Successful")
else:
    print("This will not be printed :)")
