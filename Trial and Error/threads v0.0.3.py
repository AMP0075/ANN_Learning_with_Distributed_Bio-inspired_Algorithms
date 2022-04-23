import numpy as np
from threading import Thread
import time


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def maxvalue(arr, result_in, i):
    res = result_in[:]
    max_value = -1
    for k in arr:
        if (k > max_value):
            max_value = k
    res[i] = max_value
    return res[i]


def threading(a, m, result):
    threads = []
    start = 0
    end = start + int(chunk_size)
    for i in range(m):
        b = a[start:end]
        process = ThreadWithReturnValue(target=maxvalue, args=[b, result, i])
        process.start()
        threads.append(process)
        start = end
        end = start + int(chunk_size)
    i = 0
    for process in threads:
        result[i] = process.join()
        i += 1
    a = result[:]
    return a


a = [int(np.random.random() * i) for i in range(10000000)]  # 10^7
b = a[:]
start_time = time.time()
b.sort()
maximum = b[-1]
end_time = time.time()
print("Time taken with sort function : ", end_time - start_time)  # nearly 2.5 seconds on my laptop
m = 10
chunk_size = len(a) / m
result = [0] * m

start_time = time.time()
a = threading(a, m, result)
a.sort()
if(a[-1] == maximum):
    k = "Successful"
else:
    k = "Failed"
print(k)
end_time = time.time()
print("Time taken with threads : ", end_time - start_time)  # nearly 0.3 seconds on my laptop
