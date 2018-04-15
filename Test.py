import numpy as np


a = np.arange(36).reshape(4,3,3)
print(a)
b = np.zeros((4,3))
for i in range(4):
    for j in range(3):
        b[i,j] = np.mean(a[i,j,:])

print(b)

