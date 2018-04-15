import numpy as np


a = np.arange(12).reshape(4,3)
print(a)
b = np.max(a)
print(b)
i = np.argwhere(a == b)

print(i[0][0])
print(i[0][1])
