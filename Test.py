import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from RACOG import RACOG
from imblearn.over_sampling import SMOTE

a = [1,5,5,1,3,6,6,4,5,6,7,6]
print(a[0])
print(sorted(a))
print(a)
b = np.sort(np.asarray(a))
print(len(b))
print(b)
print(np.where(b == 5))
print(np.where(b == 5)[0])
print(np.mean(np.where(b == 5)[0])+1)