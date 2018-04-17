import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

a = np.array([1.3333,2.52322,3.64324,4.823423])
print(a)
b = np.floor(a*100)
b = b.astype(int)
print(b)
a = a/sum(a)
print(a)