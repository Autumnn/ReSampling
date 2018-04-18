import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

a = np.array([[1.3333,2.52322,3.64324,4.823423],[1.3333,2.52322,3.64324,4.823423]])
print(a)
b = np.where(a <= [3.422, 5, 2, 1])
print(b)
print(np.max(b[0]))
