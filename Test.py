import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from RACOG import RACOG

methods = ["SVM", "SMOTE", "cGAN", "cGAN-O", "cGAN-SMOTE"]

for m in methods:
    if m == "SVM":
        print(m)

