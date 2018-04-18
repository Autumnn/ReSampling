import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from RACOG import RACOG

file = "UCI_Cross_Folder_npz/ecoli/ecoli_1_Cross_Folder.npz"
r = np.load(file)

Positive_Features_train = r["P_F_tr"]
Num_Positive_train = Positive_Features_train.shape[0]
Positive_Labels_train = np.linspace(1, 1, Num_Positive_train)
print(Num_Positive_train)

Negative_Features_train = r["N_F_tr"]
Num_Negative_train = Negative_Features_train.shape[0]
Negative_Labels_train = np.linspace(0, 0, Num_Negative_train)

racog = RACOG()
racog.fit(Positive_Features_train)
num = Num_Negative_train - Num_Positive_train
created_samples = racog.samples(Positive_Features_train, num)

num_Feature = Positive_Features_train.shape[1]

for i in range(0, num_Feature):
    for j in range(i+1, num_Feature):
        if i != j:
            fig = plt.figure()
            p1 = plt.scatter(Positive_Features_train[:,i], Positive_Features_train[:,j], marker = 'o', color = '#539caf', label='1', s = 10, alpha=0.9)
            p2 = plt.scatter(created_samples[:,i], created_samples[:,j], marker = '+', color = 'r', label='2', s = 50)
            File_name = "Scatter_Plot_of_" + str(i) + "_and_" + str(j) + "_Feature.png"
            fig.savefig(File_name)


