import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from RACOG import RACOG
from imblearn.over_sampling import SMOTE

file = "UCI_Cross_Folder_npz/ecoli/ecoli_1_Cross_Folder.npz"
r = np.load(file)
Positive_Features_train = r["P_F_tr"]
Num_Positive_train = Positive_Features_train.shape[0]
Positive_Labels_train = np.linspace(1, 1, Num_Positive_train)
print(Num_Positive_train)

Negative_Features_train = r["N_F_tr"]
Num_Negative_train = Negative_Features_train.shape[0]
Negative_Labels_train = np.linspace(0, 0, Num_Negative_train)

num_features = Positive_Features_train.shape[1]

Features_train_o = np.concatenate((Positive_Features_train, Negative_Features_train))
Labels_train_o = np.concatenate((Positive_Labels_train, Negative_Labels_train))
#                print(Labels_train_o)

print(np.array_str(Features_train_o[0,:],100), Labels_train_o[0])
print(np.array_str(Features_train_o[1,:],100),Labels_train_o[1])
print(np.array_str(Features_train_o[Num_Positive_train+Num_Negative_train-2,:],100),Labels_train_o[Num_Positive_train+Num_Negative_train-2])
print(np.array_str(Features_train_o[Num_Positive_train+Num_Negative_train-1,:],100),Labels_train_o[Num_Positive_train+Num_Negative_train-1])

sm = SMOTE(ratio={0:300})
Feature_train, Label_train = sm.fit_sample(Features_train_o, Labels_train_o)

num = Feature_train.shape[0]
#print(np.array_str(Feature_train[0,:],100),Label_train[0])
#print(np.array_str(Feature_train[1,:],100),Label_train[1])
print(np.array_str(Feature_train[Num_Positive_train+Num_Negative_train-2,:],100),Label_train[Num_Positive_train+Num_Negative_train-2])
print(np.array_str(Feature_train[Num_Positive_train+Num_Negative_train-1,:],100),Label_train[Num_Positive_train+Num_Negative_train-1])
print(np.array_str(Feature_train[Num_Positive_train+Num_Negative_train,:],100),Label_train[Num_Positive_train+Num_Negative_train])
print(np.array_str(Feature_train[Num_Positive_train+Num_Negative_train+1,:],100),Label_train[Num_Positive_train+Num_Negative_train+1])
print(np.array_str(Feature_train[num-2,:],100),Label_train[num-2])
print(np.array_str(Feature_train[num-1,:],100),Label_train[num-1])
