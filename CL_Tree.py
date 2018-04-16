import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

file = "UCI_Cross_Folder_npz/ecoli/ecoli_1_Cross_Folder.npz"
r = np.load(file)
Positive_Features_train = r["P_F_tr"]
Num_Positive_train = Positive_Features_train.shape[0]
Positive_Labels_train = np.linspace(1, 1, Num_Positive_train)
print(Num_Positive_train)

Positive_Features_test = r["P_F_te"]
Num_Positive_test = Positive_Features_test.shape[0]
Positive_Labels_test = np.linspace(1, 1, Num_Positive_test)

Negative_Features_train = r["N_F_tr"]
Num_Negative_train = Negative_Features_train.shape[0]
Negative_Labels_train = np.linspace(0, 0, Num_Negative_train)

Negative_Features_test = r["N_F_te"]
Num_Negative_test = Negative_Features_test.shape[0]
Negative_Labels_test = np.linspace(0, 0, Num_Negative_test)

#print(Positive_Features_train[0])

num_features = Positive_Features_train.shape[1]
print(num_features)

num_bins = 100
bins = np.linspace(0,1,num_bins+1)  #The input data must be norminalized into range [0,1]

marginal_distribution = np.zeros((num_bins, num_features))
for i in range(num_features):
    marginal_distribution[:,i] = np.histogram(Positive_Features_train[:,i], bins=bins)[0]/Num_Positive_train + 0.000000001
    marginal_distribution[:,i] = marginal_distribution[:,i]/sum(marginal_distribution[:,i])
print(marginal_distribution[0,2])

for i in range(num_features-1):
    for j in range(i+1,num_features):
        m_t = np.histogram2d(Positive_Features_train[:, i], Positive_Features_train[:, j], bins=[bins, bins])[0]
#        print(m_t)
        m_t = m_t / Num_Positive_train + 0.000000001
        m_t = m_t / sum(m_t)
#        print(m_t)
        if i == 0 and j == 1:
            marginal_pair_distribution = {(i,j):m_t}
        else:
            marginal_pair_distribution[(i,j)] = m_t

#print(marginal_pair_distribution[(i,j)][99,99])

def calculate_mutual_information(i,j):
    I = 0
    for m_i in range(marginal_pair_distribution[(i,j)].shape[0]):
        for m_j in range(marginal_pair_distribution[(i,j)].shape[1]):
            p_m = marginal_pair_distribution[i,j][m_i,m_j]
#            print(marginal_distribution[m_i,i])
#            print(marginal_distribution[m_j,j])
            I += p_m * (np.log(p_m) - np.log(marginal_distribution[m_i,i]) - np.log(marginal_distribution[m_j,j]))
#            print(I)
    return I


G = nx.Graph()
for i in range(num_features-1):
    G.add_node(i)
    for j in range(i+1,num_features):
        G.add_edge(j, i, weight=calculate_mutual_information(i, j))

T = nx.maximum_spanning_tree(G)

#nx.draw_networkx(T)
#plt.show()

print(sorted(T.edges(data=True)))




