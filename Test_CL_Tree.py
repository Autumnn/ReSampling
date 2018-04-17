import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

a = np.array([[0,0,0,0],[0,0,0,1],[0,1,1,0],[1,0,0,1]])
#print(a)
a = a.repeat(10, axis=0)
#print(a)
b = np.array([[0,0,1,0],[0,0,1,1],[0,1,1,1],[1,0,0,0],[1,1,0,0],[1,1,0,1]])
b = b.repeat(5,axis=0)
c = np.array([[1,1,1,0],[1,1,1,1]])
c = c.repeat(15, axis=0)

Positive_Features_train = np.concatenate((a, b, c))
num_features = 4
Num_Positive_train = 100

num_bins = 2
bins = np.linspace(0,1,num_bins+1)  #The input data must be norminalized into range [0,1]

marginal_distribution = np.zeros((num_bins, num_features))
for i in range(num_features):
    marginal_distribution[:,i] = np.histogram(Positive_Features_train[:,i], bins=bins)[0]/Num_Positive_train + 0.000000001
#    print(sum(marginal_distribution[:, i]))
    marginal_distribution[:,i] = marginal_distribution[:,i]/sum(marginal_distribution[:,i])
#    print(sum(marginal_distribution[:, i]))

#print(marginal_distribution)

for i in range(num_features-1):
    for j in range(i+1,num_features):
        m_t = np.histogram2d(Positive_Features_train[:, i], Positive_Features_train[:, j], bins=[bins, bins])[0]
#        print(sum(sum(m_t)))
        m_t = m_t / Num_Positive_train + 0.000000001
#        print(sum(sum(m_t)))
        m_t = m_t / sum(sum(m_t))
#        print(sum(sum(m_t)))
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
            p_m = marginal_pair_distribution[(i,j)][m_i,m_j]
#            print(marginal_distribution[m_i,i])
#            print(marginal_distribution[m_j,j])
#            print(p_m)
            I += p_m * (np.log(p_m) - np.log(marginal_distribution[m_i,i]) - np.log(marginal_distribution[m_j,j]))
    print(I)
    return I


G = nx.Graph()
for i in range(num_features-1):
    G.add_node(i)
    for j in range(i+1,num_features):
        G.add_edge(j, i, weight=-calculate_mutual_information(i, j))

T = nx.minimum_spanning_tree(G)

nx.draw_networkx(T)
plt.show()

print(sorted(T.edges(data=True)))
print(int(-1))

