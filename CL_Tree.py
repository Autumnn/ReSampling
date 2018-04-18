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
#    print(sum(marginal_distribution[:, i]))
    marginal_distribution[:,i] = marginal_distribution[:,i]/sum(marginal_distribution[:,i])
#    print(sum(marginal_distribution[:, i]))

for i in range(num_features-1):
    for j in range(i+1,num_features):
        m_t = np.histogram2d(Positive_Features_train[:, i], Positive_Features_train[:, j], bins=[bins, bins])[0]
#        print(sum(sum(m_t)))
        m_t = m_t / Num_Positive_train + 0.000000001
#        print(sum(sum(m_t)))
        m_t = m_t / sum(sum(m_t))
#        print(sum(sum(m_t)))
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
#    print(I)
    return I


G = nx.Graph()
for i in range(num_features-1):
    G.add_node(i)
    for j in range(i+1,num_features):
        G.add_edge(j, i, weight=-calculate_mutual_information(i, j))

T = nx.minimum_spanning_tree(G)     #minimum_spanning_tree use Kruskal algorithm, but maximum_spanning_tree use Boruvka's algorithm

#nx.draw_networkx(T)
#plt.show()
print(T.edges())

dependency = np.linspace(0,0,num_features)
dependency[num_features-1] = -1 #Root node
leaf = [num_features-1]
while len(T.nodes()) > 0:
    for node in leaf:
        if node in T.nodes():
            for nb in nx.all_neighbors(T,node):
                dependency[int(nb)] = node
                leaf.append(nb)
            T.remove_node(node)
        leaf.remove(node)

print(dependency)


#bayes = BayesianNetwork.from_samples(Positive_Features_train, algorithm='chow-liu', root=num_features-1)
#depend = []
#for i in bayes.structure:
#    if i:
#        depend.append(i[0])
#    else:
#        depend.append(-1)

#print(depend)



initial = True
for i in range(len(dependency)):
    if dependency[i] != -1:
        d = int(dependency[i])
        if i < d:
            ival_num = marginal_pair_distribution[(i,d)].shape[0]
            dval_num = marginal_pair_distribution[(i,d)].shape[1]
        else:
            ival_num = marginal_pair_distribution[(d,i)].shape[1]
            dval_num = marginal_pair_distribution[(d,i)].shape[0]

        conditional_probability  = np.zeros((dval_num, ival_num))

        for i_dval in range(dval_num):
            for i_ival in range(ival_num):
                if i < d:
                    conditional_probability[i_dval, i_ival] = marginal_pair_distribution[(i, d)][i_ival, i_dval]\
                                                              /sum(marginal_pair_distribution[(i, d)][:,i_dval])
                else:
                    conditional_probability[i_dval, i_ival] = marginal_pair_distribution[(d, i)][i_dval, i_ival] \
                                                              / sum(marginal_pair_distribution[(d, i)][i_dval, :])
#            print(sum(conditional_probability[i_dval,:]), i)
    else:
        conditional_probability = marginal_distribution[:,i]

    if initial:
        Prob = {i:conditional_probability}
        initial = False
    else:
        Prob[i] = conditional_probability

#    print(conditional_probability)


#num_create_sample = 10
#for i in range(num_create_sample):

def NumFactor(z):
    prob = 1
    for i in range(num_features):
        cur_val = z[i]
        d = int(dependency[i])
        if d == -1:
            pr = Prob[i][cur_val]
        else:
            d_index = int(z[d])
            pr = Prob[i][d_index, cur_val]

        prob = prob * pr

    return prob


Z = np.floor(Positive_Features_train[0,:]*100)
Z = Z.astype(int)
print(Z)
T = 300

Z_temp = Z
for t in range(T):
    for i in range(num_features):
        probability = np.linspace(0,0,num_bins)
        for j in range(num_bins):
            Z_temp[i] = j
            probability[j] = NumFactor(Z_temp)

        p_sum = sum(probability)
#        print(p_sum)
        probability = probability/p_sum
#        print(probability)

        u = np.random.random_sample()
#        print("u", u)
        add = 0
        for j in range(num_bins):
            add += probability[j]
#            print("add", add)
            if u <= add:
                new_z_i = j
#                print("add", add)
                break

        Z_temp[i] = new_z_i

    if t>100 & t%20==0:
        sample = np.linspace(0,0,num_features)
        for j in range(num_features):
            u_r = np.random.random_sample()
            sample[j] = Z_temp[j]*0.01 + u_r*0.01
        print(sample)





