import numpy as np
import networkx as nx
from Chow_Liu_Tree import Chow_Liu_Tree

class RACOG():

    def __init__(self):
        self.num_bins = 100
        self.num_features = 0
        self.dependency = []
        self.num_create_samples = 0
        self.num_samples = 0
        self.marginal_distribution = []
        self.marginal_bounds = []
        self.marginal_pair_distribution = {}
        self.Clt = Chow_Liu_Tree()
        self.conditional_probability = []
        self.Prob = {}

    def fit(self, data, num_bins):

        self.num_samples = data.shape[0]
        self.num_features = data.shape[1]
        print("Number of Features: ", self.num_features)
        self.num_bins = num_bins

        self.dependency = self.Clt.fit(data, self.num_bins)
        self.marginal_distribution = self.Clt.get_marginal_distribution()
        self.marginal_bounds = self.Clt.get_marginal_bound()
        self.marginal_pair_distribution = self.Clt.get_marginal_pair_distribution()

        #initial = True
        for i in range(len(self.dependency)):
            if self.dependency[i] != -1:
                d = int(self.dependency[i])
                if i < d:
                    ival_num = self.marginal_pair_distribution[(i, d)].shape[0]
                    dval_num = self.marginal_pair_distribution[(i, d)].shape[1]
                else:
                    ival_num = self.marginal_pair_distribution[(d, i)].shape[1]
                    dval_num = self.marginal_pair_distribution[(d, i)].shape[0]

                self.conditional_probability = np.zeros((dval_num, ival_num))

                for i_dval in range(dval_num):
                    for i_ival in range(ival_num):
                        if i < d:
                            self.conditional_probability[i_dval, i_ival] = self.marginal_pair_distribution[(i, d)][i_ival, i_dval] / sum(self.marginal_pair_distribution[(i, d)][:, i_dval])
                        else:
                            self.conditional_probability[i_dval, i_ival] = self.marginal_pair_distribution[(d, i)][i_dval, i_ival] / sum(self.marginal_pair_distribution[(d, i)][i_dval, :])
            #            print(sum(conditional_probability[i_dval,:]), i)
            else:
                self.conditional_probability = self.marginal_distribution[:, i]

            #if initial:
            #    Prob = {i: conditional_probability}
            #    initial = False
            #else:
            self.Prob[i] = self.conditional_probability

    def num_factor(self, z):
        prob = 1
        for i in range(self.num_features):
            cur_val = z[i]
            d = int(self.dependency[i])
            if d == -1:
                pr = self.Prob[i][cur_val]
            else:
                d_index = int(z[d])
                pr = self.Prob[i][d_index, cur_val]

            prob = prob * pr

        return prob

    def samples(self, minority_data, num_create_samples, burn_in=100, lag=20):
        self.num_create_samples = num_create_samples
        minority_size = minority_data.shape[0]
        epoch = np.ceil(num_create_samples/minority_size)*20 + 100 +1

        z = np.linspace(0,0,self.num_features)
        for k in range(minority_size):
            initial_sample = minority_data[k]
            for i in range(self.num_features):
                z[i] = np.max(np.where(self.marginal_bounds[:,i] <= initial_sample[i])[0])

            for t in range(epoch):
                for i in range(self.num_features):
                    probability = np.linspace(0, 0, self.num_bins)
                    for j in range(self.num_bins):
                        z[i] = j
                        probability[j] = self.num_factor(z)

                    p_sum = sum(probability)
                    #        print(p_sum)
                    probability = probability / p_sum
                    #        print(probability)

                    u = np.random.random_sample()
                    #        print("u", u)
                    add = 0
                    for j in range(self.num_bins):
                        add += probability[j]
                        #            print("add", add)
                        if u <= add:
                            new_z_i = j
                            #                print("add", add)
                            break

                    z[i] = new_z_i

                if t > burn_in & t % lag == 0:
                    sample = np.linspace(0, 0, self.num_features)
                    for j in range(self.num_features):
                        u_r = np.random.random_sample()
           ##             sample[j] = Z_temp[j] * 0.01 + u_r * 0.01
                    print(sample)




