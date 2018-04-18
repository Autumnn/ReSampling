import numpy as np
import networkx as nx
import chow_liu_tree  as clt

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

    def fit(self, data, num_bins, num_create_samples):

        self.num_samples = data.shape[0]
        self.num_features = data.shape[1]
        print("Number of Features: ", self.num_features)
        self.num_bins = num_bins
        self.num_create_samples = num_create_samples

        #bins = np.linspace(0, 1, self.num_bins + 1)  # The input data must be norminalized into range [0,1]
        self.marginal_distribution = np.zeros((self.num_bins, self.num_features))
        self.marginal_bounds = np.zeros((self.num_bins+1, self.num_features))
        for i in range(self.num_features):
            self.marginal_distribution[:, i] = np.histogram(data[:, i], bins=self.num_bins)[
                                              0] / self.num_samples + 0.000000001
            self.marginal_bounds[:, i] = np.histogram(data[:, i], bins=self.num_bins)[1]
            #    print(sum(marginal_distribution[:, i]))
            self.marginal_distribution[:, i] = self.marginal_distribution[:, i] / \
                                               sum(self.marginal_distribution[:, i])
            #    print(sum(marginal_distribution[:, i]))

        for i in range(self.num_features - 1):
            for j in range(i + 1, self.num_features):
                m_t = np.histogram2d(data[:, i], data[:, j],
                                     bins=[self.marginal_bounds[:,i], self.marginal_bounds[:,j]])[0]
                #        print(sum(sum(m_t)))
                m_t = m_t / self.num_samples + 0.000000001
                #        print(sum(sum(m_t)))
                m_t = m_t / sum(sum(m_t))
                #        print(sum(sum(m_t)))
    #            if i == 0 and j == 1:
    #                marginal_pair_distribution = {(i, j): m_t}
    #            else:
                self.marginal_pair_distribution[(i, j)] = m_t



