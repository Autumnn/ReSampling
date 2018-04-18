import numpy as np
import networkx as nx


class Chow_Liu_Tree():

    def __init__(self):
        self.dependency = []
        self.num_bins = 100
        self.num_features = 0
        self.num_samples = 0
        self.marginal_distribution = np.array([], dtype=float)
        self.marginal_bounds = np.array([], dtype=float)
        self.marginal_pair_distribution = {}

    def calculate_mutual_information(self, i, j):
        info = 0
        for m_i in range(self.marginal_pair_distribution[(i, j)].shape[0]):
            for m_j in range(self.marginal_pair_distribution[(i, j)].shape[1]):
                p_m = self.marginal_pair_distribution[(i, j)][m_i, m_j]
                #            print(marginal_distribution[m_i,i])
                #            print(marginal_distribution[m_j,j])
                #            print(p_m)
                info += p_m * (np.log(p_m) - np.log(self.marginal_distribution[m_i, i]) -
                               np.log(self.marginal_distribution[m_j, j]))
        #    print(I)
        return info

    def fit(self, data, num_bins):
        self.num_samples = data.shape[0]
        self.num_features = data.shape[1]
        print("Number of Features: ", self.num_features)
        self.num_bins = num_bins
        self.marginal_distribution = np.zeros((self.num_bins, self.num_features))

        for i in range(self.num_features):
            self.marginal_distribution[:, i] = np.histogram(data[:, i], bins=self.num_bins)[
                                                   0] / self.num_samples + 0.000000001
            self.marginal_bounds[:, i] = np.histogram(data[:, i], bins=self.num_bins)[1]
            #    print(sum(marginal_distribution[:, i]))
            self.marginal_distribution[:, i] = self.marginal_distribution[:, i] / sum(self.marginal_distribution[:, i])
            #    print(sum(marginal_distribution[:, i]))

        for i in range(self.num_features - 1):
            for j in range(i + 1, self.num_features):
                m_t = np.histogram2d(data[:, i], data[:, j],
                                     bins=[self.marginal_bounds[:, i], self.marginal_bounds[:, j]])[0]
                #        print(sum(sum(m_t)))
                m_t = m_t / self.num_samples + 0.000000001
                #        print(sum(sum(m_t)))
                m_t = m_t / sum(sum(m_t))
                #        print(sum(sum(m_t)))
                #            if i == 0 and j == 1:
                #                marginal_pair_distribution = {(i, j): m_t}
                #            else:
                self.marginal_pair_distribution[(i, j)] = m_t

        g = nx.Graph()
        for i in range(self.num_features - 1):
            g.add_node(i)
            for j in range(i + 1, self.num_features):
                g.add_edge(j, i, weight=-self.calculate_mutual_information(i, j))

        tree = nx.minimum_spanning_tree(g)
        # minimum_spanning_tree use Kruskal algorithm, but maximum_spanning_tree use Boruvka's algorithm
        print("Dependency Tree has been built, tree edges are: ")
        print(tree.edges())

        self.dependency = np.linspace(0, 0, self.num_features)
        self.dependency[self.num_features - 1] = -1  # Root node
        leaf = [self.num_features - 1]
        while len(tree.nodes()) > 0:
            for node in leaf:
                if node in tree.nodes():
                    for nb in nx.all_neighbors(tree, node):
                        self.dependency[int(nb)] = node
                        leaf.append(nb)
                    tree.remove_node(node)
                leaf.remove(node)

        print("Dependency: ", self.dependency)
        return self.dependency

    def get_marginal_distribution(self):
        return self.marginal_distribution

    def get_marginal_bound(self):
        return self.marginal_bounds

    def get_marginal_pair_distribution(self):
        return self.marginal_pair_distribution
