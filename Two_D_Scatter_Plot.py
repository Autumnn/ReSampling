from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np


def plot(Feature, Label, F_Feature, Attribute, name):
    num_positive = len(Label.nonzero()[0])
    num_negative = len(Label) - num_positive
    size = Feature.shape
    num_feature = size[1]
    Positive_Feature = np.ones((num_positive, num_feature))
    Negative_Feature = np.ones((num_negative, num_feature))
    i_p = 0
    i_n = 0
    for i_l in range(size[0]):
        if Label[i_l] == 1:
            Positive_Feature[i_p,:] = Feature[i_l,:]
            i_p += 1
        else:
            Negative_Feature[i_n,:] = Feature[i_l,:]
            i_n += 1

    start_index = size[0]-1
    S_Feature = F_Feature[start_index:,:]

    l = len(Attribute)
    for i in range(0, l):
        for j in range(i+1, l):
            x_index = Attribute[i]
            y_index = Attribute[j]
            fig = plt.figure()
            plt.scatter(Negative_Feature[:,x_index], Negative_Feature[:,y_index], marker = 'o', color = '#539caf', label='1', s = 6, alpha=0.6)
            plt.scatter(Positive_Feature[:,x_index], Positive_Feature[:,y_index], marker = '+', color = 'r', label='2', s = 50)
            plt.scatter(S_Feature[:, x_index], S_Feature[:, y_index], marker='^', color='rebeccapurple', label='1', s=3, alpha=0.3)
            File_name = "Scatter_Plot_of_" + name + "_the_" + str(i) + "_and_" + str(j) + "_Feature.png"
            fig.savefig(File_name)


