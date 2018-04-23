import numpy as np
import scipy.stats as stats


file = "Result_b.txt"
#    file = "Result_a_" + str(f_i) + ".txt"
i = 0
with open(file, "r") as r:
    print(r.name)
    for line in r:
        column = line.split("\t")
        data_set = column[0]
        method = column[1]
        accuracy = float(column[2])
        precision = float(column[4])
        recall = float(column[6])
        specificity = float(column[8])
        g_mean = float(column[10])
        f_mean = float(column[12])
        auc = float(column[14])

        if i == 0:
            temp = data_set
            Accuracy = {data_set: {method: [accuracy]}}
            Precision = {data_set: {method: [precision]}}
            Recall = {data_set: {method: [recall]}}
            Specificity = {data_set: {method: [specificity]}}
            G_mean = {data_set: {method: [g_mean]}}
            F_mean = {data_set: {method: [f_mean]}}
            AUC = {data_set: {method: [auc]}}

        elif AUC.__contains__(data_set):
            Accuracy[data_set][method] = [accuracy]
            Precision[data_set][method] = [precision]
            Recall[data_set][method] = [recall]
            Specificity[data_set][method] = [specificity]
            G_mean[data_set][method] = [g_mean]
            F_mean[data_set][method] = [f_mean]
            AUC[data_set][method] = [auc]

        else:
            Accuracy[data_set] = {method: [accuracy]}
            Precision[data_set] = {method: [precision]}
            Recall[data_set] = {method: [recall]}
            Specificity[data_set] = {method: [specificity]}
            G_mean[data_set] = {method: [g_mean]}
            F_mean[data_set] = {method: [f_mean]}
            AUC[data_set] = {method: [auc]}
        i += 1

method_list = AUC[temp].keys()

list = [Accuracy, Precision, Recall, Specificity, G_mean, F_mean, AUC]

for ll_i in list:
    i = 0
    for key, values in ll_i.items():
        # print(values)
        l_m = []
        for k, v in values.items():
            l_m.append(np.mean(v))
        if i == 0:
            l_a = np.asarray(l_m)
            l_a = np.reshape(l_a, (1, -1))
        else:
            l_a = np.concatenate((l_a, np.reshape(np.asarray(l_m), (1, -1))))
        i += 1

#    l_a = l_a.transpose()
    print(l_a.shape)
    f_c_s, p = stats.friedmanchisquare(*[l_a[x, :] for x in np.arange(l_a.shape[0])])
    print("p value:", p)






