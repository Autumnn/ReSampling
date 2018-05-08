import numpy as np
import scipy.stats as stats


file = "Result_xGBoost.txt"
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

method_list = list(AUC[temp].keys())
print(method_list)

#list = [Accuracy, Precision, Recall, Specificity, G_mean, F_mean, AUC]

#for ll_i in list:
i = 0
for key, values in F_mean.items():
    # print(values)   key: dataset name
    l_m = []
    for k, v in values.items():         # k: method name
        l_m.append(np.mean(v))
    seq = np.sort(np.asarray(l_m))
    l_am = []
    for j in range(len(l_m)):
        l_am.append(len(l_m) - np.mean(np.where(seq == l_m[j])[0]))
    if i == 0:
        l_a = np.asarray(l_am)
        l_a = np.reshape(l_am, (1, -1))
    else:
        l_a = np.concatenate((l_a, np.reshape(np.asarray(l_am), (1, -1))))
    i += 1

#    l_a = l_a.transpose()
print(l_a)
Control_method = 0
for i in np.arange(l_a.shape[1]):
    if i != Control_method:
        t_s, p = stats.ttest_ind(l_a[:,i],l_a[:,Control_method],equal_var=False)
        #print(t_s, p)
        if i == 1:
            p_value = {method_list[i]:p}
        else:
            p_value[method_list[i]] = p

holm_p = sorted(p_value.items(), key=lambda item:item[1])

for ii in holm_p:
    print(ii)






