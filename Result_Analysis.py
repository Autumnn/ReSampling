import numpy as np

file = "Result_1.txt"
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

        if i == 0 :
            temp = data_set
            Accuracy = {data_set:{method:accuracy}}
            Precision = {data_set:{method:precision}}
            Recall = {data_set:{method:recall}}
            Specificity = {data_set:{method:specificity}}
            G_mean = {data_set:{method:g_mean}}
            F_mean = {data_set:{method:f_mean}}
            AUC = {data_set:{method:auc}}

        elif AUC.__contains__(data_set):
            Accuracy[data_set][method] = accuracy
            Precision[data_set][method] = precision
            Recall[data_set][method] = recall
            Specificity[data_set][method] = specificity
            G_mean[data_set][method] = g_mean
            F_mean[data_set][method] = f_mean
            AUC[data_set][method] = auc

        else:
            Accuracy[data_set] = {method:accuracy}
            Precision[data_set] = {method:precision}
            Recall[data_set] = {method:recall}
            Specificity[data_set] = {method:specificity}
            G_mean[data_set] = {method:g_mean}
            F_mean[data_set] = {method:f_mean}
            AUC[data_set] = {method:auc}
        i += 1

method_list = AUC[temp].keys()

file_write = "Result_UCI_Analysis.txt"
with open(file_write, 'a') as w:
    head_line = "Accuracy" + '\t' + '\t'.join(str(x) + '\t' for x in method_list) + '\n'
    w.write(head_line)
    for key, values in Accuracy.items():
        #print(values)
        l_m = []
        for k, v in values.items():
            l_m.append(v)
        seq = sorted(l_m)
        w_line = key
        for i in range(len(l_m)):
            w_line += '\t' + str('%.4f' % l_m[i])
            w_line += '\t' + str(len(l_m)-seq.index(l_m[i]))
        w_line += '\n'
        w.write(w_line)
    w.write('\n')

    head_line = "Precision" + '\t' + '\t'.join(str(x) + '\t' for x in method_list) + '\n'
    w.write(head_line)
    for key, values in Precision.items():
        #print(values)
        l_m = []
        for k, v in values.items():
            l_m.append(v)
        seq = sorted(l_m)
        w_line = key
        for i in range(len(l_m)):
            w_line += '\t' + str('%.4f' % l_m[i])
            w_line += '\t' + str(len(l_m)-seq.index(l_m[i]))
        w_line += '\n'
        w.write(w_line)
    w.write('\n')

    head_line = "Recall" + '\t' + '\t'.join(str(x) + '\t' for x in method_list) + '\n'
    w.write(head_line)
    for key, values in Recall.items():
        #print(values)
        l_m = []
        for k, v in values.items():
            l_m.append(v)
        seq = sorted(l_m)
        w_line = key
        for i in range(len(l_m)):
            w_line += '\t' + str('%.4f' % l_m[i])
            w_line += '\t' + str(len(l_m)-seq.index(l_m[i]))
        w_line += '\n'
        w.write(w_line)
    w.write('\n')

    head_line = "Specificity" + '\t' + '\t'.join(str(x) + '\t' for x in method_list) + '\n'
    w.write(head_line)
    for key, values in Specificity.items():
        #print(values)
        l_m = []
        for k, v in values.items():
            l_m.append(v)
        seq = sorted(l_m)
        w_line = key
        for i in range(len(l_m)):
            w_line += '\t' + str('%.4f' % l_m[i])
            w_line += '\t' + str(len(l_m)-seq.index(l_m[i]))
        w_line += '\n'
        w.write(w_line)
    w.write('\n')

    head_line = "G-mean" + '\t' + '\t'.join(str(x) + '\t' for x in method_list) + '\n'
    w.write(head_line)
    for key, values in G_mean.items():
        #print(values)
        l_m = []
        for k, v in values.items():
            l_m.append(v)
        seq = sorted(l_m)
        w_line = key
        for i in range(len(l_m)):
            w_line += '\t' + str('%.4f' % l_m[i])
            w_line += '\t' + str(len(l_m)-seq.index(l_m[i]))
        w_line += '\n'
        w.write(w_line)
    w.write('\n')

    head_line = "F-mean" + '\t' + '\t'.join(str(x) + '\t' for x in method_list) + '\n'
    w.write(head_line)
    for key, values in F_mean.items():
        #print(values)
        l_m = []
        for k, v in values.items():
            l_m.append(v)
        seq = sorted(l_m)
        w_line = key
        for i in range(len(l_m)):
            w_line += '\t' + str('%.4f' % l_m[i])
            w_line += '\t' + str(len(l_m)-seq.index(l_m[i]))
        w_line += '\n'
        w.write(w_line)
    w.write('\n')

    head_line = "AUC" + '\t' + '\t'.join(str(x) + '\t' for x in method_list) + '\n'
    w.write(head_line)
    for key, values in AUC.items():
        #print(values)
        l_m = []
        for k, v in values.items():
            l_m.append(v)
        seq = sorted(l_m)
        w_line = key
        for i in range(len(l_m)):
            w_line += '\t' + str('%.4f' % l_m[i])
            w_line += '\t' + str(len(l_m)-seq.index(l_m[i]))
        w_line += '\n'
        w.write(w_line)
