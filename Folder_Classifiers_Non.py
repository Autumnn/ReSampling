import os
import numpy as np
from sklearn import svm, preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
import Read_Data_UCI as RD
from imblearn.metrics import geometric_mean_score, specificity_score

#  first "min_max_scalar" ant then "StratifiedKFold".

path = "UCI_Cross_Folder_npz"
dirs = os.listdir(path) #Get files in the folder
First_line = True

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files= os.listdir(dir_path) #Get files in the folder

    Num_Gamma = 12
    gamma = np.logspace(-2, 1, Num_Gamma)
    Num_C = 6
    C = np.logspace(-1, 4, Num_C)
    Accuracy = np.zeros((Num_Gamma, Num_C))
    Precision = np.zeros((Num_Gamma, Num_C))
    Recall = np.zeros((Num_Gamma, Num_C))
    Specificity = np.zeros((Num_Gamma, Num_C))
    G_Mean = np.zeros((Num_Gamma, Num_C))
    F_Mean = np.zeros((Num_Gamma, Num_C))
    AUC = np.zeros((Num_Gamma, Num_C))

    for j in range(Num_Gamma):
        for k in range(Num_C):
            # print("gamma = ", str(gamma[j]), " C = ", str(C[k]))

            Num_Cross_Folders = 5
            #skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=True)
            skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=False)
            Accuracy_temp = np.linspace(0,0,Num_Cross_Folders)
            Precision_temp = np.linspace(0,0,Num_Cross_Folders)
            Recall_temp = np.linspace(0,0,Num_Cross_Folders)
            Specificity_temp = np.linspace(0,0,Num_Cross_Folders)
            G_Mean_temp = np.linspace(0,0,Num_Cross_Folders)
            F_Mean_temp = np.linspace(0,0,Num_Cross_Folders)
            AUC_temp = np.linspace(0,0,Num_Cross_Folders)

            i = 0
            for file in files:
                name = dir_path + '/' + file
                r = np.load(name)

                Positive_Features_train = r["P_F_tr"]
                Num_Positive_train = Positive_Features_train.shape[0]
                Positive_Labels_train = np.linspace(1, 1, Num_Positive_train)

                Positive_Features_test = r["P_F_te"]
                Num_Positive_test = Positive_Features_test.shape[0]
                Positive_Labels_test = np.linspace(1, 1, Num_Positive_test)

                Negative_Features_train = r["N_F_tr"]
                Num_Negative_train = Negative_Features_train.shape[0]
                Negative_Labels_train = np.linspace(0, 0, Num_Negative_train)

                Negative_Features_test = r["N_F_te"]
                Num_Negative_test = Negative_Features_test.shape[0]
                Negative_Labels_test = np.linspace(0, 0, Num_Negative_test)

#                print(i, " folder; " , "Po_tr: ", Num_Positive_train, "Ne_tr: ", Num_Negative_train,
#                       "Po_te: ", Num_Positive_test, "Ne_te: ", Num_Negative_test)

                Features_train_o = np.concatenate((Positive_Features_train, Negative_Features_train))
                Labels_train_o = np.concatenate((Positive_Labels_train, Negative_Labels_train))
#                print(Labels_train_o)
                Feature_test = np.concatenate((Positive_Features_test, Negative_Features_test))
                Label_test = np.concatenate((Positive_Labels_test, Negative_Labels_test))
#                print(Label_test)

                clf = svm.SVC(C=C[k], kernel='rbf', gamma=gamma[j])
                clf.fit(Features_train_o, Labels_train_o)
                Label_predict = clf.predict(Feature_test)
#                print(Label_predict)

                Accuracy_temp[i] = metrics.accuracy_score(Label_test, Label_predict)
#                print(Accuracy_temp[i])
                Precision_temp[i] = metrics.precision_score(Label_test, Label_predict)
#                print(Precision_temp[i])
                Recall_temp[i] = metrics.recall_score(Label_test, Label_predict)
#                print(Recall_temp[i])
                Specificity_temp[i] = specificity_score(Label_test, Label_predict)
#                print(Specificity_temp[i])
                G_Mean_temp[i] = geometric_mean_score(Label_test, Label_predict)
#                print(G_Mean_temp[i])
                F_Mean_temp[i] = metrics.f1_score(Label_test, Label_predict)
#                print(F_Mean_temp[i])
                Label_score = clf.decision_function(Feature_test)
#                print(Label_score[i])
                AUC_temp[i] = metrics.roc_auc_score(Label_test, Label_score)
 #               print(AUC_temp[i])
                i += 1

            Accuracy[j, k] = np.mean(Accuracy_temp)
            Precision[j, k] = np.mean(Precision_temp)
            Recall[j, k] = np.mean(Recall_temp)
            Specificity[j, k] = np.mean(Specificity_temp)
            G_Mean[j, k] = np.mean(G_Mean_temp)
            F_Mean[j, k] = np.mean(F_Mean_temp)
            AUC[j, k] = np.mean(AUC_temp)

    file_wirte = "Result.txt"
    with open(file_wirte,'a') as w:
       # if First_line:
       #     metrics_list = ["Accuracy", "Precision", "Recall", "Specificity", "G-mean", "F-mean", "AUC"]
       #     first_line = "dataset" + '\t' + "method" + '\t' + '\t'.join(str(x) + '\t' + 'parameters' for x in metrics_list) + '\n'
       #     w.write(first_line)
       #     First_line = False

        line = Dir + '\t' + "SVM" + '\t'
        accuracy = np.max(Accuracy)
        accuracy_parameters = np.argwhere(Accuracy == accuracy)
        line += str(accuracy) + '\t' + str('%.3f'% gamma[accuracy_parameters[0][0]]) + \
                ',' + str('%.3f'% C[accuracy_parameters[0][1]]) + '\t'

        precision = np.max(Precision)
        precision_parameters = np.argwhere(Precision == precision)
        line += str(precision) + '\t' + str('%.3f'% gamma[precision_parameters[0][0]]) + \
                ',' + str('%.3f'% C[precision_parameters[0][1]]) + '\t'

        recall = np.max(Recall)
        recall_parameters = np.argwhere(Recall == recall)
        line += str(recall) + '\t' + str('%.3f'% gamma[recall_parameters[0][0]]) + \
                ',' + str('%.3f'% C[recall_parameters[0][1]]) + '\t'

        specificity = np.max(Specificity)
        specificity_parameters = np.argwhere(Specificity == specificity)
        line += str(specificity) + '\t' + str('%.3f'% gamma[specificity_parameters[0][0]]) + \
                ',' + str('%.3f'% C[specificity_parameters[0][1]]) + '\t'

        g_mean = np.max(G_Mean)
        g_parameters = np.argwhere(G_Mean == g_mean)
        line += str(g_mean) + '\t' + str('%.3f'% gamma[g_parameters[0][0]]) + \
                ',' + str('%.3f'% C[g_parameters[0][1]]) + '\t'

        f_mean = np.max(F_Mean)
        f_parameters = np.argwhere(F_Mean == f_mean)
        line += str(f_mean) + '\t' + str('%.3f'% gamma[f_parameters[0][0]]) + \
                ',' + str('%.3f'% C[f_parameters[0][1]]) + '\t'

        auc = np.max(AUC)
        auc_parameters = np.argwhere(AUC == auc)
        line += str(auc) + '\t' + str('%.3f'% gamma[auc_parameters[0][0]]) + \
                ',' + str('%.3f'% C[auc_parameters[0][1]]) + '\n'

        w.write(line)

