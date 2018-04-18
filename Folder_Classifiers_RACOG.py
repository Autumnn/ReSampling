import os
import numpy as np
from sklearn import svm, preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.metrics import geometric_mean_score, specificity_score
from RACOG import RACOG

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
    Num_Cross_Folders = 5
    Accuracy = np.zeros((Num_Gamma, Num_C, Num_Cross_Folders))
    Precision = np.zeros((Num_Gamma, Num_C, Num_Cross_Folders))
    Recall = np.zeros((Num_Gamma, Num_C,Num_Cross_Folders))
    Specificity = np.zeros((Num_Gamma, Num_C, Num_Cross_Folders))
    G_Mean = np.zeros((Num_Gamma, Num_C, Num_Cross_Folders))
    F_Mean = np.zeros((Num_Gamma, Num_C, Num_Cross_Folders))
    AUC = np.zeros((Num_Gamma, Num_C, Num_Cross_Folders))

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

        print(i, " folder; " , "Po_tr: ", Num_Positive_train, "Ne_tr: ", Num_Negative_train,
            "Po_te: ", Num_Positive_test, "Ne_te: ", Num_Negative_test)

        Features_train_o = np.concatenate((Positive_Features_train, Negative_Features_train))
        Labels_train_o = np.concatenate((Positive_Labels_train, Negative_Labels_train))
        #                print(Labels_train_o)
        Feature_test = np.concatenate((Positive_Features_test, Negative_Features_test))
        Label_test = np.concatenate((Positive_Labels_test, Negative_Labels_test))
        #                print(Label_test)

        #expand_rate_for_Majority = 0.5
        #num_create_samples = int(np.ceil(Num_Negative_train * expand_rate_for_Majority))
        num_create_samples = Num_Negative_train - Num_Positive_train
        condition_samples = np.linspace(0, 0, num_create_samples)

        racog = RACOG()
        racog.fit(Positive_Features_train)
        sudo_Samples = racog.samples(Positive_Features_train, num_create_samples)
        Feature_train = np.concatenate((Features_train_o, sudo_Samples))
        Label_train = np.concatenate((Labels_train_o, condition_samples))

        for j in range(Num_Gamma):
            for k in range(Num_C):
                # print("gamma = ", str(gamma[j]), " C = ", str(C[k]))

                clf = svm.SVC(C=C[k], kernel='rbf', gamma=gamma[j])
                clf.fit(Feature_train, Label_train)
                Label_predict = clf.predict(Feature_test)

                Accuracy[j, k, i] = metrics.accuracy_score(Label_test, Label_predict)
                Precision[j, k, i] = metrics.precision_score(Label_test, Label_predict)
                Recall[j, k, i] = metrics.recall_score(Label_test, Label_predict)
                Specificity[j, k, i] = specificity_score(Label_test, Label_predict)
                G_Mean[j, k, i] = geometric_mean_score(Label_test, Label_predict)
                F_Mean[j, k, i] = metrics.f1_score(Label_test, Label_predict)
                Label_score = clf.decision_function(Feature_test)
                AUC[j, k, i] = metrics.roc_auc_score(Label_test, Label_score)

        i += 1

    Accuracy_t = np.zeros((Num_Gamma, Num_C))
    Precision_t = np.zeros((Num_Gamma, Num_C))
    Recall_t = np.zeros((Num_Gamma, Num_C))
    Specificity_t = np.zeros((Num_Gamma, Num_C))
    G_Mean_t = np.zeros((Num_Gamma, Num_C))
    F_Mean_t = np.zeros((Num_Gamma, Num_C))
    AUC_t = np.zeros((Num_Gamma, Num_C))

    for j in range(Num_Gamma):
        for k in range(Num_C):
            Accuracy_t[j,k] = np.mean(Accuracy[j,k,:])
            Precision_t[j,k] = np.mean(Precision[j,k,:])
            Recall_t[j,k] = np.mean(Recall[j,k,:])
            Specificity_t[j,k] = np.mean(Specificity[j,k,:])
            G_Mean_t[j,k] = np.mean(G_Mean[j,k,:])
            F_Mean_t[j,k] = np.mean(F_Mean[j,k,:])
            AUC_t[j,k] = np.mean(AUC[j,k,:])

    file_wirte = "Result.txt"
    with open(file_wirte,'a') as w:
        #if First_line:
        #    metrics_list = ["Accuracy", "Precision", "Recall", "Specificity", "G-mean", "F-mean", "AUC"]
        #    first_line = "dataset" + '\t' + "method" + '\t' + '\t'.join(str(x) + '\t' + 'parameters' for x in metrics_list) + '\n'
        #    w.write(first_line)
        #    First_line = False

        line = Dir + '\t' + "RACOG" + '\t'
        accuracy = np.max(Accuracy_t)
        accuracy_parameters = np.argwhere(Accuracy_t == accuracy)
        line += str(accuracy) + '\t' + str('%.3f'% gamma[accuracy_parameters[0][0]]) + \
                ',' + str('%.3f'% C[accuracy_parameters[0][1]]) + '\t'

        precision = np.max(Precision_t)
        precision_parameters = np.argwhere(Precision_t == precision)
        line += str(precision) + '\t' + str('%.3f'% gamma[precision_parameters[0][0]]) + \
                ',' + str('%.3f'% C[precision_parameters[0][1]]) + '\t'

        recall = np.max(Recall_t)
        recall_parameters = np.argwhere(Recall_t == recall)
        line += str(recall) + '\t' + str('%.3f'% gamma[recall_parameters[0][0]]) + \
                ',' + str('%.3f'% C[recall_parameters[0][1]]) + '\t'

        specificity = np.max(Specificity_t)
        specificity_parameters = np.argwhere(Specificity_t == specificity)
        line += str(specificity) + '\t' + str('%.3f'% gamma[specificity_parameters[0][0]]) + \
                ',' + str('%.3f'% C[specificity_parameters[0][1]]) + '\t'

        g_mean = np.max(G_Mean_t)
        g_parameters = np.argwhere(G_Mean_t == g_mean)
        line += str(g_mean) + '\t' + str('%.3f'% gamma[g_parameters[0][0]]) + \
                ',' + str('%.3f'% C[g_parameters[0][1]]) + '\t'

        f_mean = np.max(F_Mean_t)
        f_parameters = np.argwhere(F_Mean_t == f_mean)
        line += str(f_mean) + '\t' + str('%.3f'% gamma[f_parameters[0][0]]) + \
                ',' + str('%.3f'% C[f_parameters[0][1]]) + '\t'

        auc = np.max(AUC_t)
        auc_parameters = np.argwhere(AUC_t == auc)
        line += str(auc) + '\t' + str('%.3f'% gamma[auc_parameters[0][0]]) + \
                ',' + str('%.3f'% C[auc_parameters[0][1]]) + '\n'

        w.write(line)

