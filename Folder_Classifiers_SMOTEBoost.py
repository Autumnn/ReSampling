import os
import numpy as np
from sklearn import svm, preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
import Read_Data_UCI as RD
from imblearn.metrics import geometric_mean_score, specificity_score
from SMOTEBoost import SMOTEBoost

path = "UCI_Cross_Folder_npz"
dirs = os.listdir(path) #Get files in the folder
First_line = True

for Dir in dirs:
    print("Data Set Name: ", Dir)
    dir_path = path + "/" + Dir
    files= os.listdir(dir_path) #Get files in the folder

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

        if i == 0:
            print(Num_Positive_train)
            print(Num_Negative_train)
        Num_Create_samples = Num_Negative_train - Num_Positive_train
        Num_estimators = 50
        Num_samples = int(np.ceil(Num_Create_samples / Num_estimators))

        smboost = SMOTEBoost(n_samples=Num_samples, n_estimators=Num_estimators)
        smboost.fit(Features_train_o, Labels_train_o)
        Label_predict = smboost.predict(Feature_test)

        Accuracy_temp[i] = metrics.accuracy_score(Label_test, Label_predict)
        Precision_temp[i] = metrics.precision_score(Label_test, Label_predict)
        Recall_temp[i] = metrics.recall_score(Label_test, Label_predict)
        Specificity_temp[i] = specificity_score(Label_test, Label_predict)
        G_Mean_temp[i] = geometric_mean_score(Label_test, Label_predict)
        F_Mean_temp[i] = metrics.f1_score(Label_test, Label_predict)
        Label_score = smboost.decision_function(Feature_test)
        AUC_temp[i] = metrics.roc_auc_score(Label_test, Label_score)
        i += 1

    Accuracy = np.mean(Accuracy_temp)
    Precision = np.mean(Precision_temp)
    Recall = np.mean(Recall_temp)
    Specificity = np.mean(Specificity_temp)
    G_Mean = np.mean(G_Mean_temp)
    F_Mean = np.mean(F_Mean_temp)
    AUC = np.mean(AUC_temp)

    file_wirte = "Result.txt"
    with open(file_wirte, 'a') as w:
        if First_line:
            metrics_list = ["Accuracy", "Precision", "Recall", "Specificity", "G-mean", "F-mean", "AUC"]
            first_line = "dataset" + '\t' + "method" + '\t' + '\t'.join(
                str(x) + '\t' + 'parameters' for x in metrics_list) + '\n'
            w.write(first_line)
            First_line = False

        line = Dir + '\t' + "SMOTEBoost" + '\t'
        accuracy = Accuracy
        line += str(accuracy) + '\t' + str(0) + \
                ',' + str(0) + '\t'

        precision = Precision
        line += str(precision) + '\t' + str(0) + \
                ',' + str(0) + '\t'

        recall = Recall
        line += str(recall) + '\t' + str(0) + \
                ',' + str(0) + '\t'

        specificity = Specificity
        line += str(specificity) + '\t' + str(0) + \
                ',' + str(0) + '\t'

        g_mean = G_Mean
        line += str(g_mean) + '\t' + str(0) + \
                ',' + str(0) + '\t'

        f_mean = F_Mean
        line += str(f_mean) + '\t' + str(0) + \
                ',' + str(0) + '\t'

        auc = AUC
        line += str(auc) + '\t' + str(0) + \
                ',' + str(0) + '\n'

        w.write(line)


