import os
import numpy as np
from sklearn import svm, preprocessing, metrics
from imblearn.over_sampling import SMOTE
from imblearn.metrics import geometric_mean_score

def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j

path = "UCI_Cross_Folder_npz"
dirs = os.listdir(path) #Get files in the folder

for Dir in dirs:
    Over_Sampling_Rate = np.arange(0.1, 1.1, 0.1)
    Num_trails = len(Over_Sampling_Rate)
    G_Mean = np.zeros((Num_trails, 3))
    F_Mean = np.zeros((Num_trails, 3))
    AUC = np.zeros((Num_trails, 3))
    # 1st column: G_Mean value, 2nd column: C value in SVM RBF kernal corresponding to the G_Mean value, 3rd column: Gamma value in SVM RBF kernal corresponding to the G_Mean value

#    if os.path.isdir(dir):
    n_i = 0
    for n_i in range(Num_trails):
        o_r = Over_Sampling_Rate[n_i]

        print("Data Set Name: ", Dir)
        dir_path = path + "/" + Dir

        Num_Gamma = 12
        gamma = np.logspace(-2, 1, Num_Gamma)
        Num_C = 6
        C = np.logspace(-1, 4, Num_C)
        G_Mean_temp = np.zeros((Num_Gamma, Num_C))
        F_Mean_temp = np.zeros((Num_Gamma, Num_C))
        AUC_temp = np.zeros((Num_Gamma, Num_C))

        for j in range(Num_Gamma):
            for k in range(Num_C):
                # print("gamma = ", str(gamma[j]), " C = ", str(C[k]))

                Num_Cross_Folders = 5
                g_temp = np.linspace(0, 0, Num_Cross_Folders)
                f_temp = np.linspace(0, 0, Num_Cross_Folders)
                auc_temp = np.linspace(0, 0, Num_Cross_Folders)

                files = os.listdir(dir_path)
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

                    Features_train_o = np.concatenate((Positive_Features_train, Negative_Features_train))
                    Labels_train_o = np.concatenate((Positive_Labels_train, Negative_Labels_train))

                    Num_Oversampling = int(np.ceil(o_r*(Num_Negative_train - Num_Positive_train)))

                    sm = SMOTE(ratio={'minority':Num_Oversampling})
                    Features_train, Labels_train = sm.fit_sample(Features_train_o, Labels_train_o)
                    clf = svm.SVC(C=C[k], kernel='rbf', gamma=gamma[j])
                    clf.fit(Features_train, Labels_train)

                    Feature_test = np.concatenate((Positive_Features_test, Negative_Features_test))
                    Label_test = np.concatenate((Positive_Labels_test, Negative_Labels_test))
                    Label_predict = clf.predict(Feature_test)

                    g_temp[i] = geometric_mean_score(Label_test, Label_predict)
                    f_temp[i] = metrics.f1_score(Label_test, Label_predict)
                    label_score = clf.decision_function(Feature_test)
                    auc_temp[i] = metrics.roc_auc_score(Label_test, label_score)
                    i += 1

                G_Mean_temp[j, k] = np.mean(g_temp)
                F_Mean_temp[j, k] = np.mean(f_temp)
                AUC_temp[j, k] = np.mean(auc_temp)

        G_Mean[n_i, 0] = np.max(G_Mean_temp)
        F_Mean[n_i, 0] = np.max(F_Mean_temp)
        AUC[n_i, 0] = np.max(AUC_temp)


    file_wirte_AUC = "AUC_Result.txt"
    with open(file_wirte_AUC,'a') as w:
        AUC_line = name + '\t' + "SVM" + '\t'
        AUC_line += '\t'.join(str(x) for x in AUC)
        mean = np.mean(AUC)
        var = np.var(AUC)
        AUC_line += '\t' + str(mean) + '\t' + str(var) + '\n'
        w.write(AUC_line)

    file_wirte_G = "G_Result.txt"
    with open(file_wirte_G, 'a') as w_g:
        G_line = name + '\t' + "SVM" + '\t'
        G_line += '\t'.join(str(x) for x in G_Mean)
        mean = np.mean(G_Mean)
        var = np.var(G_Mean)
        G_line += '\t' + str(mean) + '\t' + str(var) + '\n'
        w_g.write(G_line)

    file_wirte_F = "F_Result.txt"
    with open(file_wirte_F, 'a') as w_f:
        F_line = name + '\t' + "SVM" + '\t'
        F_line += '\t'.join(str(x) for x in F_Mean)
        mean = np.mean(F_Mean)
        var = np.var(F_Mean)
        F_line += '\t' + str(mean) + '\t' + str(var) + '\n'
        w_f.write(F_line)