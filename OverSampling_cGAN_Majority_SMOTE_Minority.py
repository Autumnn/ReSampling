import numpy as np
from keras.models import load_model
from sklearn import svm, preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
import os
import cGANStructure
from imblearn.over_sampling import SMOTE
from imblearn.metrics import geometric_mean_score

#  first "min_max_scalar" ant then "StratifiedKFold".

path = "KEEL_npz"
files= os.listdir(path) #Get files in the folder
for file in files:
    print("File Name: ", file)
    name = file.split(".")[0]
    dir = path + "/" + file
    r = np.load(dir)

    Positive_Features = r["P_F"]
    Num_Positive = Positive_Features.shape[0]
    Positive_Labels = np.linspace(1,1,Num_Positive)
    Negative_Features = r["N_F"]
    Num_Negative = Negative_Features.shape[0]
    Negative_Labels = np.linspace(0,0,Num_Negative)
#    Num_Features = Positive_Features.shape[1]

    Features = np.concatenate((Positive_Features, Negative_Features))
    Labels = np.concatenate((Positive_Labels, Negative_Labels))

    Num_Cross_Folders = 5
    min_max_scalar = preprocessing.MinMaxScaler()
    Re_Features = min_max_scalar.fit_transform(Features)

#    input_dim, G_dense, D_dense = cGANStructure.Structure(name)
    input_dim = 10
    G_dense = 300
    D_dense = 150

    Pre_train_epoches = 100
    Train_epoches = 10000
    Model_name = "cGAN_" + name + "_G-dense_" + str(G_dense) + "_pretrain_" + str(Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_epoches_" + str(Train_epoches) + ".h5"
    Model_path = "KEEL_cGAN_Model"
    model = load_model(Model_path + "/" + Model_name)

#skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=True)
    skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=False)
    G_Mean = np.linspace(0,0,Num_Cross_Folders)
    F_Mean = np.linspace(0,0,Num_Cross_Folders)
    AUC = np.linspace(0,0,Num_Cross_Folders)

    i = 0
    for train_index, test_index in skf.split(Re_Features, Labels):
        Feature_train_o, Feature_test = Re_Features[train_index], Re_Features[test_index]
        Label_train_o, Label_test = Labels[train_index], Labels[test_index]
        num_positive = np.array(np.nonzero(Label_train_o)).shape[1]
        num_negative = Label_train_o.shape[0] - num_positive
        if i == 0:
            print(num_positive)
            print(num_negative)
        Delta_samples= num_negative - num_positive

        expand_rate_for_Majority = 0.5
        num_create_Majority_samples = int(np.ceil(num_negative * expand_rate_for_Majority))
        Noise_Input = np.random.uniform(0, 1, size=[num_create_Majority_samples, input_dim])
        condition_samples = np.linspace(0, 0, num_create_Majority_samples)
#        print(Labels.shape)
#        print(condition_samples.shape)
        sudo_majority_Samples = model.predict([Noise_Input, condition_samples])
        Re_Features_o = np.concatenate((Feature_train_o, sudo_majority_Samples))
        Labels_o = np.concatenate((Label_train_o, condition_samples))

        sm = SMOTE()
        Feature_train, Label_train = sm.fit_sample(Re_Features_o, Labels_o)

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
                clf = svm.SVC(C=C[k], kernel='rbf', gamma=gamma[j])
                clf.fit(Feature_train, Label_train)
                Label_predict = clf.predict(Feature_test)

                G_Mean_temp[j, k] = geometric_mean_score(Label_test, Label_predict)
                F_Mean_temp[j, k] = metrics.f1_score(Label_test, Label_predict)
                Label_score = clf.decision_function(Feature_test)
                AUC_temp[j, k] = metrics.roc_auc_score(Label_test, Label_score)

        #    print(G_Mean_temp)
        #    print(F_Mean_temp)

        G_Mean[i] = np.max(G_Mean_temp)
        F_Mean[i] = np.max(F_Mean_temp)
        AUC[i] = np.max(AUC_temp)
        i += 1

    file_wirte_AUC = "AUC_Result.txt"
    with open(file_wirte_AUC, 'a') as w:
        AUC_line = name + '\t' + "cGAN-SMOTE" + '\t'
        AUC_line += '\t'.join(str(x) for x in AUC)
        mean = np.mean(AUC)
        var = np.var(AUC)
        AUC_line += '\t' + str(mean) + '\t' + str(var) + '\n'
        w.write(AUC_line)

    file_wirte_G = "G_Result.txt"
    with open(file_wirte_G, 'a') as w_g:
        G_line = name + '\t' + "cGAN-SMOTE" + '\t'
        G_line += '\t'.join(str(x) for x in G_Mean)
        mean = np.mean(G_Mean)
        var = np.var(G_Mean)
        G_line += '\t' + str(mean) + '\t' + str(var) + '\n'
        w_g.write(G_line)

    file_wirte_F = "F_Result.txt"
    with open(file_wirte_F, 'a') as w_f:
        F_line = name + '\t' + "cGAN-SMOTE" + '\t'
        F_line += '\t'.join(str(x) for x in F_Mean)
        mean = np.mean(F_Mean)
        var = np.var(F_Mean)
        F_line += '\t' + str(mean) + '\t' + str(var) + '\n'
        w_f.write(F_line)