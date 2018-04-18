import os
import numpy as np
from sklearn import svm, preprocessing, metrics
from keras.models import load_model
from imblearn.over_sampling import SMOTE
import cGANStructure
from RACOG import RACOG
from metrics_list import metric_list

path = "UCI_Cross_Folder_npz_a"
dirs = os.listdir(path) #Get files in the folder
First_line = True

for it in range(5):
    for Dir in dirs:
        print("Data Set Name: ", Dir)
        dir_path = path + "/" + Dir
        files = os.listdir(dir_path)  # Get files in the folder

        methods = ["RACOG", "SVM", "SMOTE", "cGAN", "cGAN-O", "cGAN-SMOTE"]
        for m in methods:
            Num_Gamma = 12
            gamma = np.logspace(-2, 1, Num_Gamma)
            Num_C = 6
            C = np.logspace(-1, 4, Num_C)
            Num_Cross_Folders = 5
            ml_record = metric_list(gamma, C, Num_Cross_Folders)
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

                print(i, " folder; ", "Po_tr: ", Num_Positive_train, "Ne_tr: ", Num_Negative_train,
                      "Po_te: ", Num_Positive_test, "Ne_te: ", Num_Negative_test)

                Features_train_o = np.concatenate((Positive_Features_train, Negative_Features_train))
                Labels_train_o = np.concatenate((Positive_Labels_train, Negative_Labels_train))
                #                print(Labels_train_o)
                Feature_test = np.concatenate((Positive_Features_test, Negative_Features_test))
                Label_test = np.concatenate((Positive_Labels_test, Negative_Labels_test))
                #                print(Label_test)

                if m == "SVM":
                    Feature_train = Features_train_o
                    Label_train = Labels_train_o
                elif m == "SMOTE":
                    sm = SMOTE()
                    Feature_train, Label_train = sm.fit_sample(Features_train_o, Labels_train_o)
                elif m == "cGAN":
                    input_dim, G_dense, D_dense = cGANStructure.Structure(Dir)  # for UCI data
                    Pre_train_epoches = 100
                    Train_epoches = 10000
                    Model_name = "cGAN_" + Dir + "_folder_" + str(i) + "_G-dense_" + str(G_dense) + "_pretrain_" + str(
                        Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_epoches_" + str(Train_epoches) + ".h5"
                    Model_path = "UCI_cross_folder_cGAN_Model"
                    model = load_model(Model_path + "/" + Model_name)
                    num_create_samples = Num_Negative_train - Num_Positive_train
                    Noise_Input = np.random.uniform(0, 1, size=[num_create_samples, input_dim])
                    condition_samples = np.linspace(1, 1, num_create_samples)
                    sudo_Samples = model.predict([Noise_Input, condition_samples])
                    Feature_train = np.concatenate((Features_train_o, sudo_Samples))
                    Label_train = np.concatenate((Labels_train_o, condition_samples))
                elif m == "cGAN-O":
                    input_dim, G_dense, D_dense = cGANStructure.Structure(Dir)  # for UCI data
                    Pre_train_epoches = 100
                    Train_epoches = 10000
                    Model_name = "cGAN_" + Dir + "_folder_" + str(i) + "_G-dense_" + str(G_dense) + "_pretrain_" + str(
                        Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_epoches_" + str(Train_epoches) + ".h5"
                    Model_path = "UCI_cross_folder_cGAN_Model"
                    model = load_model(Model_path + "/" + Model_name)
                    expand_rate_for_Majority = 0.5
                    num_create_samples = int(np.ceil(Num_Negative_train * expand_rate_for_Majority))
                    Noise_Input = np.random.uniform(0, 1, size=[num_create_samples, input_dim])
                    condition_samples = np.linspace(0, 0, num_create_samples)
                    sudo_Samples = model.predict([Noise_Input, condition_samples])
                    Feature_train = np.concatenate((Features_train_o, sudo_Samples))
                    Label_train = np.concatenate((Labels_train_o, condition_samples))
                elif m == "cGAN-SMOTE":
                    input_dim, G_dense, D_dense = cGANStructure.Structure(Dir)  # for UCI data
                    Pre_train_epoches = 100
                    Train_epoches = 10000
                    Model_name = "cGAN_" + Dir + "_folder_" + str(i) + "_G-dense_" + str(G_dense) + "_pretrain_" + str(
                        Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_epoches_" + str(Train_epoches) + ".h5"
                    Model_path = "UCI_cross_folder_cGAN_Model"
                    model = load_model(Model_path + "/" + Model_name)
                    expand_rate_for_Majority = 0.5
                    num_create_samples = int(np.ceil(Num_Negative_train * expand_rate_for_Majority))
                    Noise_Input = np.random.uniform(0, 1, size=[num_create_samples, input_dim])
                    condition_samples = np.linspace(0, 0, num_create_samples)
                    sudo_majority_Samples = model.predict([Noise_Input, condition_samples])
                    Re_Features_o = np.concatenate((Features_train_o, sudo_majority_Samples))
                    Labels_o = np.concatenate((Labels_train_o, condition_samples))
                    sm = SMOTE()
                    Feature_train, Label_train = sm.fit_sample(Re_Features_o, Labels_o)
                elif m == "RACOG":
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
                        ml_record.measure(j, k, i, Label_test, Label_predict)
                        Label_score = clf.decision_function(Feature_test)
                        ml_record.auc_measure(j, k, i, Label_test, Label_score)

                i += 1

            file_wirte = "Result_" + str(it) + ".txt"
            ml_record.output(file_wirte, m, Dir)



