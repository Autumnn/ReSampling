from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

def Initialize_Data(dir):
    Num_lines = len(open(dir, 'r').readlines())
    num_columns = 0
    data_info_lines = 0
    with open(dir, "r") as get_info:
        print("name", get_info.name)
        for line in get_info:
            if line.find("\"RFHRS\"") == 0:
                data_info_lines += 1
            else:
                columns = line.split(",")
                num_columns = len(columns)
                break

    global Num_Samples
    Num_Samples = Num_lines - data_info_lines
    print(Num_Samples)
    global Num_Features
    Num_Features = num_columns - 1

    global Features
    Features = np.ones((Num_Samples, Num_Features))
    global Labels
    Labels = np.ones((Num_Samples, 1))

    global Num_positive
    Num_positive = 0
    global Num_negative
    Num_negative = 0

    with open(dir, "r") as data_file:
        print("Read Data", data_file.name)
        l = 0
        for line in data_file:
            l += 1
            if l > data_info_lines:
                # print(line)
                row = line.split(",")
                length_row = len(row)
                # print('Row length',length_row)
                # print(row[0])
                for i in range(length_row):
                    if i < length_row - 1:
                        Features[l - data_info_lines - 1][i] = row[i]
                        # print(Features[l-14][i])
                    else:
                        attri = row[i].strip()

#                        if attri == '\"Stage 1\"' or attri == '\"Stage 2\"':        # A
                        if attri == '\"Stage 1\"':        # B
                            Labels[l - data_info_lines - 1][0] = 0
                            Num_negative += 1
                            # print(Labels[l-14][0])
                        else:
                            Labels[l - data_info_lines - 1][0] = 1
                            Num_positive += 1

#    print("Number of Positive: ", Num_positive)
    global Positive_Feature
    Positive_Feature = np.ones((Num_positive, Num_Features))
#    print("Num of Negative: ", Num_negative)
    global Negative_Feature
    Negative_Feature = np.ones((Num_negative, Num_Features))
    index_positive = 0
    index_negative = 0

    for i in range(Num_Samples):
        if Labels[i] == 1:
            Positive_Feature[index_positive] = Features[i]
            index_positive += 1
        else:
            Negative_Feature[index_negative] = Features[i]
            index_negative += 1

    print("Read Completed")

def get_feature():
    return Features

def get_label():
    return  Labels

def get_positive_feature():
    return Positive_Feature

def get_negative_feature():
    return Negative_Feature

dir = "KSMOTE_IECON15_InputData.csv"
Initialize_Data(dir)
name = dir.split(".")[0]

print(Positive_Feature[0])
print(Positive_Feature.shape)
print(Negative_Feature[0])
print(Negative_Feature.shape)

npy_name = name + ".npz"
np.savez(npy_name, P_F = Positive_Feature, N_F = Negative_Feature)
file = npy_name

print("File Name: ", file)
name = file.split(".")[0]
dir = file
r = np.load(dir)

Positive_Features = r["P_F"]
Num_Positive = Positive_Features.shape[0]
Positive_Labels = np.linspace(1,1,Num_Positive)
Negative_Features = r["N_F"]
Num_Negative = Negative_Features.shape[0]
Negative_Labels = np.linspace(0,0,Num_Negative)

Features = np.concatenate((Positive_Features, Negative_Features))
Labels = np.concatenate((Positive_Labels, Negative_Labels))

Num_Cross_Folders = 5
min_max_scalar = preprocessing.MinMaxScaler()
Re_Features = min_max_scalar.fit_transform(Features)

#skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=True)
skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=False)

i = 0
for train_index, test_index in skf.split(Re_Features, Labels):
    Feature_train, Feature_test = Re_Features[train_index], Re_Features[test_index]
    Label_train, Label_test = Labels[train_index], Labels[test_index]

    Positive_Feature_train = Feature_train[np.where(Label_train == 1)]
    Positive_Feature_test = Feature_test[np.where(Label_test == 1)]
    Negative_Features_train = Feature_train[np.where(Label_train == 0)]
    Negative_Features_test = Feature_test[np.where(Label_test == 0)]

    saved_name = name + "_" + str(i) + "_Cross_Folder.npz"
    np.savez(saved_name, P_F_tr = Positive_Feature_train, P_F_te = Positive_Feature_test, N_F_tr = Negative_Features_train, N_F_te = Negative_Features_test)

    i += 1