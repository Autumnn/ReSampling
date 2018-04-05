import math
import numpy as np
from sklearn import svm, preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
import Read_Data_UCI as RD
#import Read_Data as RD
import View_Distribution as VD
import Two_D_Scatter_Plot as TP
from imblearn.over_sampling import SMOTE
from imblearn.metrics import geometric_mean_score

dir = "yeast.data"
#dir = "segment0.dat"
RD.Initialize_Data(dir)

Features = RD.get_feature()
Labels = RD.get_label().ravel()

#VD.view_distribution(RD.get_positive_feature(), RD.get_negative_feature(), "yeast")

Num_Cross_Folders = 5
min_max_scaler = preprocessing.MinMaxScaler()
Re_Features = min_max_scaler.fit_transform(Features)

skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=True)
G_Mean = np.linspace(0, 0, Num_Cross_Folders)
Area_Under_ROC = np.linspace(0, 0, Num_Cross_Folders)
i = 0
for train_index, test_index in skf.split(Re_Features, Labels):
    Feature_train_o, Feature_test = Re_Features[train_index], Re_Features[test_index]
    Label_train_o, Label_test = Labels[train_index], Labels[test_index]

    sm = SMOTE()
    Feature_train, Label_train = sm.fit_sample(Feature_train_o, Label_train_o)

    Attribute = [0,1,2]
    name = "yeast_" + str(i) + "_fold"
    TP.plot(Feature_train_o, Label_train_o, Feature_train, Attribute, name)

    clf = svm.SVC(C=1, kernel='rbf', gamma=0.1)
    clf.fit(Feature_train, Label_train)
    Label_predict = clf.predict(Feature_test)
    G_Mean[i] = geometric_mean_score(Label_test, Label_predict)
    Label_score = clf.decision_function(Feature_test)
    Area_Under_ROC[i] = metrics.roc_auc_score(Label_test, Label_score)
    i += 1

print("G_Mean = ",G_Mean)
print(G_Mean.mean())
print("AUC_ROC = ", Area_Under_ROC)
print(Area_Under_ROC.mean())

