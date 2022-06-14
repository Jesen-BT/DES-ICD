from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
import pandas as pd
from DES import DESforIDS, DESforIDSwr, DESforIDSwd
import numpy as np
from skmultiflow.data.file_stream import FileStream
import os
from online_methods import OOB, UOB, RE_DI


def confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN, beta = 0, 0, 0, 0, 1
    for k in range(len(y_true)):
        if y_true[k] == 1 and y_pred[k] == 1:
            TP += 1
        if y_true[k] == 1 and y_pred[k] == 0:
            FN += 1
        if y_true[k] == 0 and y_pred[k] == 1:
            FP += 1
        if y_true[k] == 0 and y_pred[k] == 0:
            TN += 1

    return TP, FN, FP, TN

def g_mean(y_true, y_pred):
    TP, FN, FP, TN = confusion_matrix(y_true, y_pred)
    if TP + FN == 0:
        gmean = float(TN) / (TN + FP)
    elif TN + FP == 0:
        gmean = float(TP) / (TP + FN)
    else:
        gmean = np.sqrt(float(TP) / (TP + FN) * float(TN) / (TN + FP))

    return gmean

def recall(y_true, y_pred):
    class_one = 0
    class_zero = 0
    minclass = 0
    for i in range(len(y_true)):
        if y_true == 1:
            class_one = class_one + 1
        elif y_true == 0:
            class_zero = class_zero + 1
    if class_one >= class_zero:
        minclass = 0
    elif class_zero >= class_one:
        minclass = 1

    TP, FN = 0, 0

    for i in range(len(y_true)):
        if y_true[i] == minclass:
            if y_true[i] == y_pred[i]:
                TP = TP+1
            else:
                FN = FN+1

    if TP+FN == 0:
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                TP = TP+1
            else:
                FN = FN+1

    Recall = TP/(TP+FN)
    return Recall

def evaluator(models, stream, maxnumber, chunksize, name):
    X, y = stream.next_sample(500)
    for model in models:
        model.partial_fit(X=X, y=y, classes=stream.target_values)

    acc = []
    f1 = []
    gmean = []
    min_recall = []
    ypred = []
    for i in range(len(models)):
        acc.append([])
        f1.append([])
        gmean.append([])
        min_recall.append([])
        ypred.append([])

    yreal = []
    n_samples = 0
    ins_num = []
    while n_samples < maxnumber and stream.has_more_samples():
        n_samples = n_samples + 1
        X, y = stream.next_sample()
        for i in range(len(models)):
            ypred[i].append(models[i].predict(X))
        yreal.append(y)

        if len(yreal) == chunksize:
            for i in range(len(models)):
                # acc[i].append(accuracy_score(y_true=yreal, y_pred=ypred[i]))
                # f1[i].append(f1_score(y_true=yreal, y_pred=ypred[i], average='weighted'))
                gmean[i].append(g_mean(y_true=yreal, y_pred=ypred[i]))
                min_recall[i].append(recall(y_true=yreal, y_pred=ypred[i]))

            for i in range(len(models)):
                ypred[i] = []
            yreal = []
            ins_num.append(n_samples)
            #print(n_samples/maxnumber)


        for i in range(len(models)):
            if i == 5 and n_samples > 2000:
                pass
            else:
                models[i].partial_fit(X=X, y=y)

    # acc.append(ins_num)
    # f1.append(ins_num)
    gmean.append(ins_num)
    min_recall.append(ins_num)
    # out_acc = pd.DataFrame(acc)
    # out_f1 = pd.DataFrame(f1)
    out_gmean = pd.DataFrame(gmean)
    out_recall = pd.DataFrame(min_recall)
    folder_path = "result/"+name+"/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # out_acc.to_csv("result/"+name+"/acc"+".csv")
    # out_f1.to_csv("result/"+name+"/f1"+".csv")
    out_gmean.to_csv("result/"+name+"/add_gmean"+".csv")
    out_recall.to_csv("result/"+name+"/add_recall"+".csv")
    print("finish "+name)

# name = ["HYP", "SEA_abrupt", "SEA_gradual", "sine_abrupt", "sine_gradual", "tree_abrupt", "tree_gradual",
#          "WAVE_abrupt", "WAVE_gradual"]

# name = ["HYP", "SEA_abrupt", "SEA_gradual", "sine_abrupt", "sine_gradual", "tree_abrupt", "tree_gradual",
#          "WAVE_abrupt", "WAVE_gradual", "Elec", "GMSC"]
# name = ["weather"]
# for file in name:
#     stream = FileStream("DATA/"+file+".csv")
#     oob = OOB()
#     uob = UOB()
#     redi = RE_DI(max_classifier=10)
#     tree = HoeffdingTreeClassifier()
#     des = DESforIDS(max_classifier=10)
#     # arf = AdaptiveRandomForestClassifier()
#     # dwm = DynamicWeightedMajorityClassifier()
#     # learn = LearnPPNSEClassifier()
#     # lb = LeveragingBaggingClassifier()
#     # DES_IDS = DESforIDS(window_size=800)
#
#     models = [oob, uob, redi, des, tree]
#     eva = evaluator(models=models, stream=stream, maxnumber=50000, chunksize=500, name=file)


