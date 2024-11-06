import os
import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import KFold
from shutil import copy, rmtree


def Single_mode():
    data_path = r"../csv/shap/ALL.csv"
    data_csv = genfromtxt(data_path, delimiter=',')

    folds = KFold(n_splits=5, shuffle=True, random_state=1)

    for fold_i, (train_index, val_index) in enumerate(folds.split(data_csv)):

        train_data, val_data = [], []

        for i in train_index:
            train_data.append(list(data_csv[i]))
        for i in val_index:
            val_data.append(list(data_csv[i]))

        train_data_csv = pd.DataFrame(train_data)
        val_data_csv = pd.DataFrame(val_data)

        pd.DataFrame(train_data_csv).to_csv("+c_" + str(fold_i) + "_train.csv", index=False)
        pd.DataFrame(val_data_csv).to_csv("+c_" + str(fold_i) + "_val.csv", index=False)

        break


def Multi_mode():
    data_path1 = "../csv/t1wi.csv"
    data_path2 = "../csv/flair.csv"
    data_path3 = "../csv/dwi.csv"

    data_csv1 = genfromtxt(data_path1, delimiter=',')
    data_csv2 = genfromtxt(data_path2, delimiter=',')
    data_csv3 = genfromtxt(data_path3, delimiter=',')

    folds = KFold(n_splits=5, shuffle=True, random_state=1)

    for fold_i, (train_index, val_index) in enumerate(folds.split(data_csv1)):

        train_data1, val_data1 = [], []
        train_data2, val_data2 = [], []
        train_data3, val_data3 = [], []

        for i in train_index:
            train_data1.append(list(data_csv1[i]))
            train_data2.append(list(data_csv2[i]))
            train_data3.append(list(data_csv3[i]))

        for i in val_index:
            val_data1.append(list(data_csv1[i]))
            val_data2.append(list(data_csv2[i]))
            val_data3.append(list(data_csv3[i]))

        train_data_csv1 = pd.DataFrame(train_data1)
        val_data_csv1 = pd.DataFrame(val_data1)
        pd.DataFrame(train_data_csv1).to_csv("t1wi_crossVal_" + str(fold_i) + "_train.csv", index=False)
        pd.DataFrame(val_data_csv1).to_csv("t1wi_crossVal_" + str(fold_i) + "_val.csv", index=False)

        train_data_csv2 = pd.DataFrame(train_data2)
        val_data_csv2 = pd.DataFrame(val_data2)
        pd.DataFrame(train_data_csv2).to_csv("flair_crossVal_" + str(fold_i) + "_train.csv", index=False)
        pd.DataFrame(val_data_csv2).to_csv("flair_crossVal_" + str(fold_i) + "_val.csv", index=False)

        train_data_csv3 = pd.DataFrame(train_data3)
        val_data_csv3 = pd.DataFrame(val_data3)
        pd.DataFrame(train_data_csv3).to_csv("dwi_crossVal_" + str(fold_i) + "_train.csv", index=False)
        pd.DataFrame(val_data_csv3).to_csv("dwi_crossVal_" + str(fold_i) + "_val.csv", index=False)


if __name__ == '__main__':
    Single_mode()


