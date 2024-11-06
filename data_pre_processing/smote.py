import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, r_regression


def data_pre_processing(data_path: str, save_path: str):
    data_path = R"C:\Users\Sun\Desktop\MriEC\csv\风险\1原始数据/t2wi&adc&+c_risk.csv"
    print('Loading data ....')
    data_csv = shuffle(pd.read_csv(data_path), random_state=2)
    # data_csv = pd.read_csv(data_path)
    data = data_csv.drop(["label"], axis=1)
    data = data.drop(["patient"], axis=1)
    label = data_csv.loc[:, ['label']]
    patient_name = data_csv.loc[:, ['patient']]

    print("\t The number of data is {},there are {} features.".format(len(label), len(data.columns)))

    scale = StandardScaler()
    x = scale.fit_transform(data)
    x = pd.DataFrame(x, columns=data.columns)
    x = variance_threshold_selector(x, 0.00009)

    feature_name = feature_selection_rf(x, label, featureNumber=768)
    data_selected = data.loc[:, feature_name]

    scale2 = StandardScaler()
    x = scale2.fit_transform(data_selected)
    x, y = smote_augment_data(x, label)
    # y = label

    x, y = shuffle(x, y, random_state=2)

    x = pd.DataFrame(x, columns=feature_name)
    pd.DataFrame(x).to_csv("t2wi&adc&+c_risk.csv")
    y = pd.DataFrame(y)
    pd.DataFrame(y).to_csv("label.csv")


    # # pd.concat([pd_patient_name, pd_t2wi], axis=1)
    # x = pd.DataFrame(x, columns=feature_name)
    # y = pd.DataFrame(y, columns=["label"])
    # pd.concat([y, x], axis=1).to_csv(save_path,index=False)

