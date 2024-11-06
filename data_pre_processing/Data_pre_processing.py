import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, r_regression


def data_pre_processing(data_path: str, save_path: str):
    data_path = R"..\csv\1 original/t1+flair+dwi.csv"
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

    # feature_name = feature_selection_rf(x, label, featureNumber=1521*3)
    # data_selected = data.loc[:, feature_name]

    scale2 = StandardScaler()
    x = scale2.fit_transform(x)
    x, y = smote_augment_data(x, label)
    # y = label

    x, y = shuffle(x, y, random_state=2)

    # x = pd.DataFrame(x, columns=feature_name)
    x = pd.DataFrame(x)
    pd.DataFrame(x).to_csv("t2wi&adc&+c_risk.csv")
    y = pd.DataFrame(y)
    pd.DataFrame(y).to_csv("label.csv")

    # # pd.concat([pd_patient_name, pd_t2wi], axis=1)
    # x = pd.DataFrame(x, columns=feature_name)
    # y = pd.DataFrame(y, columns=["label"])
    # pd.concat([y, x], axis=1).to_csv(save_path,index=False)


def data_pre_processing2(data_path: str):
    data_csv = pd.read_csv(data_path)
    data = data_csv.drop(["label"], axis=1)
    label = data_csv.loc[:, ['label']]

    scale = StandardScaler()
    x = scale.fit_transform(data)
    x = pd.DataFrame(x, columns=data.columns)

    x = Pearson_regression_selector(x, label, threshold=0.3)
    featurename = x

    scale2 = StandardScaler()
    x = scale2.fit_transform(x)
    x, y = smote_augment_data(x, label)

    x = pd.DataFrame(x)
    pd.DataFrame(x).to_csv("feature.csv")
    y = pd.DataFrame(y)
    pd.DataFrame(y).to_csv("label.csv")
    pd.DataFrame(featurename).to_csv("featurename.csv")


def data_pre_processing_multi(data_path1: str, data_path2: str, data_path3: str):
    print('Loading data ....')
    data1 = pd.read_csv(data_path1)
    data2 = pd.read_csv(data_path2)
    data3 = pd.read_csv(data_path3)

    data_csv1, data_csv2, data_csv3 = shuffle(data1, data2, data3, random_state=2)

    data_1 = data_csv1.drop(["label"], axis=1)
    data_1 = data_1.drop(["patient"], axis=1)
    label_1 = data_csv1.loc[:, ['label']]

    data_2 = data_csv2.drop(["label"], axis=1)
    data_2 = data_2.drop(["patient"], axis=1)
    label_2 = data_csv2.loc[:, ['label']]

    data_3 = data_csv3.drop(["label"], axis=1)
    data_3 = data_3.drop(["patient"], axis=1)
    label_3 = data_csv3.loc[:, ['label']]

    scale1_1 = StandardScaler()
    x1 = scale1_1.fit_transform(data_1)
    x1 = pd.DataFrame(x1, columns=data_1.columns)
    x1 = variance_threshold_selector(x1, 0.0009)
    scale1_2 = StandardScaler()
    x2 = scale1_2.fit_transform(data_2)
    x2 = pd.DataFrame(x2, columns=data_2.columns)
    x2 = variance_threshold_selector(x2, 0.0009)
    scale1_3 = StandardScaler()
    x3 = scale1_3.fit_transform(data_3)
    x3 = pd.DataFrame(x3, columns=data_3.columns)
    x3 = variance_threshold_selector(x3, 0.0009)

    feature_name1 = feature_selection_rf(x1, label_1, featureNumber=32)
    data_selected1 = data_1.loc[:, feature_name1]
    feature_name2 = feature_selection_rf(x2, label_2, featureNumber=32)
    data_selected2 = data_2.loc[:, feature_name2]
    feature_name3 = feature_selection_rf(x3, label_3, featureNumber=32)
    data_selected3 = data_3.loc[:, feature_name3]

    scale2_1 = StandardScaler()
    x1 = scale2_1.fit_transform(data_selected1)
    x1, y1 = smote_augment_data(x1, label_1)

    scale2_2 = StandardScaler()
    x2 = scale2_2.fit_transform(data_selected2)
    x2, y2 = smote_augment_data(x2, label_2)

    scale2_3 = StandardScaler()
    x3 = scale2_3.fit_transform(data_selected3)
    x3, y3 = smote_augment_data(x3, label_3)

    x1, y1, x2, y2, x3, y3 = shuffle(x1, y1, x2, y2, x3, y3, random_state=2)

    x1 = pd.DataFrame(x1, columns=feature_name1)
    pd.DataFrame(x1).to_csv("t1wi.csv")
    pd.DataFrame(y1).to_csv("t1wi_label.csv")

    x2 = pd.DataFrame(x2, columns=feature_name2)
    pd.DataFrame(x2).to_csv("flair.csv")
    pd.DataFrame(y2).to_csv("flair_label.csv")

    x3 = pd.DataFrame(x3, columns=feature_name3)
    pd.DataFrame(x3).to_csv("dwi.csv")
    pd.DataFrame(y3).to_csv("dwi_label.csv")


def smote_augment_data(data, target):
    print("SMOTE data augmentation ....")

    overSampler = SMOTE()
    x_smote, y_smote = overSampler.fit_resample(data, target)

    print("\t The number of data after data Augmentation is {}".format(len(y_smote)))
    return x_smote, y_smote


def variance_threshold_selector(data, threshold=0.01):
    print('Removing low variance features ....')

    selector = VarianceThreshold(threshold)
    selector.fit(data)

    featureN = len(data.columns) - len(data.columns[selector.get_support(indices=True)])
    print("\t Removed {} features.".format(featureN))

    return data[data.columns[selector.get_support(indices=True)]]


def Pearson_regression_selector(data, label, threshold=0.4):
    """
    皮尔逊相关系数 特征筛选
    :param data:
    :param label:
    :param threshold:
    :return: features
    """
    print('PCC ....')
    correlation_coefficient = r_regression(data, label)

    get_support = []
    for i in range(len(correlation_coefficient)):
        if correlation_coefficient[i] >= threshold:
            get_support.append(i)
    # print(np.array(get_support))

    assert len(get_support) > 0, "the number of features is 0, you need to reset the threshold"

    return data[data.columns[np.array(get_support)]]


def feature_selection_rf(X_train, Y_train, featureNumber: int = 256):
    print('Filtering features (RandomForest)....')
    model = RandomForestRegressor(random_state=1, max_depth=200)
    model.fit(X_train, Y_train.values.ravel())
    features = X_train.columns
    importances = model.feature_importances_
    rf_indices = np.argsort(importances)[-featureNumber:]  # top 256 features

    rf_features = [features[i] for i in rf_indices]

    print('\t Selected {} features'.format(featureNumber))

    return rf_features


if __name__ == '__main__':
    data_path1 = "../csv/1 original/t1wi.csv"
    data_path2 = "../csv/1 original/flair.csv"
    data_path3 = "../csv/1 original/dwi.csv"

    # data_pre_processing(data_path1,data_path2)
    data_pre_processing_multi(data_path1, data_path2, data_path3)
    # data_pre_processing(data_path, save_path)

    # data_path1 = "../csv/BM_three_model/t1wi.csv"
    # data_path2 = "../csv/BM_three_model/flair.csv"
    # data_path3 = "../csv/BM_three_model/dwi.csv"

    # data_pre_processing_multi(data_path1, data_path2, data_path3)
