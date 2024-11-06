import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


def load_data_ns(data_path: str, data_augment: bool = False):
    print('Loading data')

    data_csv = shuffle(pd.read_csv(data_path), random_state=2)

    data = np.array(data_csv.drop(["label"], axis=1))
    label = np.array(data_csv.loc[:, ['label']]).reshape(-1)

    scale = StandardScaler()
    data = scale.fit_transform(data)

    if data_augment:
        data, label = smote_augment_data(data, label)
    data, label = shuffle(data, label, random_state=2)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

    return x_train, x_test, y_train, y_test


def load_data_s(data_path: str):
    print('Loading data ....')

    data_csv = shuffle(pd.read_csv(data_path), random_state=2)
    data = data_csv.drop(["label"], axis=1)
    label = data_csv.loc[:, ['label']]

    print("\t The number of data is {},there are {} features.".format(len(label), len(data.columns)))

    scale = StandardScaler()
    x = scale.fit_transform(data)
    x = pd.DataFrame(x, columns=data.columns)
    x = variance_threshold_selector(x, 0.0009)

    feature_name = feature_selection_rf(x, label)
    data_selected = data.loc[:, feature_name]
    print(data_selected)

    scale2 = StandardScaler()
    x = scale2.fit_transform(data_selected)
    x, y = smote_augment_data(x, label)

    x, y = shuffle(x, y, random_state=2)

    x_train, x_test, y_train, y_test = train_test_split(x, np.array(y).reshape(-1), test_size=0.2)

    return x_train, x_test, y_train, y_test


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


def feature_selection_rf(X_train, Y_train, featureNumber: int = 256):
    print('Filtering features (RandomForest)....')
    model = RandomForestRegressor(random_state=1, max_depth=200)
    model.fit(X_train, Y_train.values.ravel())
    features = X_train.columns
    importances = model.feature_importances_
    rf_indices = np.argsort(importances)[-featureNumber:]  # top 256 features
    #     rf_indices = np.argwhere(importances>0.0003)

    # for f in range(rf_indices.shape[0]):
    #     print("%d. feature %d (%f)" % (f + 1, rf_indices[f], importances[rf_indices[f]]))

    # plt.title('Feature Importances')
    # plt.barh(range(len(rf_indices)), importances[rf_indices], color='b', align='center')
    # plt.yticks(range(len(rf_indices)), [features[i] for i in rf_indices])
    # plt.xlabel('Relative Importance')
    # plt.show()
    rf_features = [features[i] for i in rf_indices]

    print('\t Selected {} features'.format(featureNumber))

    return rf_features
