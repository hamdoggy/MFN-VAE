# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader, random_split


# prepare the data_pre_processing
def prepare_data_split_data(path, batch_size=8):
    dataset = CSVDataset2(path)
    train, test = dataset.get_splits(n_test=0.2)

    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl


def prepare_data(train_path, val_path, batch_size=8):
    train_data = CSVDataset2(train_path)
    val_data = CSVDataset2(val_path)

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl


def prepare_data_multi_modl(train_path1, train_path2, train_path3, val_path1,  val_path2,  val_path3, batch_size=8):
    train_data = CSVDatasetMultiModel(train_path1, train_path2, train_path3)
    val_data = CSVDatasetMultiModel(val_path1, val_path2, val_path3)

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl


# ---------------------- data_pre_processing definition ---------------------- #
class CSVDataset1(Dataset):
    # load the data_pre_processing
    def __init__(self, data_path, random_state=2):
        print('Loading data ....')
        data_csv = shuffle(pd.read_csv(data_path), random_state=random_state)
        data = data_csv.drop(["label"], axis=1)
        label = data_csv.loc[:, ['label']]
        print("\t The number of data is {},there are {} features.".format(len(label), len(data.columns)))

        scale = StandardScaler()
        self.x = scale.fit_transform(data)
        self.x = pd.DataFrame(self.x, columns=data.columns)
        self.x = variance_threshold_selector(self.x, 0.0009)

        feature_name = feature_selection_rf(self.x, label)
        data_selected = data.loc[:, feature_name]

        scale2 = StandardScaler()
        self.x = scale2.fit_transform(data_selected)
        self.x, self.y = smote_augment_data(self.x, label)

        self.x, self.y = shuffle(self.x, self.y, random_state=random_state)

        self.x = self.x.astype('float32')
        self.y = self.y.values.astype('float32')

    # number of rows in the data_pre_processing
    def __len__(self):
        return len(self.y)

    # get a row at an index
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2):
        # determine sizes
        test_size = round(n_test * len(self.x))
        train_size = len(self.x) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


class CSVDataset2(Dataset):
    # load the data_pre_processing
    def __init__(self, data_path, random_state=2):
        print('Loading data ....')
        data_csv = shuffle(pd.read_csv(data_path), random_state=random_state)
        data = data_csv.drop(["label"], axis=1)
        label = data_csv.loc[:, ['label']]
        print("\t The number of data is {},there are {} features.".format(len(label), len(data.columns)))

        self.x, self.y = shuffle(data, label, random_state=random_state)

        self.x = self.x.values.astype('float32')
        self.y = self.y.values.astype('float32')

    # number of rows in the data_pre_processing
    def __len__(self):
        return len(self.y)

    # get a row at an index
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2):
        # determine sizes
        test_size = round(n_test * len(self.x))
        train_size = len(self.x) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


class CSVDatasetMultiModel(Dataset):
    # load the data_pre_processing
    def __init__(self, data_path1, data_path2, data_path3, random_state=2):
        # print('Loading data ....')
        DATE1 = pd.read_csv(data_path1)
        DATE2 = pd.read_csv(data_path2)
        DATE3 = pd.read_csv(data_path3)
        data_csv1, data_csv2, data_csv3 = shuffle(DATE1, DATE2, DATE3, random_state=random_state)
        data1 = data_csv1.drop(["label"], axis=1)
        data2 = data_csv2.drop(["label"], axis=1)
        data3 = data_csv2.drop(["label"], axis=1)

        # print(data_csv1.loc[:, ['label']].values.tolist())
        # print(data_csv2.loc[:, ['label']].values.tolist())
        # print(data_csv3.loc[:, ['label']].values.tolist())

        label = data_csv1.loc[:, ['label']]
        # print("\t The number of data is {},there are {} features.".format(len(label), len(data_csv1.columns)))

        self.x1, self.x2, self.x3, self.y = shuffle(data1, data2, data3, label, random_state=random_state)

        self.x1 = self.x1.values.astype('float32')
        self.x2 = self.x2.values.astype('float32')
        self.x3 = self.x3.values.astype('float32')
        self.y = self.y.values.astype('float32')

    # number of rows in the data_pre_processing
    def __len__(self):
        return len(self.y)

    # get a row at an index
    def __getitem__(self, idx):

        return self.x1[idx], self.x2[idx], self.x3[idx], self.y[idx]


# ---------------------- data processing ---------------------- #
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

    rf_features = [features[i] for i in rf_indices]

    print('\t Selected {} features'.format(featureNumber))

    return rf_features
