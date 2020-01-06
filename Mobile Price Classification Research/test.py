import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import operator
import time


class Models(object):
    def __init__(self, model, features, target):
        self.model = model
        self.function = {'SVM': self.svm,
                         'KNN': self.knn,
                         'Random Forest': self.random_forest,
                         'GNB': self.gnb,
                         'Ridge': self.ridge,
                         'XGBoost': self.xg_boost,
                         'SGD': self.sgd,
                         'Decision Tree': self.dtc
                         }
        self.features = features
        self.target = target

    def svm(self):
        chosen_model = SVC(random_state=42)
        self.fit_predict(chosen_model)

    def knn(self):
        self.model = KNeighborsClassifier()
        self.fit_predict(self.model)

    @staticmethod
    def random_forest():
        pass

    @staticmethod
    def gnb():
        pass

    @staticmethod
    def ridge():
        pass

    @staticmethod
    def xg_boost():
        pass

    @staticmethod
    def sgd():
        pass

    @staticmethod
    def dtc():
        pass

    @staticmethod
    def fit_predict(model, features, target):
        data_train, data_test, class_train, class_test = train_test_split(features, target, test_size=0.20,
                                                                          random_state=42)
        model.fit(data_train, class_train)
        model_prediction = model.predict(data_test)
        model_scores = cross_val_score(model, data_train, class_train, cv=10)
        model_confusion_matrix = metrics.confusion_matrix(model_prediction, class_test)
        model_classification_report = metrics.classification_report(model_prediction, class_test)
        return model, model_scores, model_confusion_matrix, model_classification_report


    @staticmethod
    def __error():
        raise NotImplementedError("[ERROR] MODEL NOT IMPLEMENTED YET")

    def run(self):
        self.function.get(self.model, self.__error)()


class File(object):
    def __init__(self, file):
        self.file = file
        self.functions = {'Read': self.read}

    def read(self):
        return pd.read_csv(self.file)


class PreProcessing(object):
    def __init__(self, data):
        self.data = data

    @staticmethod
    def __outlier_detection(data):
        train_data = np.array(data.values.tolist())
        train_features = train_data[:, :-1]
        train_class = train_data[:, -1]
        # As test data has no price range column
        sns.set(style="whitegrid")
        plt.figure(figsize=(16, 6))
        ax = sns.boxplot(data=train_features)
        plt.title("Checking for outliers in train data")
        # Shows col 11 has outliers
        plt.show()
        return train_features, train_class

    @staticmethod
    def __missing_check(data):
        if data.isnull().values.any():
            print("Missing Values found and there are {} null values".format(data.isnull().values.any()))
        else:
            print("No Missing Values found !")

    @staticmethod
    def __check_imbalance(data):
        print("Counts of target labels for the dataset\n")
        print(data.price_range.value_counts())
        return data.price_range.value_counts()


    @staticmethod
    def __scale_data(features):
        scaler = MinMaxScaler()
        scaler.fit(features)
        print("Pre-scaled Data : {}".format(features))
        train_features_scaled = scaler.fit_transform(features)
        print("Scaled Data : {}".format(train_features_scaled))
        return train_features_scaled

    def pre_process(self):
        features, target = self.__outlier_detection(self.data)
        print(features.shape)
        self.__missing_check(self.data)
        self.__check_imbalance(self.data)
        train_scaled = self.__scale_data(features)
        return features, target, train_scaled


train_data = File("dataset/train.csv").read()
train_features, train_class, train_scaled_data = PreProcessing(train_data).pre_process()
Models("SVM", train_scaled_data, train_class).run()


