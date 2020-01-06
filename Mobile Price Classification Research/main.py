import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFECV, SelectFromModel
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import operator
import time
import sys
import os

# Creates a log file with all my prints going in
# Useful for the Eval section and viewing runs without running code


models = ["KNN", "Random Forest", "SVM", "Ridge", "SGD", "Decision Tree", "GNB", "XGBoost"]
top_models = []


# Section 3.1
def outlier_detection(dataset_train):
    """
    :param dataset_train:
    :return:
    """
    train_data = np.array(dataset_train.values.tolist())
    train_features = train_data[:, :-1]
    train_class = train_data[:, -1]
    # As test data has no price range column
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 6))
    sns.boxplot(data=train_features)
    plt.title("Checking for outliers in train data")
    # Shows col 11 has outliers
    plt.show()
    return train_features, train_class


# Section 3.2
def missing_values_check(data):
    """
    :param data: The dataset
    :return: 0 if no missing values or the number of missing values
    """
    if not data.isnull().values.any():
        return 0
    else:
        return data.isnull().values.any()


# Section 3.5
def check_imbalance(data):
    """
    :param data: The training data set
    :return: Value counts of the target class to check if imbalanced
    """
    return data.price_range.value_counts()


# Section 3.4
def scale_data(data):
    """
    :param data: Take in the data
    :return: Scaled Data
    """
    scaler = MinMaxScaler()
    scaler.fit(data)
    # print("Pre-scaled Data : {}".format(data))
    train_features_scaled = scaler.fit_transform(data)
    # print("Scaled Data : {}".format(train_features_scaled))
    return train_features_scaled


# Section 2.2
def feature_importance(dataset, data_train, class_train):
    """
    :param dataset: the pandas instance of the dataset
    :param data_train: the training features of the dataset
    :param class_train: the target labels of the dataset
    :return: A graph showing importances of the features
    """
    # Use a base random forest classifier
    model = return_base_model("Random Forest")
    # fit it with the features and label
    model.fit(data_train, class_train)
    dataset_copy = dataset.copy()
    # Drop the target label
    dataset_copy = dataset_copy.drop(["price_range"], axis=1)
    # Using the copy without the targets
    feature_importances = pd.Series(model.feature_importances_, index=dataset_copy.columns)
    feature_importances.nlargest(20).plot(kind='bar')
    plt.show()


# Helper Function
def return_base_model(modelname):
    """
    :param modelname: Input is a modelname we need for our 8 model comparison
    :return: model with default hyper params
    """
    #   setting random state = 42 for all
    #   keeps same results
    model = None
    if modelname == "KNN":
        model = KNeighborsClassifier()
    elif modelname == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif modelname == "SVM":
        model = SVC(random_state=42)
    elif modelname == "Ridge":
        model = RidgeClassifier(random_state=42)
    elif modelname == "SGD":
        model = SGDClassifier(random_state=42)
    elif modelname == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif modelname == "GNB":
        model = GaussianNB()
    elif modelname == "XGBoost":
        model = XGBClassifier(random_state=42)
    return model


# Helper function
def baseline_model_fit_predict(modelname, data_train, class_train):
    """
    :param modelname: Name of model
    :param data_train: Train Features
    :param train_class: Train Class
    :return: modelname, scores of the model
            classification report
    """
    model = return_base_model(modelname)
    model.fit(data_train, class_train)
    model_scores = cross_val_score(model, data_train, class_train, cv=10)
    return modelname, model_scores


# Helper function
def print_classification_results(name, scores):
    """
    Method to print the classification results
    :param name: Name of the model
    :param scores: Scores after Cross Fold Validation
    """
    print("########## " + name + " #########\n")
    print(name+" - Cross Val Mean : {} , Cross Val Std Dev : {}\n".format(scores.mean(), scores.std()))
    print("########## END OF " + name + " #########\n")


def classification_models(data_train, class_train):
    """
    Method that combines the previous methods,
    This is done WIHTOUT feature selection
    :param data_train: train features
    :param class_train: train class/target label
    :return:
    """
    for item in models:
        model_name, model_scores = baseline_model_fit_predict(item, data_train, class_train)
        print_classification_results(model_name, model_scores)

    # Results
    # KNN - Cross Val Mean : 0.4800000000000001 , Cross Val Std Dev : 0.02368411915187053
    # Random Forest - Cross Val Mean : 0.8625 , Cross Val Std Dev : 0.01479019945774903
    # SVM - Cross Val Mean : 0.8756250000000001 , Cross Val Std Dev : 0.02459833581769302
    # Ridge - Cross Val Mean : 0.5981249999999999 , Cross Val Std Dev : 0.02305733343212089
    # SGD - Cross Val Mean : 0.76 , Cross Val Std Dev : 0.03657184709581948
    # Decision Tree - Cross Val Mean : 0.821875 , Cross Val Std Dev : 0.03140586131600278
    # GNB - Cross Val Mean : 0.7993750000000001 , Cross Val Std Dev : 0.029902184284764228
    # XGBoost - Cross Val Mean : 0.900625 , Cross Val Std Dev : 0.023954709870920993


def optimize(model_name, data_train, class_train):
    """
    GridCV for the top performing models after two rounds of classification
    1. Without Feature Selection/Importance
    2. With Feature Importance
    :param model_name: name of model
    :param data_train:
    :param class_train:
    :return:
    """
    start_time = time.time()
    print("$$$$$$$$$$$$$$$$$$$$$ BEGINNING " + model_name + " Hyperparameter optimization $$$$$$$$$$$$$$$$$$$$")
    if model_name == "SVM":
        param_grid = [{'kernel': ['linear', 'rbf'], 'C': [0.001, 0.01, 0.1, 1, 10],
                       'gamma':["auto", 0.001, 0.01, 0.1, 1], 'decision_function_shape':['ovo', 'ovr']}]
        estimator = SVC(random_state=42)
    elif model_name =="Random Forest":
        param_grid = [{'max_depth': [10, 25, 40, 55, 70, 85, 100],
                       'n_estimators': [100, 500, 1000, 1500, 2000]}]
        estimator = RandomForestClassifier(random_state=42)
    elif model_name == "XGBoost":
        param_grid = [{'min_child_weight': [1, 5, 10],
                        'gamma': [0.5, 1, 1.5, 2, 5],
                        'nthread': [4],
                        'subsample': [0.6, 0.8, 1.0],
                        'colsample_bytree': [0.6, 0.8, 1.0],
                        'max_depth': [3, 4, 5]}]
        estimator = XGBClassifier(random_state=42)
    else:
        param_grid = [{'n_neighbors': list(range(1, 81)), 'p': [1, 2, 3, 4, 5]},
                      {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}]
        estimator = KNeighborsClassifier()
    clf = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=10)
    clf.fit(data_train, class_train)
    print("Best Parameters with "+model_name+":")
    print(clf.best_params_, 'with a score of ', clf.best_score_)
    final_time = time.time()-start_time
    print("Finished running GridSearch on " + model_name + " in {} seconds".format(final_time))
    return clf


# Helper function
def best_models(model_name):
    if model_name == "SVM":
        model = SVC(kernel='linear', C=10, gamma="auto", decision_function_shape="ovo", random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(max_depth=25, n_estimators=1500, random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(colsample_bytree=1.0, gamma=0.5, max_depth=4, min_child_weight=1, nthread=4, subsample=0.6, random_state=42)
    else:
        model = KNeighborsClassifier(n_neighbors=73, p=1)
    return model


# Section 2.4
def uni_variate_feature_selection(model_name, train_scaled_data, train_class):
    """
    :param model_name: Name of  baseline model to use
    :param train_scaled_data: The min-max/normalized data
    :param train_class: The target value
    """
    # Need to use minMaxScaler for chi2 and SelectKBest// NO NEGATIVE VALUES
    # This reduces accuracy
    # Select top 4/ as max. accuracy obtained with 16 indexes removed, leaving 4 so take the 4
    k_best_features = SelectKBest(chi2, k=4).fit_transform(train_scaled_data, train_class)
    model = best_models(model_name)
    model.fit(train_scaled_data, train_class)
    features_scores = cross_val_score(model, train_scaled_data, train_class, cv=10)
    prediction_scores = cross_val_predict(model, train_scaled_data, train_class, cv=10)
    model_report = classification_report(train_class, prediction_scores)
    model_confusion = confusion_matrix(train_class, prediction_scores)
    print(model_name+" Univariate Mean scores : {}".format(features_scores.mean()))
    print(model_name+" Univariate Classification Report :\n {}".format(model_report))
    print(model_name+" Univariate Confusion Matrix :\n {}".format(model_confusion))
    print("End of "+model_name+" univariate feature selection with optimized hyperparameters")


# Section 3.6
def tree_based_feature_selection(models_type, data_train, class_train):
    """
    Feature importance/selection method.
    Recursively remove features and record top 3 performing models
    Calls made to remove_feature_and_plot method to plot and remove features and
    return the max accuracy for a given model
    :params: Test-train split dataset
    :return: list of the top 3 performing (cross fold accuracy-wise) models
    """
    if models_type == "base":
        random_model = RandomForestClassifier(random_state=42)
    else:
        random_model = RandomForestClassifier(max_depth=25, n_estimators=2000, random_state=42)
    # Can uncomment below to check, used the print to Make sure it was using the diff. models
    # Using feature importance with hyper parameter optimized random forest doesnt affet the score anyway
    # print("Random Forest model used for feature selection :{}".format(random_model))
    random_fit = random_model.fit(data_train, class_train)
    rank_models = []
    argsort_features = np.argsort(random_fit.feature_importances_)
    # Uncomment if needed to show ranking
    # top 4 cols are 13 0 11 and 12 i.e RAM Battery Power Px Width and Px Height
    # Sorted in ascending so the best are last
    # print("Argsort of features : {}".format(argsort_features))
    if models_type =="base":
        for item in models:
            max_accuracy = remove_feature_and_plot(item, models_type, argsort_features, data_train,  class_train)
            print(item + " Max Accuracy : {}".format(max_accuracy))
            rank_models.append((item, max_accuracy))
        # operator used as it is slightly faster
        # Sort by descending order
        rank_models.sort(key=operator.itemgetter(1), reverse=True)
        print("Models ranked from best to worst : {}".format(rank_models))
        for i in range(0, 3):
            top_models.append(rank_models[i][0])
    else:
        for item in top_models:
            max_accuracy = remove_feature_and_plot(item, models_type, argsort_features, data_train,
                                                   class_train)
            print(item + " optimized hyperparameter Max Accuracy : {}".format(max_accuracy))
    # Results of Max Accuracy
    # So Top 3 : SVM, Random Forest, XGBoost
    # KNN Max Accuracy : 0.8975
    # Random Forest Max Accuracy : 0.9175000000000001
    # SVM Max Accuracy : 0.942
    # Ridge Max Accuracy : 0.6015
    # SGD Max Accuracy : 0.7665
    # Decision Tree Max Accuracy : 0.869
    # GNB Max Accuracy : 0.8150000000000001
    # XGBoost Max Accuracy : 0.9135

    return top_models


# Section 3.6
def remove_feature_and_plot(model_name, model_type, argsort_features, data_train, class_train):
    """
    :param model_name: Name of the models
    :param model_type: Type of model i.e either base or best three/optimized
    :param argsort_features: Ascending order of important features
    :param data_train: train features
    :param class_train: train target/label
    :return: Max. accuracy (mean) for a given model
    """
    results = []
    number_features = []
    max_accuracy = [(0, 0)]
    for i in range(len(argsort_features) - 1):
        remove = argsort_features[:i + 1]
        removed_train_data = np.delete(data_train, list(remove), axis=1)
        if model_type == "base":
            model = return_base_model(model_name)
        else:
            model = best_models(model_name)
        model.fit(removed_train_data, class_train)
        model_scores = cross_val_score(model, removed_train_data, class_train, cv=10)
        final_score = model_scores.mean()
        results.append(final_score)
        if final_score > max_accuracy[0][1]:
            max_accuracy.pop(0)
            max_accuracy.append((remove, final_score))
        number_features.append(len(remove))
    plt.figure()
    plt.xlabel("Number of features removed")
    plt.ylabel(model_name)
    plt.plot(number_features, results)
    plt.title("Plot of "+model_type+" "+model_name+" as features(least important) removed")
    plt.show()
    # max_accuracy is a list of one tuple
    # (no. of indexes removed to give max accuracy, maxaccuracy)
    # so [0][1] gives max accuracy
    return max_accuracy[0][1]


# Section 2.1
def correlation_matrix(dataset):
    # Develop a correlation between the various features
    correlation_mat = dataset.corr()
    correlation_features = correlation_mat.index
    plt.figure(figsize=(20, 20))
    # Plotting the heatmap
    sns.heatmap(dataset[correlation_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()


# Helper Function
def print_cross_val_scores(model, data_train, class_train, method, name):
    scores = cross_val_score(model, data_train, class_train, cv=10)
    predictions = cross_val_predict(model, data_train, class_train, cv=10)
    model_report = classification_report(class_train, predictions)
    model_confusion = confusion_matrix(class_train, predictions)
    print('Result after '+method+' for ' + name + ' : ', scores.mean())
    print(name + " "+method+" Confusion Matrix :\n {}".format(model_confusion))
    print(name + " "+method+" Classification Report :\n {}".format(model_report))


# Section 2.5
def feature_selection_greedy_search(data_train, class_train):
    for item in top_models:
        # Random Forest takes a lot of time to run
        model = best_models(item)
        rfe_cv = RFECV(model, cv=10)
        rfe_cv.fit(data_train, class_train)
        # Best number of features
        print("Best number of features for greedy search: {}\n".format(rfe_cv.n_features_))
        # Ranking of each feature uncomment if needed
        # print(rfe_cv.ranking_)
        data_train = data_train[:, rfe_cv.support_]
        print_cross_val_scores(model, data_train, class_train, "Greedy/RFECV", item)
        print("################## END OF GREEDY FEATURE SELECTION")


# Section 2.6
def embedded_feature_selection(dataset, data_features, target, model_name):
    optimized_model = best_models(model_name)
    selector = SelectFromModel(optimized_model, max_features=20)
    selector.fit(data_features, target)
    dataset_features = dataset.drop(["price_range"], axis=1)
    embedded_model_support = selector.get_support()
    embedded_model_feature = dataset_features.loc[:, embedded_model_support].columns.tolist()
    number_of_features = len(embedded_model_feature)
    print("The optimized model needs to select {} features for embedded feature selection\n".format(number_of_features))
    # 4 for each model
    data_features = data_features[:, embedded_model_support]
    print_cross_val_scores(optimized_model, data_features, target, "Embedded Feature Selection", model_name)


def run():
    # On windows
    # train_csv = pd.read_csv('D:\\Programming\\CIT\\Labs\\ML\\Assignment2\\dataset\\train.csv')
    # test_csv = pd.read_csv('D:\\Programming\\CIT\\Labs\\ML\\Assignment2\\dataset\\test.csv')
    # On Linux
    # Please uncomment below lines (412-415) if you want to write to log.txt
    # if not os.path.exists('log.txt'):
    #     with open('log.txt', 'w'):
    #         pass
    # sys.stdout = open('log.txt', 'wt')
    train_csv = pd.read_csv('dataset/train.csv')
    print("Missing Values in Train Data : {}".format(missing_values_check(train_csv)))
    print("#########Checking for Imbalance#######")
    print(check_imbalance(train_csv))
    print("#########End of Checking for Imbalance#######")
    train_data, train_class = outlier_detection(train_csv)
    train_scaled_data = scale_data(train_data)
    feature_importance(train_csv, train_scaled_data, train_class)
    correlation_matrix(train_csv)
    classification_models(train_scaled_data, train_class)
    best_models = tree_based_feature_selection("base", train_scaled_data, train_class)
    print("Best models : {}".format(best_models))
    for model in best_models:
        optimize(model, train_scaled_data, train_class)
        # Section 2.4
        uni_variate_feature_selection(model, train_scaled_data, train_class)
        # Section 2.6
        embedded_feature_selection(train_csv, train_scaled_data, train_class, model)
    tree_based_feature_selection("optimized", train_scaled_data, train_class)
    # Section 2.5
    feature_selection_greedy_search(train_scaled_data, train_class)


if __name__ == '__main__':
    run()