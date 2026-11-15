import time
from pyexpat import features

import numpy as np
from numpy import std
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def evaluation_RF_sklearn(dataset_name, max_depth, k_fold, n_repeats, seed):
    # load data
    dataset_path = "dataframeCP4IM/" + dataset_name + ".csv"
    dataset = pd.read_csv(dataset_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    clf = RandomForestClassifier(max_depth=max_depth)

    rskf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats, random_state=seed)

    correct = 0
    nodes = 0
    num_features = 0
    duration_each_fold = 0
    list_acc = []
    # perform repeated stratified cross-validation
    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        start = time.time()
        # fit the model
        clf.fit(X[train_index], y[train_index])
        duration = time.time() - start
        duration_each_fold = duration_each_fold + duration
        # predict the class labels of test data
        y_predict = clf.predict(X[test_index])
        # compute testing accuracy
        acc = accuracy_score(y[test_index], y_predict)
        list_acc.append(acc)
        # avg number of nodes across all trees per iteration
        nodes += np.sum([t.tree_.node_count for t in clf.estimators_])
        # number of features used in a forest per iteration
        feature_ = [t.tree_.feature.tolist() for t in clf.estimators_]
        # flatten the list of feature arrays and filter out the leaf nodes (-2)
        all_features = [feature for tree_features in feature_ for feature in tree_features if feature != -2]
        # get the length over the set of feature indices
        num_features_used = len(set(all_features))
        num_features += num_features_used
        # obtain the classification accuracy on the test data
        correct = correct + acc

    avg_acc = float(correct) / (k_fold * n_repeats)
    sd = std(list_acc)
    avg_nodes = float(nodes) / (k_fold * n_repeats)
    avg_features = float(num_features) / (k_fold * n_repeats)
    avg_duration = float(duration_each_fold) / (k_fold * n_repeats)

    return avg_acc, sd, avg_nodes, avg_features, avg_duration
