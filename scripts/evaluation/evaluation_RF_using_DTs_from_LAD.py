import pandas as pd
import numpy as np
import time
import random
from numpy import std
from scipy.ndimage import standard_deviation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from scripts.lad_based.LAD import lad_based_binary_dataset_Random_Forest
from scipy.stats import mode


def evaluation_RF_using_DTs_from_LAD(dataset_name, max_depth, num_estimators, voting_method, k_fold, n_repeats, seed):
    # load data
    dataset_path = "dataframeCP4IM/" + dataset_name + ".csv"
    dataset = pd.read_csv(dataset_path)
    # matrix X
    X = dataset.iloc[:, :-1].values
    # labels
    y = dataset.iloc[:, -1].values
    # split data into 10 folds
    rskf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats, random_state=seed)

    # perform evaluation on classification task
    correct = 0
    total_time_lad = 0
    total_duration_MHSes = 0
    total_duration_subsets_ranking = 0
    total_length_subset = 0
    total_nodes = 0
    total_num_features = 0
    total_time_rf = 0
    list_acc = []
    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        start_time_rf_first_round = time.time()
        start_time_lad = time.time()
        # obtain the index of selected subsets on training set
        list_list_subsets, time_MHSes, time_subsets_ranking = lad_based_binary_dataset_Random_Forest(X=X[train_index],
                                                                                                     y=y[train_index],
                                                                                                     dataset_name=dataset_name,
                                                                                                     max_depth=max_depth,
                                                                                                     num_estimators=num_estimators,
                                                                                                     iteration=i)
        # record time
        time_lad = time.time() - start_time_lad
        total_time_lad = total_time_lad + time_lad
        total_duration_MHSes = total_duration_MHSes + time_MHSes
        total_duration_subsets_ranking = total_duration_subsets_ranking + time_subsets_ranking

        if voting_method == "majority_votes":
            # initialize an empty array to store predictions
            predictions_array = np.zeros((num_estimators, len(test_index)), dtype=int)

        elif voting_method == "soft_votes":
            # initialize an empty array to store probabilities
            probabilities_array = np.zeros((num_estimators, len(test_index), 2))
        else:
            print("Please use a voting strategy")
            return
        time_first_round = time.time() - start_time_rf_first_round
        time_for_all_trees = 0
        length_subset_for_all_trees = 0
        num_nodes = 0
        list_unique_features = []
        # loop to train each tree with minimum support set and apply majority votes
        for idx_subset in range(len(list_list_subsets)):
            length_subset_for_all_trees = length_subset_for_all_trees + len(list_list_subsets[idx_subset])
            print("Each Support Set used to build each Decision Tree: ", set(list_list_subsets[idx_subset]))
            # obtain the dataset on the selected features
            selected_features = X[:, list_list_subsets[idx_subset]]
            # each decision tree with specific max depth
            each_clf = DecisionTreeClassifier(max_depth=max_depth)
            # start time for each tree
            start_time_each_tree = time.time()
            # generate each bootstrap dataset independently for each decision tree
            train_index_bootstrap = random.choices(train_index, k=len(train_index))
            # train a tree model with each minimum support set on the training dataset
            # each_clf.fit(selected_features[train_index], y[train_index])
            # build a tree model with each minimum support set on each bootstrap dataset
            each_clf.fit(selected_features[train_index_bootstrap], y[train_index_bootstrap])
            # store number of nodes for each tree per iteration
            num_nodes += each_clf.tree_.node_count
            # feature
            feature_for_each_tree_ndarray = each_clf.tree_.feature
            # convert ndarray to list
            feature_for_each_tree = feature_for_each_tree_ndarray.tolist()
            # flatten the list of feature arrays and filter out leaf nodes (-2)
            unique_features_for_each_tree = [f for f in feature_for_each_tree if f != -2]
            # print("unique_features_for_each_tree: ", unique_features_for_each_tree)
            list_unique_features.append(unique_features_for_each_tree)
            # duration for each tree
            time_each_tree = time.time() - start_time_each_tree
            time_for_all_trees += time_each_tree
            if voting_method == "majority_votes":
                # predict the class labels of test data
                y_predict_each_classifier = each_clf.predict(selected_features[test_index])
                predictions_array[idx_subset] = y_predict_each_classifier
            elif voting_method == "soft_votes":
                y_predict_prob_each_classifier = each_clf.predict_proba(selected_features[test_index])
                probabilities_array[idx_subset] = y_predict_prob_each_classifier

        total_time_rf_each_fold = time_first_round + time_for_all_trees
        total_time_rf += total_time_rf_each_fold

        avg_length_subset_for_all_trees = length_subset_for_all_trees / len(list_list_subsets)

        total_length_subset += avg_length_subset_for_all_trees
        total_nodes += num_nodes
        # flatten the list of lists and get the unique features (union) for a forest
        unique_features = set([f for each_tree_list in list_unique_features for f in each_tree_list])
        # print("unique_features_for_a_forest: ", unique_features)
        # get the total number of unique features in a forest for each iteration
        num_unique_features = len(unique_features)
        total_num_features += num_unique_features

        if voting_method == "majority_votes":
            # transpose to group predictions by instance
            transposed = predictions_array.T
            # compute majority votes among all decision trees for each instance
            y_predict = mode(transposed, axis=1).mode.flatten()
        elif voting_method == "soft_votes":
            # sum probability for both classes across different DTs
            summed_probabilities = np.sum(probabilities_array, axis=0)
            # predict the class with the highest summed probability
            y_predict = np.argmax(summed_probabilities, axis=1)

        # obtain the classification accuracy on the test data
        acc = accuracy_score(y[test_index], y_predict)
        list_acc.append(acc)
        correct = correct + acc

    # output the average classification accuracy over all K folds
    avg_acc = float(correct) / (k_fold * n_repeats)
    std_value = std(list_acc)
    # output average duration of lad algorithm
    avg_time_lad = float(total_time_lad) / (k_fold * n_repeats)
    # output average duration for generating MHSes
    avg_time_MHSes = float(total_duration_MHSes) / (k_fold * n_repeats)
    # output average duration for ranking subsets
    avg_time_subsets_ranking = float(total_duration_subsets_ranking) / (k_fold * n_repeats)
    avg_time_rf = float(total_time_rf) / (k_fold * n_repeats)

    # output average length subset
    avg_length_subset = float(total_length_subset) / (k_fold * n_repeats)

    # output average nodes across all trees
    avg_nodes = float(total_nodes) / (k_fold * n_repeats)
    # average number of features used in a forest
    avg_num_features = float(total_num_features) / (k_fold * n_repeats)

    return avg_acc, std_value, avg_length_subset, avg_nodes, avg_num_features, avg_time_MHSes, avg_time_subsets_ranking, avg_time_lad, avg_time_rf
