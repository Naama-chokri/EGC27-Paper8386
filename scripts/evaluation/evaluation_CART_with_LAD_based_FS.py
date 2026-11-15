import pandas as pd
import numpy as np
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from scripts.lad_based.LAD import lad_based_binary_dataset


def evaluation_CART_with_LAD_based_FS(dataset_name, max_depth, k_fold, n_repeats, seed):
    # load data
    dataset_path = "dataframeCP4IM/" + dataset_name + ".csv"
    dataset = pd.read_csv(dataset_path)
    # matrix X
    X = dataset.iloc[:, :-1].values
    # labels
    y = dataset.iloc[:, -1].values
    # split data into 10 folds
    rskf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats, random_state=seed)

    clf = DecisionTreeClassifier(max_depth=max_depth)

    # perform evaluation on classification task
    correct = 0

    total_time_lad = 0
    total_duration_MHSes = 0
    total_duration_subsets_ranking = 0
    total_length_subset = 0

    list_tuple_subset_acc = []

    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        start_time_lad = time.time()
        # obtain the index of selected subsets on training set
        list_subsets_idx, time_MHSes, time_subsets_ranking = lad_based_binary_dataset(X=X[train_index],
                                                                                      y=y[train_index],
                                                                                      dataset_name=dataset_name,
                                                                                      max_depth=max_depth,
                                                                                      depth=-1,
                                                                                      iteration=i)
        # record time
        time_lad = time.time() - start_time_lad
        total_time_lad = total_time_lad + time_lad
        total_duration_MHSes = total_duration_MHSes + time_MHSes
        total_duration_subsets_ranking = total_duration_subsets_ranking + time_subsets_ranking

        # there are many high-quality subsets after scoring, randomly select one subset from list of subsets
        list_subset = list_subsets_idx[np.random.randint(0, len(list_subsets_idx))]
        # total length subsets
        total_length_subset = total_length_subset + len(list_subset)
        # obtain the dataset on the selected features
        selected_features = X[:, list_subset]
        # train a classification model with the selected features on the training dataset
        clf.fit(selected_features[train_index], y[train_index])
        # predict the class labels of test data
        y_predict = clf.predict(selected_features[test_index])

        # obtain the classification accuracy on the test data
        acc = accuracy_score(y[test_index], y_predict)
        correct = correct + acc
        list_tuple_subset_acc.append((list_subset, acc))

    # output the average classification accuracy over all K folds
    avg_acc = float(correct) / (k_fold * n_repeats)
    # output average duration of lad algorithm
    avg_time_lad = float(total_time_lad) / (k_fold * n_repeats)
    # output average duration for generating MHSes
    avg_time_MHSes = float(total_duration_MHSes) / (k_fold * n_repeats)
    # output average duration for ranking subsets
    avg_time_subsets_ranking = float(total_duration_subsets_ranking) / (k_fold * n_repeats)

    # output average length subset
    avg_length_subset = float(total_length_subset) / (k_fold * n_repeats)
    # the best subset found with the highest accuracy during the whole process
    best_subset = max(list_tuple_subset_acc, key=lambda x: x[1])[0]

    return avg_acc, best_subset, avg_length_subset, avg_time_MHSes, avg_time_subsets_ranking, avg_time_lad
