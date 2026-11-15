import time
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def evaluation_CART_without_FS(dataset_name, max_depth, k_fold, n_repeats, seed):
    # load data
    dataset_path = "dataframeCP4IM/" + dataset_name + ".csv"
    dataset = pd.read_csv(dataset_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    clf = DecisionTreeClassifier(max_depth=max_depth)

    rskf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats, random_state=seed)

    correct = 0
    duration_each_fold = 0

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

        # obtain the classification accuracy on the test data
        correct = correct + acc

    avg_acc = float(correct) / (k_fold * n_repeats)
    avg_duration = float(duration_each_fold) / (k_fold * n_repeats)

    return avg_acc, avg_duration
