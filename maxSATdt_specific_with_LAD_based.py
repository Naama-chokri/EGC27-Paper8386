import pandas as pd
import numpy as np
import os, sys
import time
from scripts.lad_based.LAD import lad_based_binary_dataset
from scripts.utility.change_directory_and_delete_file import (
    change_directory_and_delete_all_sub_dataframe_for_binary_dataset,
    change_directory_and_delete_all_input_and_output_hypergraph_files_for_binary_dataset)

from scripts.dtencoder import DTEncoder
from data import Data
from options import Options
from dt import BinaryDecisionTree
from sklearn.model_selection import RepeatedStratifiedKFold


def prepare_result_path(basepath, file_name, test_mode, options):
    path = basepath

    # 2019-12-11: change the name of dt_size considering the "atleast" constraint
    atleast_version = 'atleast_' if options.atleast > 0 else ''
    dt_size = 'dt_' + atleast_version + str(options.nbnodes)
    dt_max_depth = 'max_depth_' + str(options.max_depth)
    dt_depth = 'depth_' + str(options.depth)

    reduced_version = 'original' if options.reduced <= 0 else 'reduced'

    if test_mode >= 8:
        test_way = 'test-ratio_' + str(options.test_ratio)
    elif test_mode >= 4:
        test_way = 'rest'
    elif test_mode >= 2:
        test_way = str(options.k_fold) + '-fold'
    elif test_mode == 1:
        test_way = 'percentage_' + str(options.percentage)
    else:
        print("No such test mode!!")
        exit()

    # set the priority of max_depth setting and size of tree setting

    # DT(max)
    if options.nbnodes > 0:
        # when n is set at first
        if options.max_depth <= 0 and options.depth <= 0:
            folder_names = [file_name, dt_size, reduced_version, test_way]
        elif options.max_depth > 0:
            folder_names = [file_name, dt_size, dt_max_depth, reduced_version, test_way]
        else:
            folder_names = [file_name, dt_size, dt_depth, reduced_version, test_way]

    # DT(exact)
    else:
        # the size is blocked
        if options.max_depth <= 0 and options.depth <= 0:
            print("At least need one parameter indicating the max_depth or size of tree!!")
            exit()
        elif options.max_depth > 0:
            folder_names = [file_name, dt_max_depth, reduced_version, test_way]
        else:
            folder_names = [file_name, dt_depth, reduced_version, test_way]

    for sub_path in folder_names:
        path = path + '/' + sub_path
        try:
            os.makedirs(path)
        except OSError:
            pass
    return path


def evaluation_MaxSAT_dt_specific_with_LAD_based_FS(data, options):
    # load data
    dataset_path = data.fname
    dataset = pd.read_csv(dataset_path)
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    df_label = dataset[["label"]]
    # matrix X
    X = dataset.iloc[:, :-1].values
    # labels
    y = dataset.iloc[:, -1].values
    # split data into 10 folds
    rskf = RepeatedStratifiedKFold(n_splits=options.k_fold, n_repeats=options.n_repeat, random_state=options.seed)

    # set the list of storing results
    maxsatdt_avg_acc_s, maxsatdt_time_s, wcnf_time_s, maxsatdt_sat_s, maxsatdt_depth_s, maxsatdt_nbnode_s = [], [], [], [], [], []
    # 2020-9-2 add lists for new information
    nb_var_s, nb_hard_s, nb_soft_s, nb_literal_s, status_s = [], [], [], [], []

    total_time_lad = 0
    total_duration_MHSes = 0
    total_duration_subsets_ranking = 0
    total_length_subset = 0

    list_tuple_subset_acc = []

    assert options.k_fold > 1
    print("------------In the stratified " + str(options.k_fold) + "-fold cross-validation with " + str(
        options.n_repeat) + " repetition mode---------------")

    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        print("----------- In the " + str(i + 1) + "-fold validation---------------")
        # FIX: exclude the test fold from the MaxSAT encoding, as in maxSATdt_specific.py.
        # Without this line options.out_index keeps its default value [] and the tree
        # is learned on all examples, including the test fold.
        options.out_index = test_index
        start_time_lad = time.time()
        # obtain the index of selected subsets on training set
        list_subsets_idx, time_MHSes, time_subsets_ranking = lad_based_binary_dataset(X=X[train_index],
                                                                                      y=y[train_index],
                                                                                      dataset_name=dataset_name,
                                                                                      max_depth=options.max_depth,
                                                                                      depth=options.depth,
                                                                                      iteration=i)
        # record time
        time_lad = time.time() - start_time_lad
        total_time_lad = total_time_lad + time_lad
        total_duration_MHSes = total_duration_MHSes + time_MHSes
        total_duration_subsets_ranking = total_duration_subsets_ranking + time_subsets_ranking

        print("Number of high-quality subsets: ", str(len(list_subsets_idx)))

        # there are many high-quality subsets after scoring, randomly select one subset from list of subsets
        # FIX (reproducibility): seeded tie-breaking among the best-scoring MSSes
        list_subset = list_subsets_idx[np.random.RandomState(options.seed + i).randint(0, len(list_subsets_idx))]
        list_string_features = ['f_' + str(i) for i in list_subset]
        df_features = dataset[dataset.columns.intersection(list_string_features)]

        sub_dataset = pd.concat([df_features, df_label], axis=1)

        if options.max_depth == -1:  # exact depth
            sub_dataset_path = "sub_dataframe/" + dataset_name + "/" + dataset_name + "_MinFeatures_" + "exact_depth_" + str(
                options.depth) + "_" + str(i) + "_iteration.csv"
            sub_dataset.to_csv(sub_dataset_path, index=False, header=True)
        else:  # max depth
            sub_dataset_path = "sub_dataframe/" + dataset_name + "/" + dataset_name + "_MinFeatures_" + "depth_" + str(
                options.max_depth) + "_" + str(i) + "_iteration.csv"
            sub_dataset.to_csv(sub_dataset_path, index=False, header=True)

        sub_data = Data(filename=sub_dataset_path, separator=options.separator,
                        test_ratio=options.test_ratio, ratio=options.ratio, seed=options.seed,
                        sampling=options.sampling)

        # construct the decision tree by MaxSAT using sub dataframe build on a minimum support set
        dtencoder = DTEncoder(sub_data, options)

        # DT(exact)
        if options.nbnodes > 0:
            print("DT(size): the model that learns trees by controlling the size")
            o_res, maxsat_time, sol_file, _, _, s, nb_var, nb_soft, nb_hard, nb_l, wcnf_time = dtencoder.generate_decision_tree_of_maxSAT_using_min_features(
                N=options.nbnodes, reduced=options.reduced,
                complete=options.maxsat_solver_complete, maxsat_solver=options.maxsat_solver,
                gtimeout=options.maxsat_solver_gtimeout,
                complete_tlimit=options.complete_tlimit, max_depth=options.max_depth, depth=options.depth,
                atleast=options.atleast,
                core_ptime=options.core_ptime, wcnf_timeout=options.wcnf_timeout)

        # DT(max)
        else:
            print("DT(depth): the model that learns trees with an exact depth and an upper bound on the size.")
            if options.max_depth > 0 or options.depth > 0:
                # the upper bound on the size (nodes)
                N = int(np.exp2(options.max_depth + 1)) - 1 if options.max_depth > 0 else int(
                    np.exp2(options.depth + 1)) - 1
                options.atleast = 1
                o_res, maxsat_time, sol_file, _, _, s, nb_var, nb_soft, nb_hard, nb_l, wcnf_time = dtencoder.generate_decision_tree_of_maxSAT_using_min_features(
                    N=N, reduced=options.reduced,
                    complete=options.maxsat_solver_complete, maxsat_solver=options.maxsat_solver,
                    gtimeout=options.maxsat_solver_gtimeout,
                    complete_tlimit=options.complete_tlimit, max_depth=options.max_depth, depth=options.depth,
                    atleast=options.atleast,
                    core_ptime=options.core_ptime, wcnf_timeout=options.wcnf_timeout)
            else:
                print("Must choose the size of tree, the max_depth of tree, or the exact depth of tree!!")
                exit()
        # change the name of solution file by command
        os.rename(sol_file, sol_file.split('.')[0] + "_" + str(i) + ".sol")
        # considering adding an index for each sol file for k-fold
        sol_file = sol_file.split('.')[0] + "_" + str(i) + ".sol"
        # generate the binary decision tree by sol file
        bin_dt = BinaryDecisionTree(sol_path=sol_file, name_feature=sub_data.names)
        cnt_right_test = 0
        cnt_right_train = 0
        cnt_wrong_test = 0
        cnt_wrong_train = 0
        for index in range(len(sub_data.samps)):
            if bin_dt.classify_example(sub_data.samps[index]):
                if index in test_index:
                    cnt_right_test += 1
                else:
                    cnt_right_train += 1
            else:
                if index in test_index:
                    cnt_wrong_test += 1
                else:
                    cnt_wrong_train += 1
        print("cnt_right_test: ", cnt_right_test)
        print("cnt_wrong_test: ", cnt_wrong_test)
        print("cnt_right_train: ", cnt_right_train)
        print("cnt_wrong_train: ", cnt_wrong_train)

        # testing accuracy
        testing_acc = cnt_right_test * 1.0 / len(test_index)
        maxsatdt_avg_acc_s.append(testing_acc)
        list_tuple_subset_acc.append((list_subset, testing_acc))
        # training accuracy
        training_acc = cnt_right_train * 1.0 / len(train_index)
        maxsatdt_sat_s.append(training_acc)

        maxsatdt_time_s.append(maxsat_time)
        maxsatdt_depth_s.append(bin_dt.get_depth())
        maxsatdt_nbnode_s.append(bin_dt.get_nbnodes())
        nb_var_s.append(nb_var)
        nb_hard_s.append(nb_hard)
        nb_soft_s.append(nb_soft)
        nb_literal_s.append(nb_l)
        status_s.append(s)
        wcnf_time_s.append(wcnf_time)

        # total length subsets
        total_length_subset = total_length_subset + len(list_subset)

    # output average duration of lad algorithm
    avg_time_lad = float(total_time_lad) / (options.k_fold * options.n_repeat)
    # output average duration for generating MHSes
    avg_time_MHSes = float(total_duration_MHSes) / (options.k_fold * options.n_repeat)
    # output average duration for ranking subsets
    avg_time_subsets_ranking = float(total_duration_subsets_ranking) / (options.k_fold * options.n_repeat)

    # output average length subset
    avg_length_subset = float(total_length_subset) / (options.k_fold * options.n_repeat)
    # the best subset found with the highest accuracy during the whole process
    best_subset = max(list_tuple_subset_acc, key=lambda x: x[1])[0]

    opt_list = lambda l: [s for s in l if s.find("OPT") >= 0]
    avg_opt_list = lambda l: len(opt_list(l)) * 1.0 / len(l)
    avg_list = lambda l: sum(l) * 1.0 / len(l)

    return ((avg_list(maxsatdt_avg_acc_s), avg_list(maxsatdt_time_s), avg_list(wcnf_time_s), avg_list(maxsatdt_sat_s),
             avg_list(maxsatdt_depth_s), avg_list(maxsatdt_nbnode_s), avg_opt_list(status_s), avg_list(nb_var_s),
             avg_list(nb_soft_s), avg_list(nb_hard_s), avg_list(nb_literal_s), best_subset, avg_length_subset,
             avg_time_MHSes, avg_time_subsets_ranking, avg_time_lad))


if __name__ == '__main__':
    # parsing the command-line options
    options = Options(sys.argv)

    # parsing data
    if options.files:
        data = Data(filename=options.files[0], separator=options.separator,
                    test_ratio=options.test_ratio, ratio=options.ratio, seed=options.seed, sampling=options.sampling)
        file_name = options.files[0].split('/')[-1].split('.')[0]
        print(file_name)
    else:
        data = Data(fpointer=sys.stdin, mapfile=options.mapfile,
                    separator=options.separator)

    if options.approach == 'MaxSATdtencoding':

        # check the max_depth and the size
        if options.nbnodes > 0 and options.max_depth > 0:
            assert options.max_depth >= int(np.log2(options.nbnodes + 1)) - 1

        result_maxsat_dt_path = 'results_maxsatdt_specific'
        assert os.path.exists(result_maxsat_dt_path)
        assert not (options.max_depth > 0 and options.depth > 0)

        # To simplify the test mode, here we just implement the rest mode and the test_ratio mode
        # The k-fold cross validation and percentage could be added afterwards
        # 2019-11-7: Add the option of k-fold cross-validation
        test_mode = 0
        if options.rest:
            test_mode = test_mode + 4
        if options.k_fold > 0:
            test_mode = test_mode + 2
        if not options.percentage == 1:
            test_mode = test_mode + 1
        if options.test_ratio > 0:
            test_mode = test_mode + 8

        result_maxsat_dt_path = prepare_result_path(result_maxsat_dt_path, file_name, test_mode, options)

        # the case of stratified k-fold cross-validation
        maxsatdt_avg_acc, maxsatdt_time, wcnf_time, maxsatdt_sat, maxsatdt_depth, maxsatdt_nbnodes, avg_opt, avg_nb_var, avg_nb_s, avg_nb_h, avg_nb_l, best_subset, avg_length_subset, avg_time_MHSes, avg_time_subsets_ranking, avg_time_lad = evaluation_MaxSAT_dt_specific_with_LAD_based_FS(
            data, options)

        # considering the name option of complete solver or incomplete
        complete = '' if options.maxsat_solver_complete == 1 else 'incomplete'
        incomplete_time = '' if options.maxsat_solver_complete == 1 else str(options.maxsat_solver_gtimeout)

        # store the test accuracy and the training time of decision tree found by MaxSAT in txt file
        with open(result_maxsat_dt_path + '/result_with_LAD_based_' + complete + '_' + str(
                options.seed) + '_' + incomplete_time +
                  's_using_' + str(options.k_fold) + "_fold_cv_with_" + str(options.n_repeat) + '_repetition.txt',
                  'w') as f:
            f.write('avg_test|' + str(maxsatdt_avg_acc) + '\n')
            f.write('avg_train|' + str(maxsatdt_sat) + '\n')
            f.write('time|' + str(maxsatdt_time) + '\n')
            f.write('wcnf_time|' + str(wcnf_time) + '\n')
            f.write('depth|' + str(maxsatdt_depth) + '\n')
            if options.atleast > 0:
                f.write('nbnodes|' + str(maxsatdt_nbnodes) + '\n')
            f.write('opt|' + str(avg_opt) + '\n')
            f.write('nb_var|' + str(avg_nb_var) + '\n')
            f.write('nb_soft|' + str(avg_nb_s) + '\n')
            f.write('nb_hard|' + str(avg_nb_h) + '\n')
            f.write('nb_literal|' + str(avg_nb_l) + '\n')
            # LAD statistics
            f.write('best_subset|' + str(best_subset) + '\n')
            f.write('avg_length_subset|' + str(avg_length_subset) + '\n')
            f.write('avg_time_MHSes|' + str(avg_time_MHSes) + '\n')
            f.write('avg_time_subsets_ranking|' + str(avg_time_subsets_ranking) + '\n')
            f.write('avg_time_lad|' + str(avg_time_lad) + '\n')
            change_directory_and_delete_all_sub_dataframe_for_binary_dataset(
                dataset_name=data.fname.split('/')[-1].split('.')[0],
                max_depth=options.max_depth, depth=options.depth)
            change_directory_and_delete_all_input_and_output_hypergraph_files_for_binary_dataset(
                dataset_name=data.fname.split('/')[-1].split('.')[0],
                max_depth=options.max_depth, depth=options.depth)
        print("\n\n\n")
        print(f"After {options.k_fold} folds cross validation with {options.n_repeat} repetitions: ")
        if options.max_depth > 0:
            print(f"For dataset {file_name} using max_depth={options.max_depth}:\n")
            print("#################################################\n"
                 f"#     LAD + DT(max), depth={options.max_depth}  #\n"
                  "#################################################")
        elif options.depth > 0:
            print(f"For dataset {file_name} using exact_depth={options.depth}:\n")
            print("#################################################\n"
                  f"#    LAD + DT(exact), depth={options.depth}    #\n"
                  "#################################################")
        print('avg_test|' + str(maxsatdt_avg_acc))
        print('avg_train|' + str(maxsatdt_sat))
        print('time|' + str(maxsatdt_time))
        print('wcnf_time|' + str(wcnf_time))
        print('depth|' + str(maxsatdt_depth))
        if options.atleast > 0:
            print('nbnodes|' + str(maxsatdt_nbnodes))
        print('opt|' + str(avg_opt))
        print('nb_var|' + str(avg_nb_var))
        print('nb_soft|' + str(avg_nb_s))
        print('nb_hard|' + str(avg_nb_h))
        print('nb_literal|' + str(avg_nb_l))
        # LAD statistics
        print('best_subset|' + str(best_subset))
        print('avg_length_subset|' + str(avg_length_subset))
        print('avg_time_MHSes|' + str(avg_time_MHSes))
        print('avg_time_subsets_ranking|' + str(avg_time_subsets_ranking))
        print('avg_time_lad|' + str(avg_time_lad))
