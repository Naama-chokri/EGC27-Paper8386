#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
# Author: Hao HU
# date: 2019-7-11
# This script aims to get the info in training accuracy, training time, and testing accuracy for
# those decision tree in specific size found by MaxSAT encoding
#
# # print function as in Python3
# ==============================================================================
from __future__ import print_function

# for importing the correct version of pysat and hitman
# ==============================================================================
import os, sys
import numpy as np

from scripts.dtencoder import DTEncoder

from data import Data
from options import Options
from dt import BinaryDecisionTree
from sklearn.model_selection import RepeatedStratifiedKFold


# ===========================================================
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


def maxsatdt_stratified_k_fold_with_repetition_mode(data, options):
    """
        same input and output as the previous one.
    """
    assert options.k_fold > 1
    print("------------In the stratified " + str(options.k_fold) + "-fold cross-validation with " + str(options.n_repeat) + " repetition mode---------------")

    rskf = RepeatedStratifiedKFold(n_splits=options.k_fold, n_repeats=options.n_repeat, random_state=options.seed)
    # set the list of storing results
    maxsatdt_avg_acc_s, maxsatdt_time_s, wcnf_time_s, maxsatdt_sat_s, maxsatdt_depth_s, maxsatdt_nbnode_s = [], [], [], [], [], []
    # 2020-9-2 add lists for new information
    nb_var_s, nb_hard_s, nb_soft_s, nb_literal_s, status_s = [], [], [], [], []

    # for i in range(options.k_fold):
    for i, (train_index, test_index) in enumerate(rskf.split(data.samps, data.labels)):
        print("----------- In the " + str(i + 1) + "-fold validation---------------")
        # set the out_index for each split, which will be given in constructing the decision tree by MaxSAT encoding
        options.out_index = test_index

        # construct the decision tree by MaxSAT
        dtencoder = DTEncoder(data, options)

        # DT(exact)
        if options.nbnodes > 0:
            print("DT(size): the model that learns trees by controlling the size")
            o_res, maxsat_time, sol_file, _, _, s, nb_var, nb_soft, nb_hard, nb_l, wcnf_time = dtencoder.generate_decision_tree_of_maxSAT(
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
                o_res, maxsat_time, sol_file, _, _, s, nb_var, nb_soft, nb_hard, nb_l, wcnf_time = dtencoder.generate_decision_tree_of_maxSAT(
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
        bin_dt = BinaryDecisionTree(sol_path=sol_file, name_feature=data.names)

        cnt_right_test = 0
        cnt_right_train = 0
        cnt_wrong_test = 0
        cnt_wrong_train = 0
        for index in range(len(data.samps)):
            if bin_dt.classify_example(data.samps[index]):
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
        maxsatdt_avg_acc_s.append(cnt_right_test * 1.0 / len(test_index))
        # training accuracy
        maxsatdt_sat_s.append(cnt_right_train * 1.0 / len(train_index))

        maxsatdt_time_s.append(maxsat_time)
        maxsatdt_depth_s.append(bin_dt.get_depth())
        maxsatdt_nbnode_s.append(bin_dt.get_nbnodes())
        nb_var_s.append(nb_var)
        nb_hard_s.append(nb_hard)
        nb_soft_s.append(nb_soft)
        nb_literal_s.append(nb_l)
        status_s.append(s)
        wcnf_time_s.append(wcnf_time)

    opt_list = lambda l: [s for s in l if s.find("OPT") >= 0]
    avg_opt_list = lambda l: len(opt_list(l)) * 1.0 / len(l)
    avg_list = lambda l: sum(l) * 1.0 / len(l)

    return (avg_list(maxsatdt_avg_acc_s), avg_list(maxsatdt_time_s), avg_list(wcnf_time_s), avg_list(maxsatdt_sat_s),
            avg_list(maxsatdt_depth_s), avg_list(maxsatdt_nbnode_s), avg_opt_list(status_s), avg_list(nb_var_s),
            avg_list(nb_soft_s), avg_list(nb_hard_s), avg_list(nb_literal_s))


def maxsatdt_percentage_mode():
    pass


# ===========================================================
if __name__ == '__main__':
    # parsing the command-line options
    options = Options(sys.argv)

    # parsing data
    if options.files:
        data = Data(filename=options.files[0], separator=options.separator,
                    test_ratio=options.test_ratio, ratio=options.ratio, seed=options.seed, sampling=options.sampling)
        file_name = options.files[0].split('/')[-1].split('.')[0]
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
        maxsatdt_avg_acc, maxsatdt_time, wcnf_time, maxsatdt_sat, maxsatdt_depth, maxsatdt_nbnodes, avg_opt, avg_nb_var, avg_nb_s, avg_nb_h, avg_nb_l = maxsatdt_stratified_k_fold_with_repetition_mode(
            data, options)

        # considering the name option of complete solver or incomplete
        complete = '' if options.maxsat_solver_complete == 1 else 'incomplete'
        incomplete_time = '' if options.maxsat_solver_complete == 1 else str(options.maxsat_solver_gtimeout)

        # store the test accuracy and the training time of decision tree found by MaxSAT in txt file
        with open(result_maxsat_dt_path + '/result_' + complete + '_' + str(options.seed) + '_' + incomplete_time +
                  's_using_' + str(options.k_fold) + "_fold_cv_with_" + str(options.n_repeat) + '_repetition.txt', 'w') as f:
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
        print("\n\n\n")
        print(f"After {options.k_fold} folds cross validation with {options.n_repeat} repetitions: ")
        if options.max_depth > 0:
            print(f"For dataset {file_name} using max_depth={options.max_depth}:\n")
            print("#################################################\n"
                 f"#        DT(max), depth={options.max_depth}     #\n"
                  "#################################################")
        elif options.depth > 0:
            print(f"For dataset {file_name} using exact_depth={options.depth}:\n")
            print("#################################################\n"
                  f"#     DT(exact), depth={options.depth}         #\n"
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
