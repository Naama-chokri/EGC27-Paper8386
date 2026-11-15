import os
import time
from array import array

import numpy as np
import pandas as pd
import heapq
import random
from tqdm import tqdm
import math

from scripts.subsets_ranking.subset_ranking_binary_dataset import \
    merit_calculation_for_binary_dataset_using_matthews_coef_with_numba
from scripts.utility.get_output_MHSes import get_output_pmmcs_prs_minimal_hitting_set, \
    get_output_pmmcs_prs_minimal_hitting_set_100K, get_output_pmmcs_prs_minimal_hitting_set_1M


def lad_based_binary_dataset(X, y, dataset_name, max_depth, depth, iteration):
    labels = np.unique(y)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            u = X[y == labels[i]]
            v = X[y == labels[j]]
            # u_reshaped = u[:, None, :]
            # v_reshaped = v[None, :, :]
            # xor_result = u_reshaped ^ v_reshaped
            # xor_results = xor_result.reshape((-1, xor_result.shape[2]))

            u_packed = np.packbits(u, axis=1)
            v_packed = np.packbits(v, axis=1)
            xor_result_packed = u_packed[:, None, :] ^ v_packed[None, :, :]
            xor_result_packed = xor_result_packed.reshape((-1, xor_result_packed.shape[2]))
            xor_result_unpacked = np.unpackbits(xor_result_packed, axis=1)[:, :u.shape[1]]

    # arrays_hyper_edges = [np.where(row == 1)[0] for row in xor_results]
    arrays_hyper_edges = [np.where(row == 1)[0] for row in xor_result_unpacked]
    # convert numpy arrays to list of integers
    list_hyper_edges = [list(arr) for arr in arrays_hyper_edges]

    if max_depth == -1:  # condition for exact depth
        hypergraph_file = (
                "output_pmmcs/" + dataset_name + "/hypergraph_sub_dataset/hypergraph_" + dataset_name + "_exact_depth_" +
                str(depth) + "_" + str(iteration) + ".txt")
    else:  # condition for max depth
        # write hypergraph to a text file
        hypergraph_file = (
                "output_pmmcs/" + dataset_name + "/hypergraph_sub_dataset/hypergraph_" + dataset_name + "_depth_" +
                str(max_depth) + "_" + str(iteration) + ".txt")
    with open(hypergraph_file, 'w') as f:
        for row in list_hyper_edges:
            # join integers in each row with space
            row_str = ' '.join(str(i) for i in row)
            f.write(row_str + '\n')

    if max_depth == -1:  # condition for exact depth
        # get the solution of MHSes and save it to a text file
        input_path_for_MHSes = "../output_pmmcs/" + dataset_name + "/hypergraph_sub_dataset/hypergraph_" + dataset_name + "_exact_depth_" + str(
            depth) + "_" + str(iteration) + ".txt"
        output_path_for_MHSes = "../output_pmmcs/" + dataset_name + "/output_hypergraph_sub_dataset/output_" + dataset_name + "_exact_depth_" + str(
            depth) + "_" + str(iteration) + ".txt"
    else:  # condition for max depth
        # get the solution of MHSes and save it to a text file
        input_path_for_MHSes = "../output_pmmcs/" + dataset_name + "/hypergraph_sub_dataset/hypergraph_" + dataset_name + "_depth_" + str(
            max_depth) + "_" + str(iteration) + ".txt"
        output_path_for_MHSes = "../output_pmmcs/" + dataset_name + "/output_hypergraph_sub_dataset/output_" + dataset_name + "_depth_" + str(
            max_depth) + "_" + str(iteration) + ".txt"
    start_time_MHSes = time.time()
    get_output_pmmcs_prs_minimal_hitting_set(input_file=input_path_for_MHSes,
                                             output_file=output_path_for_MHSes,
                                             option_algo="pmmcs", num_thread=0)
    duration_MHSes = time.time() - start_time_MHSes

    os.chdir(f"..")
    # list to store subsets
    list_quality_subsets_idx = []
    max_score = 0

    if max_depth == -1:  # output hypergraph for exact depth
        # define a ranking method to score all the subsets and select the best ones
        output_subsets_path = "output_pmmcs/" + dataset_name + "/output_hypergraph_sub_dataset/output_" + dataset_name + "_exact_depth_" + str(
            depth) + "_" + str(iteration) + ".txt"

    else:  # output hypergraph for max depth
        # define a ranking method to score all the subsets and select the best ones
        output_subsets_path = "output_pmmcs/" + dataset_name + "/output_hypergraph_sub_dataset/output_" + dataset_name + "_depth_" + str(
            max_depth) + "_" + str(iteration) + ".txt"
    start_time_subsets_ranking = time.time()
    with open(output_subsets_path, "r") as output_MHSes:
        for line in tqdm(output_MHSes.readlines()):
            # split the line into integers
            list_subset = [int(num) for num in line.strip().split()]

            # TODO: change ranking score
            score = merit_calculation_for_binary_dataset_using_matthews_coef_with_numba(X[:, list_subset], y)

            if max_score < score:
                if len(list_quality_subsets_idx) > 0:
                    list_quality_subsets_idx.clear()
                max_score = score
                list_quality_subsets_idx.append(list_subset)
            # case when merits score is the same
            elif max_score == score:
                # keep adding the best tuples of that subset and its score
                list_quality_subsets_idx.append(list_subset)
            else:
                continue
    duration_subsets_ranking = time.time() - start_time_subsets_ranking

    return list_quality_subsets_idx, duration_MHSes, duration_subsets_ranking


def lad_based_binary_dataset_Random_Forest(X, y, dataset_name, max_depth, num_estimators, iteration):
    labels = np.unique(y)
    # write hypergraph to a text file
    hypergraph_file = (
            "output_pmmcs/" + dataset_name + "/hypergraph_sub_dataset/hypergraph_" + dataset_name + "_depth_" +
            str(max_depth) + "_for_RF_" + str(iteration) + ".txt")
    with open(hypergraph_file, 'w') as f:
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                u = X[y == labels[i]]
                v = X[y == labels[j]]
                # u_reshaped = u[:, None, :]
                # v_reshaped = v[None, :, :]
                # xor_result = u_reshaped ^ v_reshaped
                # xor_results = xor_result.reshape((-1, xor_result.shape[2]))
                u_packed = np.packbits(u, axis=1)
                v_packed = np.packbits(v, axis=1)
                xor_result_packed = u_packed[:, None, :] ^ v_packed[None, :, :]
                xor_result_packed = xor_result_packed.reshape((-1, xor_result_packed.shape[2]))
                xor_result_unpacked = np.unpackbits(xor_result_packed, axis=1)[:, :u.shape[1]]
        # arrays_hyper_edges = [np.where(row == 1)[0] for row in xor_results]
        arrays_hyper_edges = [np.where(row == 1)[0] for row in xor_result_unpacked]
        # convert numpy arrays to list of integers
        # list_hyper_edges = [list(arr) for arr in arrays_hyper_edges]
        for row in arrays_hyper_edges:
            # join integers in each row with space
            row_str = ' '.join(str(i) for i in list(row))
            f.write(row_str + '\n')
    # get the solution of MHSes and save it to a text file
    input_path_for_MHSes = "../output_pmmcs/" + dataset_name + "/hypergraph_sub_dataset/hypergraph_" + dataset_name + "_depth_" + str(
        max_depth) + "_for_RF_" + str(iteration) + ".txt"
    output_path_for_MHSes = "../output_pmmcs/" + dataset_name + "/output_hypergraph_sub_dataset/output_" + dataset_name + "_depth_" + str(
        max_depth) + "_for_RF_" + str(iteration) + ".txt"
    start_time_MHSes = time.time()
    get_output_pmmcs_prs_minimal_hitting_set_100K(input_file=input_path_for_MHSes,
                                                  output_file=output_path_for_MHSes,
                                                  option_algo="pmmcs", num_thread=20)
    duration_MHSes = time.time() - start_time_MHSes

    os.chdir(f"..")
    # define a ranking method to score all the subsets and select the best ones
    output_subsets_path = "output_pmmcs/" + dataset_name + "/output_hypergraph_sub_dataset/output_" + dataset_name + "_depth_" + str(
        max_depth) + "_for_RF_" + str(iteration) + ".txt"

    list_list_subsets = []
    print("Iteration: ", iteration)
    print("Randomly pick " + str(num_estimators) + " minimum support sets")
    start_time_subsets_ranking = time.time()
    with open(output_subsets_path, "r") as output_MHSes:
        for i, line in tqdm(enumerate(output_MHSes)):
            current_line = list(map(int, line.split()))
            if len(list_list_subsets) < num_estimators:
                if current_line not in list_list_subsets:
                    list_list_subsets.append(current_line)
            else:
                # randomly decide to replace an existing line with decreasing probability, ensuring uniqueness
                r = random.randint(0, i)
                if r < num_estimators and current_line not in list_list_subsets:
                    list_list_subsets[r] = current_line
    duration_subsets_ranking = time.time() - start_time_subsets_ranking

    return list_list_subsets, duration_MHSes, duration_subsets_ranking


def lad_based_binary_dataset_Random_Forest_v2(X, y, dataset_name, max_depth, num_estimators, iteration):
    labels = np.unique(y)
    # write hypergraph to a text file
    hypergraph_file = (
            "output_pmmcs/" + dataset_name + "/hypergraph_sub_dataset/hypergraph_" + dataset_name + "_depth_" +
            str(max_depth) + "_for_RF_" + str(iteration) + ".txt")
    with open(hypergraph_file, 'w') as f:
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                u = X[y == labels[i]]
                v = X[y == labels[j]]
                # u_reshaped = u[:, None, :]
                # v_reshaped = v[None, :, :]
                # xor_result = u_reshaped ^ v_reshaped
                # xor_results = xor_result.reshape((-1, xor_result.shape[2]))
                u_packed = np.packbits(u, axis=1)
                v_packed = np.packbits(v, axis=1)
                xor_result_packed = u_packed[:, None, :] ^ v_packed[None, :, :]
                xor_result_packed = xor_result_packed.reshape((-1, xor_result_packed.shape[2]))
                xor_result_unpacked = np.unpackbits(xor_result_packed, axis=1)[:, :u.shape[1]]
        # arrays_hyper_edges = [np.where(row == 1)[0] for row in xor_results]
        arrays_hyper_edges = [np.where(row == 1)[0] for row in xor_result_unpacked]
        # convert numpy arrays to list of integers
        # list_hyper_edges = [list(arr) for arr in arrays_hyper_edges]
        for row in arrays_hyper_edges:
            # join integers in each row with space
            row_str = ' '.join(str(i) for i in list(row))
            f.write(row_str + '\n')
    # get the solution of MHSes and save it to a text file
    input_path_for_MHSes = "../output_pmmcs/" + dataset_name + "/hypergraph_sub_dataset/hypergraph_" + dataset_name + "_depth_" + str(
        max_depth) + "_for_RF_" + str(iteration) + ".txt"
    output_path_for_MHSes = "../output_pmmcs/" + dataset_name + "/output_hypergraph_sub_dataset/output_" + dataset_name + "_depth_" + str(
        max_depth) + "_for_RF_" + str(iteration) + ".txt"
    start_time_MHSes = time.time()
    get_output_pmmcs_prs_minimal_hitting_set_100K(input_file=input_path_for_MHSes,
                                                  output_file=output_path_for_MHSes,
                                                  option_algo="pmmcs", num_thread=20)
    duration_MHSes = time.time() - start_time_MHSes

    os.chdir(f"..")
    # define a ranking method to score all the subsets and select the best ones
    output_subsets_path = "output_pmmcs/" + dataset_name + "/output_hypergraph_sub_dataset/output_" + dataset_name + "_depth_" + str(
        max_depth) + "_for_RF_" + str(iteration) + ".txt"

    list_list_subsets = []
    print("Randomly pick " + str(num_estimators) + " minimum support sets")
    start_time_subsets_ranking = time.time()
    with open(output_subsets_path, "r") as output_MHSes:
        for i, line in tqdm(enumerate(output_MHSes)):
            current_line = list(map(int, line.split()))
            if len(list_list_subsets) < num_estimators:
                if current_line not in list_list_subsets:
                    list_list_subsets.append(current_line)
            else:
                # randomly decide to replace an existing line with decreasing probability, ensuring uniqueness
                r = random.randint(0, i)
                if r < num_estimators and current_line not in list_list_subsets:
                    list_list_subsets[r] = current_line
    duration_subsets_ranking = time.time() - start_time_subsets_ranking

    return list_list_subsets, duration_MHSes, duration_subsets_ranking


def calculate_number_chunk(u_value, v_value, max_memory_usage):
    dim_u = u_value.shape[0]
    dim_v = v_value.shape[0]
    column_dimension = u_value.shape[1]

    chunk_size_square = (max_memory_usage * math.pow(1024, 3)) / column_dimension

    return math.floor(math.sqrt(chunk_size_square)), dim_u, dim_v, column_dimension


def lad_based_binary_dataset_Random_Forest_with_max_memory(X, y, dataset_name, max_depth, num_estimators, iteration,
                                                           max_mem_usage):
    labels = np.unique(y)
    # write hypergraph to a text file
    hypergraph_file = (
            "output_pmmcs/" + dataset_name + "/hypergraph_sub_dataset/hypergraph_" + dataset_name + "_depth_" +
            str(max_depth) + "_for_RF_" + str(iteration) + ".txt")
    with open(hypergraph_file, 'w') as fout:
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                u = X[y == labels[i]]
                v = X[y == labels[j]]
                chunk_size, dim_u, dim_v, col_dimension = calculate_number_chunk(u, v, max_mem_usage)
                for ii in range(0, dim_u, chunk_size):
                    for jj in range(0, dim_v, chunk_size):
                        # u matrix
                        u_chunk = u[ii:ii + chunk_size]
                        # v matrix
                        v_chunk = v[jj:jj + chunk_size]
                        # packbits to reduce the memory size
                        u_chunk_packed = np.packbits(u_chunk, axis=1)
                        v_chunk_packed = np.packbits(v_chunk, axis=1)
                        res_xor_chunk_packed = u_chunk_packed[:, None, :] ^ v_chunk_packed[None, :, :]
                        res_xor_chunk_packed = res_xor_chunk_packed.reshape((-1, res_xor_chunk_packed.shape[2]))
                        # unpack the results
                        res_xors = np.unpackbits(res_xor_chunk_packed, axis=1)[:, :u_chunk.shape[1]]
                        arrays_hyper_edges = [np.where(row == 1)[0] for row in res_xors]
                        # convert numpy arrays to list of integers
                        # list_hyper_edges = [list(arr) for arr in arrays_hyper_edges]
                        for row in arrays_hyper_edges:
                            # join integers in each row with space
                            row_str = ' '.join(str(i) for i in list(row))
                            fout.write(row_str + '\n')
    # get the solution of MHSes and save it to a text file
    input_path_for_MHSes = "../output_pmmcs/" + dataset_name + "/hypergraph_sub_dataset/hypergraph_" + dataset_name + "_depth_" + str(
        max_depth) + "_for_RF_" + str(iteration) + ".txt"
    output_path_for_MHSes = "../output_pmmcs/" + dataset_name + "/output_hypergraph_sub_dataset/output_" + dataset_name + "_depth_" + str(
        max_depth) + "_for_RF_" + str(iteration) + ".txt"
    start_time_MHSes = time.time()
    get_output_pmmcs_prs_minimal_hitting_set_100K(input_file=input_path_for_MHSes,
                                                  output_file=output_path_for_MHSes,
                                                  option_algo="pmmcs", num_thread=20)
    duration_MHSes = time.time() - start_time_MHSes

    os.chdir(f"..")
    # define a ranking method to score all the subsets and select the best ones
    output_subsets_path = "output_pmmcs/" + dataset_name + "/output_hypergraph_sub_dataset/output_" + dataset_name + "_depth_" + str(
        max_depth) + "_for_RF_" + str(iteration) + ".txt"

    list_list_subsets = []
    print("Iteration: ", iteration)
    print("Randomly pick " + str(num_estimators) + " minimum support sets")
    start_time_subsets_ranking = time.time()
    with open(output_subsets_path, "r") as output_MHSes:
        for i, line in tqdm(enumerate(output_MHSes)):
            current_line = list(map(int, line.split()))
            if len(list_list_subsets) < num_estimators:
                if current_line not in list_list_subsets:
                    list_list_subsets.append(current_line)
            else:
                # randomly decide to replace an existing line with decreasing probability, ensuring uniqueness
                r = random.randint(0, i)
                if r < num_estimators and current_line not in list_list_subsets:
                    list_list_subsets[r] = current_line
    duration_subsets_ranking = time.time() - start_time_subsets_ranking

    return list_list_subsets, duration_MHSes, duration_subsets_ranking
