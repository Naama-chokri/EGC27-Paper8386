#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
# Author: Hao HU
# date: 2019-4-1
# The class of Sampler aims to make abstract of sampling for the dataset
# There are some ways of sampling realized below:(in updating):
#       1. stratified_sampling: 
#           global random sampling, but conserving the percentage of positive and negative examples in the sampling
#           That is a kind of sampling without putting back.
#       2. Bootstrap:(waited in 2019-4-1)
#           global random sampling with putting back, which will not conserving the percentage of positive and negative
#           That will be used for the random forest

# # print function as in Python3
#==============================================================================
from __future__ import print_function

import random
import time


#==============================================================================
class Sampler(object):
    """
        Class for realizing the the different kind of sampling function
    """

    def __init__(self, approach='stratified', k=0, percentage=0.25, seed=2019):
        """
            Constructor
        """
        self.approach = approach
        # here the percentage means the percentage of sampling
        self.percentage = percentage
        self.k = k
        random.seed(seed)

    def do_sampling_by_per(self, array_index, array_labels):
        """
            The number of sampling depends on the percentage.
            return the array of index which is sampled.
        """
        if not len(array_index) == len(array_labels):
            raise ValueError("The length of labels does not equal the index!!")

        if self.approach == 'stratified':
            selected_index = self.stratified_sampling(array_index, array_labels)
        elif self.approach == 'bootstrap':
            selected_index = self.bootstrap_sampling(array_index, array_labels)
        else:
            raise ValueError("The " + self.approach + " method is not available in the Sample class!!")

        return selected_index

    def do_sampling_by_k(self, array_index, array_labels):
        """
            sampling by the k_fold. 
            Return the array of k items, and each item contains 1/k samples for the whole dataset
        """
        if self.k <= 1 or self.k > len(array_index):
            raise ValueError("The k can not be equal or less than 1, and can not be greater than the size of data")

        assert len(array_index) == len(array_labels)

        # considering the k-fold can only use the stratified sampling
        k_fold_index_s = self.get_k_fold(array_index, array_labels)

        return k_fold_index_s

    def do_sampling_by_k_pure(self, array_index):
        """
            2020-4-3 Hao HU
            sampling in the pure k-fold cross-validation, without perserving the class distribution
            Return the array of k items, each item contains one fold for the whole dataset
        """
        if self.k <= 1 or self.k > len(array_index):
            return ValueError("The k can not be equal or less than 1, and can not be greater than the size of dataset")

        # considering the pure random k-fold cross-validation
        k_fold_index_s = self.get_k_fold_pure(array_index)

        return k_fold_index_s

    def stratified_sampling(self, array_index, array_labels):
        # divide the index into the positive class and negative class
        positive_example_index = []
        negative_example_index = []

        # here considering the index in the array_index
        for i in range(len(array_index)):
            if array_labels[i] == 0:
                negative_example_index.append(array_index[i])
            else:
                positive_example_index.append(array_index[i])

        part_pos_num = int(len(positive_example_index) * self.percentage)
        part_neg_num = int(len(negative_example_index) * self.percentage)

        pos_out = []
        neg_out = []

        for i in range(part_pos_num):
            r_index = random.randint(0, len(positive_example_index) - 1)
            pos_out.append(positive_example_index[r_index])
            del positive_example_index[r_index]

        for i in range(part_neg_num):
            r_index = random.randint(0, len(negative_example_index) - 1)
            neg_out.append(negative_example_index[r_index])
            del negative_example_index[r_index]

        # return the chosen index from the array_index
        return pos_out + neg_out

    # 2019-5-15 Hao HU
    def bootstrap_sampling(self, array_index, array_labels):
        """
            the base idea of bootstrap sampling is the sampling with replacement. It could be represented as 3 steps:
                1. randomly choose a sample from raw data set
                2. copy it, and put it back, which makes it could be chosen at the next time
                3. repeat the two steps before until the sample set size grows as the same with the raw data set
            Here, considering the ratio used, we will not sampling the size of dataset m, just sampling m*percentage size
            The return value of this function will also be the index, which will have repated index.
        """
        bootstarp_size = len(array_index) * self.percentage
        selected_index = []
        # sampling with replacememt
        for i in range(bootstarp_size):
            r_index = random.randint(0, len(array_index) - 1)
            selected_index.append(r_index)
        return selected_index

    def get_k_fold(self, array_index, array_label):
        # divide the pos and neg
        p_e_index = []
        n_e_index = []
        for i in range(len(array_index)):
            if array_label[i] == 0:
                n_e_index.append(i)
            else:
                p_e_index.append(i)

        # change for python3 !!! -> add the operation of int
        # becasue in python3, 3/2 = 1.5 not 1 like python 2.7
        part_p_num = int(len(p_e_index) / self.k)
        rest_p_num = len(p_e_index) % self.k
        part_n_num = int(len(n_e_index) / self.k)
        rest_n_num = len(n_e_index) % self.k

        k_fold_index_s = []
        for i in range(self.k):
            k_part_index = []
            part_p_num_temp = part_p_num
            part_n_num_temp = part_n_num
            if rest_p_num > 0:
                part_p_num_temp = part_p_num_temp + 1
                rest_p_num = rest_p_num - 1
            if rest_n_num > 0:
                part_n_num_temp = part_n_num_temp + 1
                rest_n_num = rest_n_num - 1
            for pos_i in range(part_p_num_temp):
                random_index = random.randint(0, len(p_e_index) - 1)
                k_part_index.append(p_e_index[random_index])
                del p_e_index[random_index]

            for neg_i in range(part_n_num_temp):
                random_index = random.randint(0, len(n_e_index) - 1)
                k_part_index.append(n_e_index[random_index])
                del n_e_index[random_index]

            k_fold_index_s.append(k_part_index)
        return k_fold_index_s

    def get_k_fold_pure(self, array_index):
        # no need to divide the positive and negative
        whole_size = len(array_index)
        array_index_c = list(array_index)

        part_num = int(whole_size / self.k)
        rest_num = int(whole_size % self.k)

        k_fold_index_s = []

        for i in range(self.k):
            k_part_index = []
            part_num_temp = part_num
            if rest_num > 0:
                part_num_temp += 1
                rest_num -= 1
            for index_item in range(part_num_temp):
                random_index = random.randint(0, len(array_index_c) - 1)
                #print(random_index)
                k_part_index.append(array_index_c[random_index])
                #print("Out")
                del array_index_c[random_index]

            k_fold_index_s.append(k_part_index)
        return k_fold_index_s
