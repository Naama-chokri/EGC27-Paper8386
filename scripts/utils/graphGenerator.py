#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
# Author: Hao HU
# date: 2019-4-1
# The class of GraphGenerator aims to simplify the generating, storing different kind of images in this project
# Now, three kinds of images are supported(can add after):
#       1. Average accuracy 
#       2. Mid accuracy
#       3. total time of training

# # print function as in Python3
#==============================================================================
from __future__ import print_function

import inspect, os, sys
sys.path.insert(0, os.path.join(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])), '../pysat-module/'))
sys.path.insert(0, os.path.join(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])), '../hitman/py/'))

import matplotlib.pyplot as plt

#==============================================================================
class GraphGenerator(object):
    """
        class for drawing and saving figures describing the accuracy and time
    """
    def __init__(self, base_path, dataset_name="dataset", create_path=1, sat_way='maxSAT', test_way='rest', reduced=0, sampling='stratified'):
        """
            add create_path option to create the folder automaticly or not
        """
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.sat_way = sat_way
        self.test_way = test_way
        self.reduced = reduced
        self.reduced_name = "" if self.reduced == 0 else "reduced_"
        self.sampling = sampling
        # create the selected folder, and then change the base path
        if create_path == 1:
            self.base_path = self.prepare_valid_fig_path()
        else:
            pass


    def prepare_valid_fig_path(self):
        """
            function for creating the folder if it does not exist
            The folder structure is described in the report 
        """
        folder_names = [self.dataset_name, self.sat_way, self.sampling, self.test_way]
        path = self.base_path
        for sub_path in folder_names:
            path = path + '/' + sub_path
            if not os.path.exists(path):
                os.mkdir(path)
        return path
    
    def generate_figure_general(self, data_x, data_y, x_label, y_label, title, fig_name, y_range=[]):
        """
            generate the figure of average information received from the cluster
        """
        fig = plt.figure()
        plt.plot(data_x, data_y, linewidth=2, marker='.')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(data_x)
        if not len(y_range) == 0:
            plt.yticks(y_range)
        plt.savefig(self.base_path + '/' + fig_name)
        plt.close()

    # 2019-6-27: function for plotting multiple lines in the same figure
    def generate_figure_multiple_lines_general(self, data_x_array, data_y_array, x_label, y_label, title, fig_name, y_range=[], line_legend=[]):
        """
            line_legend -> an option to set the name of lines, which has the same size as data_x_array
        """
        if not len(data_x_array) == len(data_y_array):
            raise ValueError("The x-axis value is not in the size with y-axis value")
        plt.figure(figsize=(6,4))
        markers = ['o', 'v', '+', 'x', 'd']
        for i in range(len(data_x_array)):
            plt.plot(data_x_array[i], data_y_array[i], marker=markers[i])
        plt.title(title)

        # set the xticks as the longest size in the data_x_array
        x_sizes = [len(i) for i in data_x_array]
        x_label_index = x_sizes.index(max(x_sizes))
        plt.xticks(data_x_array[x_label_index])
        if not len(y_range) == 0:
            plt.yticks(y_range)
        if len(line_legend) == len(data_y_array):
            plt.legend(line_legend, loc='best')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(self.base_path + '/' + fig_name)
        plt.close()        


    def generate_figure_by_selected_options(self, mode, data_x, data_y, ratio=1, per=1, k=0, seed=2019):
        """
            mode -> indicates the kind of figure generated.
                0 -> time(for k_fold, is the average time of training for each fold)
                1 -> accuracy(for k_fold, is the average accuracy of testing set)
                2 -> middle accuracy, only valid for k_fold
                3 -> percent of satisfait examples in maxSAT, only valid for maxSAT
        """
        if mode < 0 or mode > 3:
            raise ValueError("The mode chosen is not valid!!!")

        if not len(data_x) == len(data_y):
            raise ValueError("The length of data in x dimension and y dimension does not equal!!")

        fig_name, fig_title, fig_x_name, fig_y_name = self.generate_name_content_of_figure(mode, ratio, per=per, k=k, seed=seed)

        # generate the figure
        fig = plt.figure()
        plt.plot(data_x, data_y, linewidth=2, marker='.')
        plt.xlabel(fig_x_name)
        plt.ylabel(fig_y_name)
        plt.title(fig_title)
        plt.xticks(data_x)
        plt.savefig(self.base_path + '/' + fig_name)
        plt.close()

    
    def generate_name_content_of_figure(self, mode, ratio, per=1, k=0, seed=2019):
        fig_name = ""
        fig_title = ""
        fig_x_name = "number of nodes"
        fig_y_name = ""
        # for the time figure
        if mode == 0:
            fig_title = self.sat_way + " in " + self.test_way + " mode: The time of training " + self.reduced_name
            fig_y_name = "time used in training"
            if self.test_way == 'rest':
                fig_name = "rest_time_" + str(ratio) + "_" + str(seed) + self.reduced_name +  ".png"
            elif self.test_way == 'percentage':
                fig_name = "per_time_" + str(per) + "_" + str(ratio) + "_" + str(seed) + self.reduced_name + ".png"
            elif self.test_way == 'k_fold':
                fig_name = "kfold_time_" + str(k) + "_" + str(ratio) + "_" + str(seed) + self.reduced_name + ".png"
            else:
                raise ValueError("The test way chosen is not valid!!")
        elif mode == 1:
            fig_title = self.sat_way + " in " + self.test_way + " mode: The accuracy of testing set " + self.reduced_name
            fig_y_name = "accuracy of testing set"
            if self.test_way == 'rest':
                fig_name = "rest_acc_" + str(ratio) + "_" + str(seed) + self.reduced_name + ".png"
            elif self.test_way == 'percentage':
                fig_name = "per_acc_" + str(per) + "_" + str(ratio) + "_" + str(seed) + self.reduced_name + ".png"
            elif self.test_way == 'k_fold':
                fig_name = "kfold_avg_acc_" + str(k) + "_" + str(ratio) + "_" + str(seed) + self.reduced_name + ".png"
            else:
                raise ValueError("The test way chosen is not valid!!")
        elif mode == 2 and self.test_way == 'k_fold':
            fig_title = self.sat_way + " in " + self.test_way + " mode: The middle accuracy of testing set "+ self.reduced_name
            fig_y_name = "middle accuracy of testing set"
            fig_name = "kfold_mid_acc_" + str(k) + "_" + str(ratio) + self.reduced_name + ".png"
        elif mode == 3 and self.sat_way == 'maxSAT':
            fig_title = self.sat_way + " in " + self.test_way + " mode: The percent of satisfait examples"
            fig_y_name = "satisfait examples percentage"
            if self.test_way == 'rest':
                fig_name = "rest_sat_" + str(ratio) + "_" + str(seed) + self.reduced_name + ".png"
            elif self.test_way == 'percentage':
                fig_name = "per_sat_" + str(per) + "_" + str(ratio) + "_" + str(seed) + self.reduced_name + ".png"
            elif self.test_way == 'k_fold':
                fig_name = "kfold_avg_sat_" + str(k) + "_" + str(ratio) + "_" + str(seed) + self.reduced_name + ".png"
            else:
                raise ValueError("The test way chosen is not valid!!")
        else:
            raise ValueError("The mode chosen is not valid or not suitable with the kind of sat way or test way!!")
        return (fig_name, fig_title, fig_x_name, fig_y_name)
        


