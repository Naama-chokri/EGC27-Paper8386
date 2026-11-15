#!/usr/bin/env python
# -*- coding:utf-8 -*-
# #
## datasetTransformer.py
## Created on: March 21, 2019
## Author; Hao HU
##
#------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------#
import os


class DatasetTransformer(object):

    def __init__(self, data_path, name_path=None, encoding='one-hot', sep=','):

        self.data_path = data_path
        self.name_path = name_path
        # 2019-3-21: Now do not considering different type of encoding
        # all consider as the 'one-hot' encoding
        self.encoding = encoding
        self.sep = sep

    def generate_info(self, f_value_list, f_bin_num):
        dataset_name = self.data_path.split('/')[-1]
        info_str = dataset_name + ':\n' + 'Original feature number: ' + str(len(f_value_list)) + '\n' \
            'Original feature values:\n'
        for f_name, f_value in f_value_list:
            info_str = info_str + f_name + ': ' + str(f_value) + '\n'
        info_str = info_str + 'Transformed binary number: ' + str(f_bin_num) + '\n'
        return info_str
        
    def transform(self):
        # judge is there the file of name
        if self.name_path:
            bin_f, f_value_list = self.judge_binary_with_name_file()
        else:
            bin_f, f_value_list = self.judge_binary_without_name_file()

        if bin_f:
            dataset_path, f_bin_num = self.write_feature_to_dataset(f_value_list)
        else:  # in this condition
            dataset_path, f_bin_num = self.convert_to_all_binary(f_value_list)
        
        # generating the string of information
        info_str = self.generate_info(f_value_list, f_bin_num)

        return dataset_path, info_str

    def judge_binary_with_name_file(self):
        """
            To simplify the format of .names
            Using '|' as the indicator of comment
            Using '.' as the end indicator of features
            Using ':' as the sepator of name of features and value of feature

            Return:
                1. bin_f -> indicate all features of data are binary or not
                2. feature_range_list -> [(feature_name, range)]
        """
        # read the name file
        bin_f = True
        feature_range_list = []
        label_name = 'label'
        with open(self.name_path, 'r+') as f:
            for line in f.readlines():
                line = line.strip()
                if not len(line) or line.startswith('|'):
                    continue
                # judge it is the label or not
                if not line.endswith('.'):
                    label_name = line.split(',')[0]
                else:
                    f_temp = line.split(':')
                    f_number_value = len(f_temp[1].split(','))
                    feature_range_list.append((f_temp[0], f_number_value))
                    if f_number_value > 2:
                        bin_f = False
            # add the label at last
        feature_range_list.append((label_name, 2))
        return bin_f, feature_range_list

    def judge_binary_without_name_file(self):
        """
            Because there is no supported name file, so just use f_i to name each feature
            And for the range of value for each feature, it will be got after iterating all data

            To simplify the format of generating the name of feature, so we use 'f' as beginning
        """
        feature_values_list = []
        bin_f = True
        label_name = 'label'
        with open(self.data_path, 'r+') as f:
            cnt = 0
            for line in f.readlines():
                # 2019-5-31 Hao HU: add the ignoring of '@' at first of line for dataset from CP4IM
                line = line.strip()
                if line.startswith('@') or len(line) == 0:
                    continue
                # last item is the label, do not consider
                features_value = line.split(self.sep)[:-1]

                # for i in range(len(features_value)):
                #     value = int(features_value[i].strip())
                #     if cnt == 0:
                #         feature_range_list.append(('f'+str(i), value+1))
                #     else:
                #         f_name, f_range_current = feature_range_list[i]
                #         if value + 1 > f_range_current:
                #             feature_range_list[i] = (f_name, value+1)
                #     if value + 1 > 2:
                #         bin_f = False
                # cnt = cnt + 1
                #------------------------------------------------------------------------------
                # 2019-5-31 Hao HU: improve the way of getting the value range of each feature
                # Previous version: if feature_1 = 100, creating all features from 0 to 100 
                #   |_> increasing the binary features transformed too much
                # Changed version: just count the different value showed in the dataset
                #   |_> advantage: decrease maximumly the number of binary features transformed
                #   |_> disadvantage: not good at new coming data
                for i in range(len(features_value)):
                    value = int(features_value[i].strip())
                    if cnt == 0:
                        # change the range into list of values
                        feature_values_list.append(('f'+str(i), [value]))
                    else:
                        f_name, f_value_list = feature_values_list[i]
                        if not value in f_value_list:
                            f_value_list.append(value)
                    
                    if len(feature_values_list[i][1]) > 2:
                        bin_f = False
                cnt = cnt + 1
        feature_values_list.append((label_name, [0,1]))

        return bin_f, feature_values_list

    def write_feature_to_dataset(self, f_value_list):
        """
            That is the function of add the first line in the file, when every feature is 
            changed into binary
        """ 
        # considering the formal file is in the format of txt
        new_data_set_path = self.data_path.replace('.txt', '-un.csv')
        
        # generate the line for feature name
        feature_name_info = ''
        # f_range_list has already give the order of binary feature
        for (f_name, f_value) in f_value_list:
            feature_name_info = feature_name_info + f_name + ","
        feature_name_info = feature_name_info + "\n"
        # write the feature info at the first line
        with open(new_data_set_path, 'w') as f:
             f.write(feature_name_info)
             with open(self.data_path, 'r+') as f_data:
                 for line in f_data.readlines():
                    # avoid the comment of '@'
                    if line.startswith('@') or len(line) == 0:
                        continue
                    f.write(line)
        return new_data_set_path, len(f_value_list)

    def convert_to_all_binary(self, f_value_list):
        """
            Function for converting the normalized dataset with feature non-binary to binary.
            The type of encoding depends on the type of encoding chosen in the self

        """
        print(self.data_path)
        new_data_set_path = self.data_path.replace('.txt', '-un.csv')
        if self.encoding == 'one-hot':
            # generating the line of new features in one-hot encoding based on its range
            f_bin_list = []
            for (f_name, f_value) in f_value_list[:-1]:
                for value in f_value:
                    f_bin_list.append(f_name + "_" + str(value))
            label_name, _ = f_value_list[-1]
            f_bin_list.append(label_name)
            feature_name_info = ",".join(f_bin_list) + "\n"

            with open(new_data_set_path, 'w') as f:
                # write feature information line
                f.write(feature_name_info)

                # convert every record in the data file
                with open(self.data_path, 'r+') as f_data:
                    for line in f_data.readlines():
                        line = line.strip()
                        # ignore the comment starting with '@'
                        if line.startswith('@') or len(line) == 0:
                            continue
                        # convert line
                        f_value_t = self.convert_line_to_binary(line, f_value_list)
                        # ignoring the label of each record
                        if not len(f_value_t) == len(f_bin_list):
                            raise ValueError("The dataset has a wrong record in " + line)
                        f.write(",".join(map(str, f_value_t)) + "\n")
        return new_data_set_path, len(f_bin_list)
        
    def convert_line_to_binary(self, line, f_value_list):
        f_values = [int(v.strip()) for v in line.split(self.sep)]
        # convert every feature value into the one-hot encoding
        f_value_t = []
        for i in range(len(f_value_list)-1):
            _, f_value_real = f_value_list[i]
            temp = [0] * len(f_value_real)
            temp[f_value_real.index(f_values[i])] = 1
            f_value_t = f_value_t + temp
        # add the label value at the end
        f_value_t.append(f_values[-1])
        return f_value_t
