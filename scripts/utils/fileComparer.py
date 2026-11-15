## class of comparing differences between two files. It will get the differences in symbols 
## Date: 2019-3-19
## Author: Hao HU
#-----------------------------------------------------------------------#

#-----------------------------------------------------------------------#
class FileComparer(object):
    
    def __init__(self, file_path_first):
        """
            Only one file as the base of comparision
        """
        self.file_path_f = file_path_first

        # assume that the name of file is in the last part of the file path
        self.file_name_f = self.file_path_f.split('/')[-1]

    def is_equal(self, file_path_second):
        """
            The function of comparing two files are equal to each other or not.
            It will directly stop when it finds a different line

            Return -> the bool of indicating two files are equal line by line or not
        """
        with open(self.file_path_f, 'r+') as f_f:
            with open(file_path_second, 'r+') as f_s:
                line_f = f_f.readline().strip()
                line_s = f_s.readline().strip()
                while line_f and line_s and line_f == line_s:
                    line_f = f_f.readline().strip()
                    line_s = f_s.readline().strip()
                if not line_f and not line_s:
                    return True
                else:
                    return False

    def find_all_diff_between_files(self, file_path_second):
        """
            find all differences between two files.

            Return -> diff_array: [('c',value_in_file1, value_in_file2), ...]
                option: (in opinion of file_1 to file_2)
                    c -> change
                    m -> more
                    l -> less
        """
        diff_array = []
        with open(self.file_path_f, 'r+') as f_f:
            with open(file_path_second, 'r+') as f_s:
                line_f = f_f.readline().strip()
                line_s = f_s.readline().strip()

                while line_f and line_s:
                    # compare the diffence
                    if not line_f == line_s:
                        diff_array.append(('c', line_f, line_s))
                    line_f = f_f.readline().strip()
                    line_s = f_s.readline().strip()
                
                # considering the different size
                if not line_f and line_s:
                    while line_s:
                        diff_array.append(('l', line_s))
                        line_s = f_s.readline().strip()
                elif line_f and not line_s:
                    while line_f:
                        diff_array.append('m', line_f)
                        line_f = f_f.readline().strip()
                else:
                    pass
        
        return diff_array

    def find_all_diff_between_cnf_solution(self, file_path_second, var_id_dict):
        """
            Find all differences between two cnf solution files
        """
        diff_array = self.find_all_diff_between_files(file_path_second)
        diff_sol = []
        for (status, line_f, line_s) in diff_array:
            # processing the line_f and line_s as a solution 
            neg_f = True if line_f[0] == '-' else False
            neg_s = True if line_s[0] == '-' else False
            id_f = int(line_f[1:]) if neg_f else int(line_f)
            id_s = int(line_s[1:]) if neg_s else int(line_s)

            # find the symbols from the dictionary of id-variable_name
            # if it does not contain the value of selected key, it will directly raise KeyError
            diff_sol.append((status, var_id_dict[id_f], var_id_dict[id_s]))
        
        return diff_sol

    def find_diff_between_solution_file_and_model(self, model, var_id_dict):
        """
            This function aims to realize finding all differences between model and solution file.
            Here, the solution model compares to the file of solution.
            Becase of there are so many different solution generated for constructing the same tree, so there is
            no need for seeing all differences in detail. Just knowing which variables are been changed is a good idea
        """
        var_changed = []
        print(model)
        with open(self.file_path_f, 'r+') as f:
            lines = f.readlines()
            index = 0
            for line in lines:
                value = int(line.strip())
                if not value == model[index]:
                    var_changed.append(var_id_dict[index])
                index = index + 1
        return var_changed