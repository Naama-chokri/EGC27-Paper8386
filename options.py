#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## options.py
##
##  Created on: Sep 20, 2017
##      Author: Alexey S. Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

#
#==============================================================================
from __future__ import print_function
import getopt
import math
import os
from pysat.card import EncType
import sys
from decimal import Decimal

import time
import ast

import json

#
#==============================================================================
encmap = {
    "seqc": EncType.seqcounter,
    "cardn": EncType.cardnetwrk,
    "sortn": EncType.sortnetwrk,
    "tot": EncType.totalizer,
    "mtot": EncType.mtotalizer,
    "kmtot": EncType.kmtotalizer
}


#
#==============================================================================
class Options(object):
    """
        Class for representing command-line options.
    """

    def __init__(self, command):
        """
            Constructor.
        """

        self.approach = 'pbased'
        self.approx = 0
        self.enc = 'cardn'
        self.files = None
        self.filter = 0
        self.solver = 'm22'
        self.primer = 'lbx'
        self.indprime = False
        self.mapfile = None
        self.plimit = 0
        self.nooverlap = False
        self.cdump = False
        self.pdump = False
        self.rdump = False
        self.separator = ','
        self.trim = 0
        self.use_cld = False
        self.verb = 0
        self.weighted = False
        self.nbnodes = -1
        self.iti_result_file = None
        self.accmin = 0.95
        self.scilearn = None
        
        #---------------------------#
        # 2019-3-8 Hao HU: add the option of fold for testing
        self.k_fold = 0
        self.i_fold = 0
        self.n_repeat = 3

        # here the percentage for testing set is for the sub-dataset after ratio
        self.percentage = 1
        self.out_index = []

        # 2019-3-22 Hao HU: add the option of encoding for the transformer
        self.encoding = 'one-hot'

        # the ratio of choosing the sub dataset -> because the insufficient calculation ability of complete MaxSAT solver, we choose a part of 
        # original dataset as the new dataset.
        self.ratio = 1

        # 2019-7-2 Hao HU: add the option of setting testing set for the original dataset
        self.test_ratio = 0

        # 2019-7-8 Hao HU: add the potenial option of ratio for validation set
        # self.validation_ratio = 0

        # 2019-4-10 Hao HU: add the option of choosing the testing set
        # If the rest option is True, the testing set will be chosen as the left part of dataset after choosing the benchmark by the ratio
        # and option ''k_fold'' and ''percentage'' will have no use.
        self.rest = False

        # 2019-4-12 Hao HU: add the option of choosing the mode of constraints used
        # for the lambda and tau in Nina code, and new constraints
        # Use the binary ordering code as permission in linux: 0 -> 00 non nina, non new; 1 -> 01 non Nina, new, etc
        # 2019-6-25 Hao HU: useless now!!
        self.mode_constraint = 0

        # 2019-4-23 Hao HU: add the option of defining the seed used for sampling
        self.seed = 2019

        # 2019-5-8 Hao HU: add the option of using reduced encoding or not
        self.reduced = 0

        # 2019-12-2 Hao HU: add the option of controlling the maximum depth of decision tree
        # -1 in default, indicating no constraints for controlling the max_depth
        # other positive values indicating the constraints will be generated.
        self.max_depth = -1
        
        # 2019-12-18 Hao HU: add the option of exact depth of decision tree found by MaxSAT
        self.depth = -1

        # 2019-12-10 Hao HU: add for indicating using "atleast" constraint
        self.atleast = -1

        # 2019-5-20 Hao HU: add the option for class Data: which will decide the sampling approach of choosing training set
        self.sampling = 'stratified'

        # 2019-7-2 Hao HU: add a dict to indicate the possible plusiers Options objects
        # it is for the ensemble learning. in the format of {Option:num_learner}
        # self.options_dict = {}

        # 2019-7-8 Hao HU: add the dict of parameters of forest of dt found by MaxSAT for Bagging and Random Forest
        # the key of element in this dict is the size of decision tree, and the value is the number of base learners
        # now just set them, next find the way to get the dict from the command
        self.para_forest = {7:151}

        # 2019-10-17 Hao HU: add the option of using the complete MaxSAT solver or incomplete MaxSAT solver
        # 1 -> complete MaxSAT solver/ 0 -> incomplete MaxSAT solver
        self.maxsat_solver_complete = 1

        # 2019-10-17 Hao HU: option for chosen the MaxSAT solver, the candiates are as follows
        # complete: RC2
        # incomplete: loandra
        self.maxsat_solver = 'RC2'

        # 2019-10-17 Hao HU: add the option of global time limit when using the incomplete solver (only useful when solver_complete = 0)
        # 2019-12-26 hao HU: add one meaning for this option: for indicating the time limit for DL8.5
        self.maxsat_solver_gtimeout = 600

        # 2019-10-23 add the option of setting timeout for complete MaxSAT solver or not
        # 0 -> not set, others -> set
        self.complete_tlimit = 0

        # 2019-11-14 Hao HU: option for setting the number of iterations for Adaboost
        self.itr_boost = 50

        # 2021-6-2 Hao Hu: add the option of time limit for generating variables & constraints and writing them into WCNF file
        # In default, it is set as 30mins to let normal case pass.
        self.wcnf_timeout = 1800

        # 2020-5-13 Hao HU: option for setting the timeout for Loandra core-guided phase (in seconds)
        # the default value is 30 seconds, which is same as the Loandra.
        self.core_ptime = 30
        #---------------------------#

        if command:
            self.parse_command(command)

    # 2019-6-26: Hao HU add the function of parsing from json file -> waited to be finished
    def parse_json(self, json_path=None, json_content=None):
        """
            Parser for json format
        """
        if json_path is None and json_content is None:
            raise ValueError("The json file path and json content could not be all None")
        
        if not json_path is None and not json_content is None:
            raise ValueError("The json file path and the json content could not be all not None")

        self.json_path = json_path
        if not json_path is None:
            with open(json_path, 'r') as f:
                content = json.load(f)
        else:
            content = json_content

        approach = content['approach']

        if approach == 'dtencoding':
            pass
        elif approach == 'MaxSATdtencoding':
            pass
        elif approach == 'bagging':
            pass
        else:
            pass

    def parse_json_dtencoding(self, parameters):
        """
            Parser for the json file for the decision trees found by SAT
            several setting could be set:
                seperator + encoding + sampling + data_file + seed -> data_processing
                ratio +  k_fold + percentage + rest -> set the size of training set and testing set
                nbnodes + reduced -> define the start size of tree and the version of SAT encoding
        """
        pass
        
    def parse_json_content(self, parmeters):
        """ 
            Parser for json content
        """
        for opt, arg in parmeters.items():
            pass

    def parse_command(self, command):
        """
            Parser.
        """
        self.command = command
        try:
            opts, args = getopt.getopt(command[1:],
                                    'a:de:f:hil:m:op:s:t:vw',
                                    ['approach=',
                                        'approx=',
                                        'cdump',
                                        'do-cld',
                                        'enc=',
                                        'filter=',
                                        'help',
                                        'indprime'
                                        'map-file=',
                                        'no-overlap',
                                        'pdump',
                                        'rdump',
                                        'plimit=',
                                        'primer=',
                                        'sep=',
                                        'solver=',
                                        'trim=',
                                        'verbose',
                                        'weighted',
                                        'nbnodes=',
                                        'accmin=',
                                        'scilearn=',
                                        'iti=',
                                        'kfold=',
                                        'nrepeat=',
                                        'percentage=',
                                        'encoding=',
                                        'ratio=',
                                        'rest=',
                                        'mode_constraint=',
                                        'seed=',
                                        'reduced=',
                                        'sampling=',
                                        'test_ratio=',
                                        'para_forest=',
                                        'complete=',
                                        'maxsat_solver=',
                                        'gtimeout=',
                                        'complete_tlimit=',
                                        'itr_boost=',
                                        'max_depth=',
                                        'atleast=',
                                        'depth=',
                                        'ifold=',
                                        'core_ptime=',
                                        'wcnf_timeout='])
        except err:
            sys.stderr.write(str(err).capitalize())
            self.usage()
            sys.exit(1)
        
        for opt, arg in opts:
            if opt in ('-a', '--approach'):
                self.approach = str(arg)
            elif opt == '--approx':
                self.approx = int(arg)
            elif opt == '--cdump':
                self.cdump = True
            elif opt in ('-d', '--do-cld'):
                self.use_cld = True
            elif opt in ('-e', '--enc'):
                self.enc = str(arg)
            elif opt in ('-f', '--filter'):
                if arg == 'all':
                    arg = -1
                self.filter = int(arg)
            elif opt in ('-h', '--help'):
                self.usage()
                sys.exit(0)
            elif opt in ('-i', '--indprime'):
                self.indprime = True
                self.filter = -1  # filter all unnecessary primes
            elif opt in ('-l', '--plimit'):
                self.plimit = int(arg)
            elif opt in ('-m', '--map-file'):
                self.mapfile = str(arg)
            elif opt in ('-o', '--no-overlap'):
                self.nooverlap = True
            elif opt in ('-p', '--primer'):
                self.primer = str(arg)
            elif opt == '--pdump':
                self.pdump = True
            elif opt == '--rdump':
                self.rdump = True
            elif opt == '--sep':
                self.separator = str(arg)
            elif opt in ('-s', '--solver'):
                self.solver = str(arg)
            elif opt in ('-t', '--trim'):
                self.trim = int(arg)
            elif opt in ('-v', '--verbose'):
                self.verb += 1
            elif opt in ('-w', '--weighted'):
                self.weighted = True
            elif opt in ('--nbnodes'):
                print("Number of nodes start: " + arg)
                self.nbnodes = int(arg)
            elif opt in ('--accmin'):
                print(arg)
                self.accmin = float(arg)
            elif opt in ('--scilearn'):
                self.scilearn = int(arg)
            elif opt in ('--iti'):
                print(arg)
                self.iti_result_file = str(arg)
            elif opt in ('--kfold'):
                print("The " + arg + "-fold in cross-validation")
                self.k_fold = int(arg)
            elif opt in ('--nrepeat'):
                print("The " + arg + "repeats for cross-validation")
                self.n_repeat = int(arg)
            elif opt in ('--ifold'):
                self.i_fold = int(arg)
            elif opt in ('--per', '--percentage'):
                print("The percentage chosen for testing set: " + arg)
                self.percentage = Decimal(arg)
            elif opt in ('--data-encoding'):
                self.encoding = str(arg)
            elif opt in ('--ratio'):
                print("The ratio of choosing the sub dataset: " + arg)
                self.ratio = Decimal(arg)
            elif opt in ('--rest'):
                print("Taking the rest as testing set: " + arg)
                # the option will only be trippled when the value is 'True'
                self.rest = True if arg == 'True' else False
            elif opt in ('--mode_constraint'):
                print(arg)
                self.mode_constraint = int(arg)
            elif opt in ('--seed'):
                print("The random seed chosen: " + arg)
                self.seed = int(arg)
            elif opt in ('--reduced'):
                print("Using the reduced version or not: " + arg)
                self.reduced = int(arg)
            elif opt in ('--sampling'):
                print(arg)
                self.sampling = str(arg)
            elif opt in ('--test_ratio'):
                print(arg)
                self.test_ratio = Decimal(arg)
            elif opt in ('--para_forest'):
                print("The parameters for building the forest: " + arg)
                self.para_forest = ast.literal_eval(arg)
            elif opt in ('--complete'):
                print("Using complete MaxSAT solver or not: " + arg)
                self.maxsat_solver_complete = int(arg)
            elif opt in ('--maxsat_solver'):
                print("The MaxSAT solver used: " + arg)
                self.maxsat_solver = str(arg)
            elif opt in ('--gtimeout'):
                print("The global timeout used in incomplete MaxSAT solver: " + arg)
                self.maxsat_solver_gtimeout = int(arg)
            elif opt in ('--core_ptime'):
                print("The timeout used for core-guided phase in Loandra is: " + arg)
                self.core_ptime = int(arg)
            elif opt in ('--wcnf_timeout'):
                print("The timeout used in generating WCNF file: " + arg)
                self.wcnf_timeout = int(arg)
            elif opt in ('--complete_tlimit'):
                print("Set timeout for complete MaxSAT solver or not: " + arg)
                self.complete_tlimit = int(arg)
            elif opt in ('--itr_boost'):
                print("The number of base learner for Boosting is: " + arg)
                self.itr_boost = int(arg)
            elif opt in ('--max_depth'):
                print("The maximum depth for decision tree found is: " + arg)
                self.max_depth = int(arg)
            elif opt in ('--atleast'):
                if int(arg) > 0:
                    print("Using \"atleast\" constraint!")
                else:
                    pass
                self.atleast = int(arg)
            elif opt in ('--depth'):
                print("The exact depth for decision tree found is: " + arg)
                self.depth = int(arg)
            else:
                assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

        self.enc = encmap[self.enc]
        self.files = args

    def usage(self):
        """
            Print usage message.
        """

        print('Usage: ' + os.path.basename(self.command[0]) + ' [options] training-data')
        print('Options:')
        print('        -a, --approach=<string>    Approach to use')
        print('                                   Available values: mp92, pbased (default = pbased)')
        print('        --approx=<int>             Approximate set cover with at most k attempts')
        print('                                   Available values: [0 .. INT_MAX] (default = 0)')
        print('        --cdump                    Dump largest consistent subset of input samples')
        print('        -d, --do-cld               Do D clause calls')
        print('        -e, --enc=<string>         Encoding to use')
        print('                                   Available values: cardn, kmtot, mtot, sortn, tot (default = cardn)')
        print('        -f, --filter=<int>         Filter out unnecessary combinations of this size when computing primes')
        print('                                   Available values: [0 .. INT_MAX], all (default = 0)')
        print('        -h, --help')
        print('        -i, --indprime             Compute primes for each sample separately')
        print('        -l, --plimit=<int>         Compute at most this number of primes per sample')
        print('                                   Available values: [0 .. INT_MAX] (default = 0)')
        print('        -m, --map-file=<string>    Path to a file containing a mapping to original feature values. (default: none)')
        print('        -o, --no-overlap           Force no overlap on the training data')
        print('        -p, --primer=<string>      Prime implicant enumerator to use')
        print('                                   Available values: lbx, mcsls, sorted (default = lbx)')
        print('        --pdump                    Dump MaxSAT formula for enumerating primes')
        print('                                   (makes sense only when using an MCS-based primer)')
        print('        --rdump                    Dump resulting CSV table')
        print('        --sep=<string>             Field separator used in input file (default = \' \')')
        print('        -s, --solver=<string>      SAT solver to use')
        print('                                   Available values: g3, lgl, m22, mc, mgh (default = m22)')
        print('        -t, --trim=<int>           Trim unsatisfiable core at most this number of times')
        print('                                   Available values: [0 .. INT_MAX] (default = 0)')
        print('        -v, --verbose              Be more verbose')
        print('        -w, --weighted             Minimize the total number of literals')
        print('        --nbnodes                  Number of nodes in tree is fixed')
        print('        --kfold                    Indicate the k-fold validation')
        print('        --nrepeat                  Indicate the number of repeat')
        print('        --data-encoding            Indicate the type of encoding used in transformation of non-binary dataset')
        print('        --ratio                    The ratio of choosing a big dataset as a small benchmark')
        print('        --rest                     Indicating the way of choosing the testing set. Only will be used when the value is "True"')
        print('        --mode_constraint          The mode of using the constraint, indicating using the nina\'s constraint or new constraints we proposed')
        print('        --seed                     The seed used for sampling')
        print('        --reduced                  Integer indicating using the reduced encoding or not')
        print('        --sampling                 The sampling approach used in selecting the training set')
        print('        --test_ratio               The ratio of choosing testing set from the original dataset')