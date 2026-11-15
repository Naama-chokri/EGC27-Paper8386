#!/usr/bin/env python
# -*- coding:utf-8 -*-
##
## data.py
##
##  Created on: Sep 20, 2017
##      Author: Alexey S. Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

#
# ==============================================================================
from __future__ import print_function
import collections
import itertools
import os

import time

from scripts.utils import Sampler


#
# ==============================================================================
class Data(object):
    """
        Class for representing data (transactions).
    """

    def __init__(self, filename=None, fpointer=None, mapfile=None, separator=' ', ratio=0.75, test_ratio=0, seed=2019,
                 sampling='stratified'):
        """
            Constructor and parser.
        """

        self.names = None
        self.nm2id = None
        self.samps = None
        self.labels = None
        self.wghts = None
        self.feats = None
        self.fvmap = None
        self.ovmap = {}
        self.fvars = None
        self.fname = filename
        self.mname = mapfile
        self.ratio = ratio

        # add the option of test_orig_ratio, which means it will choose the testing set from the original dataset at first
        self.test_ratio = test_ratio
        self.deleted = set([])

        # ----------------------------------------------------#
        # Hao HU 2019-4-17: Add an option of storing the rest
        # samps after sampling.
        self.sampes_rest = None
        # Hao HU 2019-4-23: Add an option of seed for deciding the seed
        # in the sampler
        self.seed = seed
        # Hao HU 2019-5-20: Add an option of sampling for indicating the sampling approach of choosing the training set
        self.sampling = sampling
        # ----------------------------------------------------#

        assert self.test_ratio >= 0 and self.test_ratio < 1
        assert self.test_ratio + self.ratio <= 1

        if filename:
            print("The dataset of " + filename)
            with open(filename, 'r') as fp:
                self.parse(fp, separator, self.sampling)
        elif fpointer:
            self.parse(fpointer, separator, self.sampling)

        if self.mname:
            self.read_orig_values()

    def parse(self, fp, separator, sampling):
        """
            Parse input file.
            In the parse process, there is only one split.
            if test_ratio == 0: then samps -> part chosen by ratio
            else: samps -> part chosen by (1-test_ratio)
            Attention !! the test_ratio here is designed for Ensemble Learning, which means the original dataset should left one part as testing
                set for all base learners. In fact, the ratio here will be transformed into ratio/(1-test_ratio)
        """

        # reading data set from file
        lines = fp.readlines()

        # reading preamble
        self.names = lines[0].strip().split(separator)
        self.feats = [set([]) for n in self.names]
        del (lines[0])

        # now, no use for test_ratio
        # if self.test_ratio == 0:
        #     # the improved version: using the class of sampler
        #     # more improved version: add the different kinds of sampling approach
        #     # 2019-7-2 Hao HU: change the percentage of sampling by considering the ratio of choosing testing set at first
        #     sampler = Sampler(approach=sampling, percentage=self.ratio, seed=self.seed)
        # else:
        #     sampler = Sampler(approach=sampling, percentage=1-self.test_ratio, seed=self.seed)

        # considering the last character is '\n'
        self.labels = [int(line[-2]) for line in lines]
        print("The original size is " + str(len(self.labels)))
        if self.ratio < 1:
            sampler = Sampler(approach=sampling, percentage=self.ratio, seed=self.seed)
            line_numbers_chosen = sampler.do_sampling_by_per(range(len(lines)), self.labels)
        else:
            line_numbers_chosen = range(len(lines))

        # self.line_numbers_chosen = line_numbers_chosen
        lines_chosen, lines_rest = [], []
        for i in range(len(lines)):
            if i in line_numbers_chosen:
                lines_chosen.append(lines[i])
            else:
                lines_rest.append(lines[i])
        lines = lines_chosen

        # filling name to id mapping
        self.nm2id = {name: i for i, name in enumerate(self.names)}

        self.nonbin2bin = {}
        for name in self.nm2id:
            spl = name.rsplit(':', 1)
            if spl[0] not in self.nonbin2bin:
                self.nonbin2bin[spl[0]] = [name]
            else:
                self.nonbin2bin[spl[0]].append(name)
        # reading training samples
        self.samps, self.wghts, self.sampes_rest = [], [], []

        for line in lines:
            sample = [int(fv) for fv in line.strip().split(separator)]
            self.samps.append(sample)

        for line in lines_rest:
            sample = [int(fv) for fv in line.strip().split(separator)]
            self.sampes_rest.append(sample)

        print("The samples chosen as samples_train:" + str(len(self.samps)))
        print("The samples chosen as samples_test:" + str(len(self.sampes_rest)))

    def binary_data(self):
        outs = {}
        nb_classes = 0
        # for samples
        for i in range(len(self.samps)):
            samp_bin, out = self.samps[i][:-1], self.samps[i][-1]
            if not out in outs:
                outs[out] = nb_classes
                nb_classes = nb_classes + 1
            for l in samp_bin:
                if l > 0:
                    name, lit = self.fvmap.opp[l]
                    j = self.nm2id[name]

                    if len(self.feats[j]) > 2:
                        samp_bin += [-self.fvmap.dir[(name, l)] for l in list(self.feats[j].difference(set([lit])))]

            self.samps[i] = samp_bin + [outs[out]]

        # for samples left
        for i in range(len(self.sampes_rest)):
            samp_bin, out = self.sampes_rest[i][:-1], self.sampes_rest[i][-1]
            if not out in outs:
                outs[out] = nb_classes
                nb_classes = nb_classes + 1
            for l in samp_bin:
                if l > 0:
                    name, lit = self.fvmap.opp[l]
                    j = self.nm2id[name]

                    if len(self.feats[j]) > 2:
                        samp_bin += [-self.fvmap.dir[(name, l)] for l in list(self.feats[j].difference(set([lit])))]

            self.sampes_rest[i] = samp_bin + [outs[out]]

        if (len(outs) > 2):
            print("Non-binary classes", outs)
            exit()
        else:
            print("Ok, binaryclasses", outs)

    def read_orig_values(self):
        """
            Read original values for all the features.
            (from a separate CSV file)
        """

        self.ovmap = {}

        for line in open(self.mname, 'r'):
            featval, bits = line.strip().split(',')
            feat, val = featval.split(':')

            for i, b in enumerate(bits):
                f = '{0}:b{1}'.format(feat, i + 1)
                v = self.fvmap.dir[(f, '1')]

                if v not in self.ovmap:
                    self.ovmap[v] = [feat]

                if -v not in self.ovmap:
                    self.ovmap[-v] = [feat]

                self.ovmap[v if b == '1' else -v].append(val)

    def dump_result(self, primes, covers):
        """
            Save result to a CSV file.
        """

        fname = '{0}-result.csv'.format(os.path.splitext(self.fname)[0])

        for f in self.feats:
            if len(f) > 2:
                print('c2 non-binary features detected; not dumping the result')
                return

        with open(fname, 'w') as fp:
            print(','.join(self.names), file=fp)

            for cid in covers:
                for pid in covers[cid]:
                    feats = ['' for n in xrange(len(self.names))]

                    for l in primes[cid][pid - 1]:
                        name, val = self.fvmap.opp[l]
                        feats[self.nm2id[name]] = val

                    # label
                    name, val = self.fvmap.opp[cid]
                    feats[self.nm2id[name]] = val

                    print(','.join(feats), file=fp)
