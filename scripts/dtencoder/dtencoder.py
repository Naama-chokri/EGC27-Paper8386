#!/usr/bin/env python
# -*- coding:utf-8 -*-
# #
# # primer.py
# #
# #  Created on: Dec 4, 2017
# #      Author: Alexey S. Ignatiev
# #      E-mail: aignatiev@ciencias.ulisboa.pt
# #

#
# ==============================================================================
from __future__ import print_function
import inspect, os, sys, signal
import subprocess
from os import pread

sys.path.insert(0, os.path.join(
    os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])), '../pysat-module/'))
sys.path.insert(0, os.path.join(
    os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])), '../hitman/py'))

from scripts.utils import *
import collections
import itertools
from pysat.formula import WCNF
from pysat.solvers import Solver
import resource
import socket
from pysat.card import *
from pysat.formula import WCNF
import math
import time

import numpy as np

from pysat.solvers import Solver  # standard way to import the library
from pysat.solvers import Minisat22, Glucose3  # more direct way
from pysat.formula import WCNF  # for getting the wcnf format
from pysat.examples.rc2 import RC2  # for using the RC2 solver


#
# import matplotlib.pyplot as plt
# import networkx as nx
# from networkx.drawing.nx_pydot import write_dot
def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                  pos=None, parent=None):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node of current branch

       width: horizontal space allocated for this branch - avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
       pos: a dict saying where all nodes go if they have been assigned
       parent: parent of this branch.'''
    if pos == None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    neighbors = G.neighbors(root)
    nbs = []
    for neighbor in neighbors:
        nbs.append(neighbor)

    if parent != None:  # this should be removed for directed graphs.
        if (parent in nbs):
            nbs.remove(parent)  # if directed, then parent not in neighbors.
    if len(nbs) != 0:
        dx = width / len(nbs)

        nextx = xcenter - width / 2 - dx / 2
        for neighbor in nbs:
            nextx += dx
            pos = hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos,
                                parent=root)

    return pos


# ==============================================================================
class DTEncoder(object):

    def __init__(self, data, options):
        """
            Constructor.
        """

        self.init_stime = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        self.init_ctime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime

        # saving data
        self.data = data
        # # binarizing the data properly
        # outs = {}
        # nb_classes = 0
        # for i in range(len(self.data.samps)):
        #     samp_bin, out = self.data.samps[i][:-1], self.data.samps[i][-1]
        #     if (out not in outs):
        #         outs[out] = nb_classes
        #         nb_classes += 1

        #     for l in samp_bin:
        #         if l > 0:  # negative literal means that the feature is binary
        #             name, lit = self.data.fvmap.opp[l]
        #             j = self.data.nm2id[name]

        #             if len(self.data.feats[j]) > 2:
        #                 samp_bin += [-self.data.fvmap.dir[(name, l)] for l in list(self.data.feats[j].difference(set([lit])))]

        #     self.data.samps[i] = samp_bin + [outs[out]]        

        # for i in range(len(self.data.sampes_rest)):
        #     samp_bin, out = self.data.sampes_rest[i][:-1], self.data.sampes_rest[i][-1]
        #     if (out not in outs):
        #         outs[out] = nb_classes
        #         nb_classes += 1

        #     for l in samp_bin:
        #         if l > 0:  # negative literal means that the feature is binary
        #             name, lit = self.data.fvmap.opp[l]
        #             j = self.data.nm2id[name]

        #             if len(self.data.feats[j]) > 2:
        #                 samp_bin += [-self.data.fvmap.dir[(name, l)] for l in list(self.data.feats[j].difference(set([lit])))]

        #     self.data.sampes_rest[i] = samp_bin + [outs[out]]

        # print(self.data.samps[:5])
        self.max_id = 1
        # self.max_id = max(self.data.fvmap.dir.values()) + 1

        # saving options
        self.options = options
        self.var2ids = {}

        self.s = Solver(name='g3')
        # self.s.file = open(self.options.files[0] + ".dump", "w+")

        # -----------------------------------#
        # 2019-3-6: Hao HU - add the wcnf var for getting the formula in wcnf format
        self.wcnf = WCNF()
        # 2019-5-7: Hao HU - add the counter of times used the constraints of equal one
        self.c_equal_one = 0
        # -----------------------------------#

    def format_indexes(self, inds):
        return "_" + "".join(["[{}]".format(x) for x in inds])

    def create_indexed_variable_name(self, name, inds):
        x_id = ("{}".format(name))
        x_id = x_id + self.format_indexes(inds)
        x_id = x_id.replace("-", "n_")
        return x_id

    def get_varid(self, var):
        if var not in self.var2ids:
            self.var2ids[var] = self.max_id
            self.max_id += 1
        return self.var2ids[var]

    def lookup_varid(self, var):
        if var not in self.var2ids:
            print("The " + var + " is not existed in the dict of variables!!")
            # print("Requested a variable  {} that does not exist".format(var))
            exit()
        return var, self.var2ids[var]

    def lookup_var_name_by_id(self, id):
        for var, id_e in self.var2ids.items():
            if id == id_e:
                return var
        return "Nothing"

    def soft_lookup_varid(self, var, replace=True):
        if var not in self.var2ids:
            # ###print("Requested a variable  {} that does not exist".format(var), end = " Replace with ")

            if (replace == True):
                # ###print(" True")
                return self.lookup_varid(self.create_indexed_variable_name(LABEL_TRUE_VARS, []))
            if (replace == False):
                # ###print(" False")
                return self.lookup_varid(self.create_indexed_variable_name(LABEL_FALSE_VARS, []))
            exit()
        return var, self.var2ids[var]

    def get_l_bounds(self, i, N=None):
        return range(i + 1, min(2 * i, N - 1) + 1)

    def get_r_bounds(self, i, N=None):
        return range(i + 2, min(2 * i + 1, N) + 1)

    def get_p_bounds(self, j, N=None):
        return range(int(max(math.floor(j / 2), 1)), (min((j - 1), N) + 1))

    def get_lambda_bounds(self, i):
        return range(int(math.floor(i / 2)) + 1)

    def get_tau_bounds(self, i):
        return range(i + 1)

    # 2019-12-2: add the function of getting the bound of height of node i
    # return the [|_log2(i+1)_|, |-(i-1)/2-|]
    def get_h_bounds(self, i, N=None):
        return range(int(math.ceil(np.log2(i + 1))) - 1, int(math.ceil((i - 1) * 1.0 / 2)) + 1)

    # *************************************************************#
    # ------------------------------------------------------#
    # Hao HU 2019-5-2: add the variable for general x_ij
    # x_ij <-> l_ij, j \in LR(i)
    # In the same time, based on the constraint of l_ij <-> r_ij+1 and p_ij <-> l_ij
    # we can replace all variables of r and p by the x
    # ------------------------------------------------------#
    def generate_x_variables(self, N):
        for i in range(1, N + 1):
            for j in self.get_l_bounds(i=i, N=N):
                if j % 2 == 0:
                    var_name = self.create_indexed_variable_name(LABEL_X_VARS, [i, j])
                    self.get_varid(var_name)

    def get_l_variablle_by_x(self, i, j):
        l_var_name, l_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_X_VARS, [i, j]))
        return l_var_name, l_var_id

    def get_r_variable_by_x(self, i, j):
        # because for getting the r_ij, j \in RR(i), so it equals to the l_ij-1
        r_var_name, r_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_X_VARS, [i, j - 1]))
        return r_var_name, r_var_id

    def get_p_variable_by_x_l(self, j, i, N):
        if not (j % 2 == 0 and j in self.get_l_bounds(i=i, N=N)):
            print("The i and j given is completely wrong!!")
            exit()
        p_var_name, p_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_X_VARS, [i, j]))
        return p_var_name, p_var_id

    def get_p_variable_by_x_r(self, j, i, N):
        if not (j % 2 == 1 and j in self.get_r_bounds(i=i, N=N)):
            print("The i and j given is completely wrong!!")
            exit()
        p_var_name, p_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_X_VARS, [i, j - 1]))
        return p_var_name, p_var_id

    # def get_p_variable_by_x(self, j, i, N):
    #     if j % 2 == 0 and j in self.get_l_bounds(i=i, N=N):
    #         # which means the p_ji will be got from by the l_ij
    #         p_var_name, p_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_X_VARS, [i, j]))
    #         return p_var_name, p_var_id
    #     elif j % 2 == 1 and j in self.get_r_bounds(i=i, N=N):
    #         p_var_name, p_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_X_VARS, [i, j-1]))
    #         return p_var_name, p_var_id
    #     else:
    #         print("The " + str(i) + "th node will not be the parent of " + str(j) + "th node!!")
    #         exit()    
    # *************************************************************#

    def generete_l_variables(self, N):
        # l ij to indicate that node i has node j as the *left* child
        # i+1 <= j <= min(2*i, N-1)
        for i in range(1, N + 1):
            for j in self.get_l_bounds(i=i, N=N):
                if (j % 2 == 0):
                    var_name = self.create_indexed_variable_name(LABEL_L_VARS, [i, j])
                    self.get_varid(var_name)

    def generete_r_variables(self, N):
        # r ij to indicate that node i has node j as the *right* child
        # i+2 <= j <= min(2*i+1, N)
        for i in range(1, N + 1):
            for j in self.get_r_bounds(i=i, N=N):
                if (j % 2 == 1):
                    var_name = self.create_indexed_variable_name(LABEL_R_VARS, [i, j])
                    self.get_varid(var_name)

    def generete_p_variables(self, N):
        # p ji = 1 iff the parent of node j is i.
        for j in range(2, N + 1):
            for i in self.get_p_bounds(j=j, N=N):
                var_name = self.create_indexed_variable_name(LABEL_P_VARS, [j, i])
                self.get_varid(var_name)

    def generete_v_variables(self, N):
        # A propositional variable v i = 1 iff i is a leaf node
        for i in range(1, N + 1):
            var_name = self.create_indexed_variable_name(LABEL_V_VARS, [i])
            self.get_varid(var_name)

    def generete_lambda_variables(self, N):
        for i in range(1, N + 1):
            for t in self.get_lambda_bounds(i=i):
                var_name = self.create_indexed_variable_name(LABEL_LAMBDA_VARS, [t, i])
                self.get_varid(var_name)

    def generete_tau_variables(self, N):
        for i in range(1, N + 1):
            for t in self.get_tau_bounds(i=i):
                var_name = self.create_indexed_variable_name(LABEL_TAU_VARS, [t, i])
                self.get_varid(var_name)

    def generete_constant_variables(self):
        var_name = self.create_indexed_variable_name(LABEL_TRUE_VARS, [])
        self.get_varid(var_name)

        var_name = self.create_indexed_variable_name(LABEL_FALSE_VARS, [])
        self.get_varid(var_name)

    def generete_a_variables(self, N, K):
        # this variable is assigned value 1 iff feature f r is assigned to node j, with 1 ≤ r ≤ K.
        for r in range(1, K + 1):
            for j in range(1, N + 1):
                var_name = self.create_indexed_variable_name(LABEL_A_VARS, [r, j])
                self.get_varid(var_name)

    def generete_u_variables(self, N, K):
        # u rj : this variable is assigned value 1 iff feature f r is being discriminated against by node j.
        for r in range(1, K + 1):
            for j in range(1, N + 1):
                var_name = self.create_indexed_variable_name(LABEL_U_VARS, [r, j])
                self.get_varid(var_name)

    def generete_d0_variables(self, N, K):
        # this variable is assigned value 1 iff feature f r is discriminated for value 0 by node j, or by one of its
        # ancestors.Concretely, any example exhibiting f r = 0 will be discriminated by node j iff d 0 rj = 1.
        for r in range(1, K + 1):
            for j in range(1, N + 1):
                var_name = self.create_indexed_variable_name(LABEL_D_ZERO_VARS, [r, j])
                self.get_varid(var_name)

    def generete_d1_variables(self, N, K):
        # this variable is assigned value 1 iff feature f r is discriminated for value 1 by node j, or by one of its
        # ancestors.Concretely, any example exhibiting f r = 1 will be discriminated by node j iff d 1 rj = 1.
        for r in range(1, K + 1):
            for j in range(1, N + 1):
                var_name = self.create_indexed_variable_name(LABEL_D_ONE_VARS, [r, j])
                self.get_varid(var_name)

    def generete_c_variables(self, N):
        # c j : class of leaf node i, which can either be 0 or 1.
        for i in range(1, N + 1):
            var_name = self.create_indexed_variable_name(LABEL_C_VARS, [i])
            self.get_varid(var_name)

    # ----------------------------------------#
    def generate_h_variables(self, N):
        """
            2019-12-2: Hao HU
            generate the variable set h_{it}
            indicating all possible heights for node i
        """
        for i in range(1, N + 1):
            for t in self.get_h_bounds(i=i, N=N):
                var_name = self.create_indexed_variable_name(LABEL_H_VARS, [i, t])
                # print(var_name)
                self.get_varid(var_name)

    def generate_m_variables(self, N):
        """
            2019-12-6: Hao HU
            generate variables indicating at least how many nodes are used for constructing the decision tree
            m_i indicating at least i nodes are used
        """
        # assert N is odd
        assert N % 2 == 1
        # here, create only variables for odd nodes
        for i in range(1, N + 1):
            if i % 2 == 0:
                continue
            var_name = self.create_indexed_variable_name(LABEL_M_VARS, [i])
            self.get_varid(var_name)

    # ----------------------------------------#

    # ----------------------------------------#
    def generate_b_variables(self, M):
        # 2019-3-5: Hao HU
        # create the variables of blocking literals for each example
        # M is the size of examples used
        # b q : blocking literal for example q, where idea is b q is true will lead example q is discriminated correctly
        for i in range(1, M + 1):
            var_name = self.create_indexed_variable_name(LABEL_B_VARS, [i])
            self.get_varid(var_name)

    # ---------------------------------------#

    # -----------------------------------------#
    # Hao HU: 2019-4-9 Add the function of only generating the variables 
    # concerned with constraints of binary tree
    def generate_binary_tree_variables_without_l_t(self, N):
        self.generete_v_variables(N=N)
        self.generete_l_variables(N=N)
        self.generete_r_variables(N=N)
        self.generete_p_variables(N=N)
        # self.generete_lambda_variables(N=N)
        # self.generete_tau_variables(N=N)

    # -----------------------------------------#
    def generate_variables(self, N, K, max_depth=-1, depth=-1, atleast=-1, reduced=-1):

        """
            2019-12-9: add the option "at least" indicating using m_i indicating use at least i nodes
            Four ways:
                1) n = N -> max_depth = -1 && atleast = -1(original) 
                2) n = N, h <= max_depth -> max_depth > 0 && atleast = -1
                3) n <= N -> max_depth = -1 && atleast > 0
                4) n <= N, h <= max_depth -> max_depth > 0 && atleast > 0
                4*) h <= max_depth -> n <= N_max = 2^(H+1)-1 -> max_depth > 0 && atleast > 0
                    => judge before using this function, actually in mode 3
            Attention! when use the at-least constraint, the N must be odd
        """
        self.generete_constant_variables()
        self.generete_v_variables(N=N)
        if reduced <= 0:
            self.generete_l_variables(N=N)
            self.generete_r_variables(N=N)
            self.generete_p_variables(N=N)
        else:
            self.generate_x_variables(N=N)

        if atleast > 0:
            self.generate_m_variables(N=N)
        if max_depth > 0 or depth > 0:
            self.generate_h_variables(N=N)
        self.generete_lambda_variables(N=N)
        self.generete_tau_variables(N=N)

        self.generete_a_variables(N=N, K=K)
        self.generete_u_variables(N=N, K=K)
        self.generete_d0_variables(N=N, K=K)
        self.generete_d1_variables(N=N, K=K)
        self.generete_c_variables(N=N)

        self.generate_b_variables(M=len(self.data.samps))

    # -------------------------------------------#
    # ! Part of constraints !!
    def generate_constants_fixed(self):
        print("Constraint: constant fixed")
        v_var_name, v_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_TRUE_VARS, []))
        self.wcnf.append([v_id])
        self.s.add_clause([v_id])
        # ###print("True is fixed: add constraint -{}({})".format(v_var_name, [v_id]))

        v_var_name, v_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_FALSE_VARS, []))
        self.wcnf.append([-v_id])
        self.s.add_clause([-v_id])

    def generate_root_is_fixed(self, N):
        v_var_name, v_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [1]))
        self.wcnf.append([-v_id])
        self.s.add_clause([-v_id])
        print("Constraint: root is fixed!")
        # print("-" + v_var_name)

    def generate_leaf_has_no_children(self, N, reduced=-1, atleast=-1):
        # v_i\limply\neg l_{ij} &\quad\quad & i+1\le j\le\min(2i,N-1) \land\even(j)
        # atleast -> indicating using the atleast constraint or not => m_j+1 -> (v_i -> -l_i_j)
        print("Constraint: leaf has no chidren")
        for i in range(1, N + 1):
            for j in self.get_l_bounds(i, N):
                clause = []
                v_var_name, v_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [i]))
                clause.append(-v_id)

                # only choose the even ones - perfer j%2 == 0:
                if (j % 2 == 1):
                    continue

                if atleast > 0:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))
                    clause.append(-m_var_id)
                if reduced > 0:
                    l_var_name, l_var_id = self.get_l_variablle_by_x(i, j)
                else:
                    l_var_name, l_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_L_VARS, [i, j]))
                clause.append(-l_var_id)
                self.wcnf.append(clause)
                self.s.add_clause(clause)
                # if atleast > 0:
                #     print("-" + v_var_name + ", -" + m_var_name + ", -" + l_var_name)

    def generate_leaf_and_right(self, N, atleast=-1):
        # l_{ij}\leftrightarrow r_{ij+1} & \quad\quad & \tn{with~} i+1\le j\le\min(2i,N-1)\land\even(j)\\
        print("Constraint: leaf and right!")
        for i in range(1, N + 1):
            for j in self.get_l_bounds(i, N):
                if (j % 2 == 0):
                    clauses_f = []

                    l_var_name, l_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_L_VARS, [i, j]))
                    r_var_name, r_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_R_VARS, [i, j + 1]))
                    if atleast > 0:
                        m_var_name, m_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))
                        clauses_f.append([-m_var_id, -l_var_id, r_var_id])
                        clauses_f.append([-m_var_id, l_var_id, -r_var_id])
                    else:
                        clauses_f.append([-l_var_id, r_var_id])
                        clauses_f.append([l_var_id, -r_var_id])
                    # add two clauses in same time
                    self.wcnf.extend(clauses_f)
                    self.s.append_formula(clauses_f)
                    # if atleast > 0:
                    #     print("-" + m_var_name + ", -" + l_var_name + ", " + r_var_name)
                    #     print("-" + m_var_name + ", " + l_var_name + ", -" + r_var_name)
                    # self.s.add_clause([-l_var_id, r_var_id])
                    # self.s.add_clause([l_var_id, -r_var_id])
                    # ###print("Left and right children together: add constraint {}({}) <->  {}({})".format(l_var_name, l_var_id, r_var_name, r_var_id))

    def generate_reified_card(self, reif_var_name, l_names, atleast_var_name=None):
        # generating the constraint of sum of reif_var_name is 1
        # -v -> l1+..ln=1
        # 2019-12-9: add the option of atleast_var_name -> if not None, so m -> (-v -> l1+..+ln = 1)
        if not atleast_var_name == None:
            _, m_id = self.lookup_varid(atleast_var_name)

        _, v_id = self.lookup_varid(reif_var_name)

        clause = []
        for l_var in l_names:
            _, l_id = self.lookup_varid(l_var)
            clause.append(l_id)
        # use the constraints for equals one
        y_literal_s, clauses_equal_one = self.generate_equal_one_constraint(lits=clause)

        # then change the -v -> (c_1 /\ c_2 /\ ... /\ c_n) => ((v \/ c_1) /\ (v \/ c_2) ...)
        # add v for every clause generated from the equal_one constraint
        # 2019-12-9: same idea when using the atleast constraint => (-m \/ v \/ c_1) /\ etc
        if len(clauses_equal_one) == 0:
            if atleast_var_name == None:
                self.wcnf.append([v_id])
                self.s.add_clause([v_id])
            else:
                self.wcnf.append([-m_id, v_id])
                self.s.add_clause([-m_id, v_id])
                # print("-" + atleast_var_name + ", " + reif_var_name)
        else:
            for c in clauses_equal_one:
                c.append(v_id)
                if not atleast_var_name == None:
                    c.append(-m_id)
                self.wcnf.append(c)
                self.s.add_clause(c)

        # if v_i, it should directly give all y_literal as 0
        # v_i -> (not y_0 /\ not y_1 /\ etc)
        # 2019-12-9: change with the atleast => (-m_j \/ v_i) -> (-y_0 /\ -y_1 etc)  => (m_j /\ -v_i) \/ (-y_o /\ -y_1 etc)
        if not len(y_literal_s) == 0:
            for y_lit_id in y_literal_s:
                if atleast_var_name == None:
                    self.wcnf.append([-v_id, -y_lit_id])
                    self.s.add_clause([-v_id, -y_lit_id])
                # Attetion!! 2019-12-11
                else:
                    temp_m_neg_v_var_name, temp_m_neg_v_var_id = self.reified_and(atleast_var_name, reif_var_name,
                                                                                  a_pos=1, b_pos=-1)
                    self.wcnf.append([temp_m_neg_v_var_id, -y_lit_id])
                    self.s.add_clause([temp_m_neg_v_var_id, -y_lit_id])

    def generate_nonleaf_at_least_one_child(self, N, reduced=-1, atleast=-1):
        #   \neg v_i\rightarrow\left(\sum\limits_{\substack{j=i+1\\\even(j)}}^{\min(2i,N-1)}l_{ij}\geq 1\right)
        # v_i \/ li(i+1)\/ .. \/ li(min(2i,N-1)
        # 2019-12-9: add option of atleast -> m_i -> (-v_i -> (\sum_{LR(i)}^{l_ij /\ m_j+1} = 1))]
        print("Constraint: nonleaf node must have at least one child")
        for i in range(1, N + 1):
            v_var_name, v_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [i]))

            if atleast > 0:
                # get m_i
                if i % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i + 1]))

            l_names = []
            for j in self.get_l_bounds(i, N):
                if (j % 2 == 1):
                    continue
                if reduced > 0:
                    l_var_name, l_var_id = self.get_l_variablle_by_x(i, j)
                else:
                    l_var_name, l_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_L_VARS, [i, j]))

                if atleast > 0:
                    m_j_var_name, m_j_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))
                    temp_l_m_j_var_name, temp_l_m_j_var_id = self.reified_and(l_var_name, m_j_var_name)
                    l_names.append(temp_l_m_j_var_name)
                else:
                    l_names.append(l_var_name)
            # ###print("Leaf has not children: card : add constraint -{} -> {} = 1".format(v_var_name, l_names))
            if atleast > 0:
                self.generate_reified_card(v_var_name, l_names, m_var_name)
            else:
                self.generate_reified_card(v_var_name, l_names)

    def generate_parent_has_child(self, N, atleast=-1):
        print("Constraint: relationship between parent and child")
        for j in range(2, N + 1):
            for i in self.get_p_bounds(j=j, N=N):
                clauses_f = []
                p_var_name, p_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_P_VARS, [j, i]))
                if (j % 2 == 0):
                    if (j in self.get_l_bounds(i, N)):
                        l_var_name, l_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_L_VARS, [i, j]))
                        # ###print("A parent has a child: add constraint {} <-> {}".format(p_var_name, l_var_name))
                        if atleast > 0:
                            m_var_name, m_var_id = self.lookup_varid(
                                self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))
                            clauses_f.append([-m_var_id, -p_var_id, l_var_id])
                            clauses_f.append([-m_var_id, p_var_id, -l_var_id])
                        else:
                            clauses_f.append([-p_var_id, l_var_id])
                            clauses_f.append([p_var_id, -l_var_id])
                        self.wcnf.extend(clauses_f)
                        self.s.append_formula(clauses_f)
                        # self.s.add_clause([-p_var_id, l_var_id])
                        # self.s.add_clause([p_var_id, -l_var_id])
                else:
                    if (j in self.get_r_bounds(i, N)):
                        r_var_name, r_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_R_VARS, [i, j]))
                        # ###print("A parent has a child: add constraint {} <-> {}".format(p_var_name, r_var_name))
                        if atleast > 0:
                            m_var_name, m_var_id = self.lookup_varid(
                                self.create_indexed_variable_name(LABEL_M_VARS, [j]))
                            clauses_f.append([-m_var_id, -p_var_id, r_var_id])
                            clauses_f.append([-m_var_id, p_var_id, -r_var_id])
                        else:
                            clauses_f.append([-p_var_id, r_var_id])
                            clauses_f.append([p_var_id, -r_var_id])
                        self.wcnf.extend(clauses_f)
                        self.s.append_formula(clauses_f)
                        # self.s.add_clause([-p_var_id, r_var_id])
                        # self.s.add_clause([p_var_id, -r_var_id])

    def generate_node_has_one_parent(self, N, atleast=-1):
        print("Constraint: every node must have one parent, original version")
        for j in range(2, N + 1):
            p_var_ids = []
            p_var_names = []
            for i in self.get_p_bounds(j=j, N=N):
                p_var_name, p_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_P_VARS, [j, i]))
                p_var_ids.append(p_var_id)
                p_var_names.append(p_var_name)

            # if j == N:
            #     print(p_var_ids)
            if atleast > 0:
                if j % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))

            if (len(p_var_ids) > 1):
                #####print(p_var_names)
                # one_out = CardEnc().equals(lits=p_var_ids, top_id=self.max_id, encoding=self.options.enc)
                # self.max_id = one_out.nv
                y_ids, one_out = self.generate_equal_one_constraint(p_var_ids)
                # copying cardinality constraint
                for cl in one_out:
                    if atleast > 0:
                        cl.append(-m_var_id)
                    self.wcnf.append(cl)
                    self.s.add_clause(cl)
            else:
                # if we have only one variable we just set this var = 1
                if atleast > 0:
                    self.wcnf.append([-m_var_id, p_var_id])
                    self.s.add_clause([-m_var_id, p_var_id])
                else:
                    self.wcnf.append([p_var_id])
                    self.s.add_clause([p_var_id])
                #####print(p_var_ids)
                # ###print("A child has one parent: add constraint {} = 1".format(p_var_names))

    def generate_node_has_one_parent_reduced(self, N, atleast=-1):
        # Hao HU 2019-5-2: change the way of setting the constraints for a node has a parent
        print("Constraint: every node must have one parent, reduced version")
        p_var_ids_s = []
        for j in range(2, N + 1):
            flag_add = False
            p_var_ids = []
            p_var_names = []
            for i in self.get_p_bounds(j=j, N=N):
                # filter the number of node in the parent
                if j in self.get_l_bounds(i=i, N=N) and j % 2 == 0:
                    p_var_name, p_var_id = self.get_p_variable_by_x_l(j, i, N)
                    p_var_ids.append(p_var_id)
                    p_var_names.append(p_var_name)
                elif j in self.get_r_bounds(i=i, N=N) and j % 2 == 1:
                    p_var_name, p_var_id = self.get_p_variable_by_x_r(j, i, N)
                    p_var_ids.append(p_var_id)
                    p_var_names.append(p_var_name)
                else:
                    pass

            if len(p_var_ids_s) == 0:
                p_var_ids_s.append(p_var_ids)
                flag_add = True
            else:
                flag_break = False
                for item_ids_p in p_var_ids_s:
                    if item_ids_p == p_var_ids:
                        flag_break = True
                        break
                flag_add = not flag_break
            if flag_add:
                y_ids, one_out = self.generate_equal_one_constraint(lits=p_var_ids)
                if atleast > 0:
                    if j % 2 == 1:
                        m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j]))
                    else:
                        m_var_name, m_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))
                for cl in one_out:
                    if atleast > 0:
                        cl.append(-m_var_id)
                    self.wcnf.append(cl)
                    self.s.add_clause(cl)

    # -----------------------------------------------------------------------------------#
    # 2019-12-2: add the constraints for controlling maximum depth in decision tree
    def generate_root_in_layer_zero(self, N):
        """
            h_{1, 0} = 1 => the root must be in the layer 0
        """
        print("Constraint: depth| root is in depth zero")
        v_var_name, v_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_H_VARS, [1, 0]))
        self.wcnf.append([v_id])
        self.s.add_clause([v_id])

    def generate_node_be_in_exact_layer(self, N, atleast=-1):
        """
            the constraint for every node must be in one exact layer
        """
        print("Constraint: depth| every node must be one exact depth")
        for i in range(1, N + 1):
            h_var_ids = []
            h_var_names = []
            for t in self.get_h_bounds(i=i, N=N):
                h_var_name, h_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_H_VARS, [i, t]))
                h_var_ids.append(h_var_id)
                h_var_names.append(h_var_name)

            if atleast > 0:
                if i % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i + 1]))

            if len(h_var_ids) > 1:
                y_ids, one_out = self.generate_equal_one_constraint(lits=h_var_ids)
                for cl in one_out:
                    if atleast > 0:
                        cl.append(-m_var_id)
                    self.wcnf.append(cl)
                    self.s.add_clause(cl)
            else:
                # if we only have one variable we just set this var as 1
                if atleast > 0:
                    self.wcnf.append([-m_var_id, h_var_ids[-1]])
                    self.s.add_clause([-m_var_id, h_var_ids[-1]])
                else:
                    self.wcnf.append([h_var_ids[-1]])
                    self.s.add_clause([h_var_ids[-1]])

    def generate_relationship_increase_depth(self, N, reduced=-1, atleast=-1):
        """
            the constraint of increasing the depth by the relationship of left child and right child
            reduced -> indicating using reduced encoding or not
            2019-12-9: add option atleast
        """
        print("Constraint: depth| relationship of increasing the depth")
        for i in range(1, N + 1):
            for t in self.get_h_bounds(i=i, N=N):
                clauses_f = []
                h_var_name, h_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_H_VARS, [i, t]))
                # considering increasing the depth for the left child
                for j in self.get_l_bounds(i=i, N=N):
                    if j % 2 == 1:
                        continue
                    if reduced > 0:
                        l_var_name, l_var_id = self.get_l_variablle_by_x(i, j)
                    else:
                        l_var_name, l_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_L_VARS, [i, j]))
                    h_plus_var_name, h_plus_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_H_VARS, [j, t + 1]))

                    # add the condition of using atleast constraint
                    if atleast > 0:
                        m_var_name, m_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))
                        clauses_f.append([-m_var_id, -h_var_id, -l_var_id, h_plus_var_id])
                    else:
                        clauses_f.append([-h_var_id, -l_var_id, h_plus_var_id])
                # considering increasing the depth for the right child
                for j in self.get_r_bounds(i=i, N=N):
                    if j % 2 == 0:
                        continue
                    if reduced > 0:
                        r_var_name, r_var_id = self.get_r_variable_by_x(i, j)
                    else:
                        r_var_name, r_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_R_VARS, [i, j]))
                    h_plus_var_name, h_plus_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_H_VARS, [j, t + 1]))

                    if atleast > 0:
                        m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j]))
                        clauses_f.append([-m_var_id, -h_var_id, -r_var_id, h_plus_var_id])
                    else:
                        clauses_f.append([-h_var_id, -r_var_id, h_plus_var_id])
                # extend the clauses_f into the wcnf
                self.wcnf.extend(clauses_f)
                self.s.append_formula(clauses_f)

    def generate_contrlloing_maximum_depth_constraint(self, N, H, atleast=-1):
        """
            the constraint for controlling the maximum depth is H by making sure all nodes in layer H are leaf node
            H -> the max_depth chosen => need be checked before setting the constraint for tree in size N
        """
        print("Constraint: max_depth| controlling the maximum depth")
        # check the H is aviable for size N or not, making sure the H is greater than the minimum possible depth
        # if H is greater than the maximum possible depth, then H is set as the maximum possible depth
        h_possible = self.get_h_bounds(i=N, N=N)
        min_h_possible = h_possible[0]
        max_h_possible = h_possible[-1]
        print("Info: For decision tree in size " + str(N) + ", the valid depth are between " + str(
            min_h_possible) + " and " + str(max_h_possible))
        assert H >= min_h_possible

        if H > max_h_possible:
            H = max_h_possible
            print("Info: " + str(H) + " is greater than the maximum possible depth, so set as the last value")

        # set the constraint -> i \in [2H, min(2^(H+1) - 1, N)]
        node_bound_in_layer_h = range(2 * H, min(int(np.exp2(H + 1) - 1), N) + 1)
        # print(node_bound_in_layer_h)
        # time.sleep(100)
        for i in node_bound_in_layer_h:
            h_var_name, h_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_H_VARS, [i, H]))
            v_var_name, v_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [i]))
            if atleast > 0:
                if i % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i + 1]))
                # add constraint m_{i} --> h_{iH} --> v_{i} 
                self.wcnf.append([-m_var_id, -h_var_id, v_var_id])
                self.s.add_clause([-m_var_id, -h_var_id, v_var_id])
            else:
                # add constraint h_{iH} --> v_{i}
                self.wcnf.append([-h_var_id, v_var_id])
                self.s.add_clause([-h_var_id, v_var_id])

    def generate_controlling_depth_constraint(self, N, H, atleast=-1):
        """
            the constraint for controlling the exact depth for decision tree found
            add one constraint for the last one constraint
            \/h_iH = 1 => making sure that in layer H there is node
            If atleast is chosen, we can set the lower bound
        """
        print("Constraint: exact_depth| controlling the exact depth")
        h_possible = self.get_h_bounds(i=N, N=N)
        min_h_possible = h_possible[0]
        max_h_possible = h_possible[-1]
        print("Info: For decision tree in size " + str(N) + ", the valid depth are between " + str(
            min_h_possible) + " and " + str(max_h_possible))
        assert H >= min_h_possible

        if H > max_h_possible:
            H = max_h_possible
            print("Info: " + str(H) + " is greater than the maximum possible depth, so set as the last value")

        # set the constraint
        node_bound_in_layer_h = range(2 * H, min(int(np.exp2(H + 1) - 1), N) + 1)
        clause_layer_H = []
        for i in node_bound_in_layer_h:
            h_var_name, h_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_H_VARS, [i, H]))
            v_var_name, v_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [i]))
            if atleast > 0:
                if i % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i + 1]))
                # add constraint m_{i} --> h_{iH} --> v_{i} 
                self.wcnf.append([-m_var_id, -h_var_id, v_var_id])
                self.s.add_clause([-m_var_id, -h_var_id, v_var_id])
                temp_m_h_var_name, temp_m_h_var_id = self.reified_and(m_var_name, h_var_name)
                clause_layer_H.append(temp_m_h_var_id)
            else:
                # add constraint h_{iH} --> v_{i}
                self.wcnf.append([-h_var_id, v_var_id])
                self.s.add_clause([-h_var_id, v_var_id])
                clause_layer_H.append(h_var_id)

        self.wcnf.append(clause_layer_H)
        self.s.add_clause(clause_layer_H)
        if atleast > 0:
            # set the lower bound of number of nodes used => n >= 2*H+1
            m_lower_var_name, m_lower_var_id = self.lookup_varid(
                self.create_indexed_variable_name(LABEL_M_VARS, [2 * H + 1]))
            print(m_lower_var_name + " is True!!")
            self.wcnf.append([m_lower_var_id])
            self.s.add_clause([m_lower_var_id])

    # -----------------------------------------------------------------------------------#
    # 2019-12-9: add the constraints for relationship of using "atleast" i nodes
    def generate_atleast_three_node_used(self, N):
        """
            constraint m_3 = True
        """
        m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [3]))
        self.wcnf.append([m_var_id])
        self.s.add_clause([m_var_id])
        print("Constraint: atleast| at least 3 nodes are used!")

        # test !!
        # m_9_var_name, m_9_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [11]))
        # self.wcnf.append([m_9_var_id])
        # self.s.add_clause([m_9_var_id])

    def generate_atleast_increasing_relationship(self, N):
        """
            constraint m_{i+2} -> m_{i}, indicating using i+2 nodes, then, i nodes must be used
        """
        for i in range(1, N - 2 + 1):
            if i % 2 == 0:
                continue
            # making sure the i is odd
            m_plus_var_name, m_plus_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i + 2]))
            m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i]))
            self.wcnf.append([-m_plus_var_id, m_var_id])
            self.s.add_clause([-m_plus_var_id, m_var_id])
        print("Constraint: atleast| at least increasing relationship!")

    # -----------------------------------------------------------------------------------#
    # Hao HU 2019-4-19: Add the function of generating the constraints of equaling 1
    def generate_equal_one_constraint(self, lits=[]):
        """
            Function of generating the constraint for l1 \/ l2 \/ ... \/ lN = 1
            lits -> the array of literals indicating li
            do not add clauses directly, return the clauses, also the y_literals generated for enforcing the constraint
        """
        new_clauses = []
        y_lit_i_id_s = []
        if len(lits) == 0:
            pass
        elif len(lits) == 1:
            new_clauses.append([lits[0]])
        elif len(lits) == 2:
            new_clauses.append([lits[0], lits[1]])
            new_clauses.append([-lits[0], -lits[1]])
        else:
            # create the variables for counting the value of current variable
            # The name of new variable will be made as the concatetation of current variable and _y
            for i in range(len(lits)):
                lit_name = self.lookup_var_name_by_id(lits[i])
                y_lit_i_name = self.create_indexed_variable_name(
                    str(self.c_equal_one) + '_' + lit_name + '_' + LABEL_Y_VARS + '_' + str(i), [])
                y_lit_i_id = self.get_varid(y_lit_i_name)
                y_lit_i_id_s.append(y_lit_i_id)
            # constraint 1: y_lit_0 <-> lit_0
            self.c_equal_one = self.c_equal_one + 1
            new_clauses.append([-lits[0], y_lit_i_id_s[0]])
            new_clauses.append([lits[0], -y_lit_i_id_s[0]])

            # constraint 2: y_lit_N = 1(last y_literal must be True)
            new_clauses.append([y_lit_i_id_s[-1]])

            # constraint 3: literal transfer -> the counter feature
            # y_literal_i-1 -> y_literal_i
            # constraint 4: connection between counter literal and literal(attention!! It should be <-> relationship)
            # (not y_literal_i-1 /\ y_literal_i) <-> l_i
            for i in range(1, len(y_lit_i_id_s)):
                # constraint 3
                new_clauses.append([-y_lit_i_id_s[i - 1], y_lit_i_id_s[i]])

                # constraint 4
                new_clauses.append([y_lit_i_id_s[i - 1], -y_lit_i_id_s[i], lits[i]])
                new_clauses.append([-y_lit_i_id_s[i - 1], -lits[i]])
                new_clauses.append([-lits[i], y_lit_i_id_s[i]])

        return y_lit_i_id_s, new_clauses

    # -----------------------------------------------------------------------------------#

    def generate_ordering_enforce(self, N):
        for i in range(1, N + 1):
            for k in range(i + 1, N + 1):
                v_i_var_name, v_i_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [i]))
                v_k_var_name, v_k_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [k]))
                for j in self.get_l_bounds(i, N):
                    if (j % 2 == 1):
                        continue
                    if (j <= k):
                        continue
                    l_i_j_var_name, l_i_j_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_L_VARS, [i, j]))
                    for a in range(j - k - 1 + 1):
                        l_k_j_m_a_var_name, l_k_j_m_a_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_L_VARS, [k, j - a]))
                        # ###print("Ordering  is enforced: add constraint {} \/ {}  \/ {} \/ {}".format(v_i_var_name, v_k_var_name, l_i_j_var_name, "-" + l_k_j_m_a_var_name))

    # waited to be confirmed for "atleast" constraint
    def generate_refine_bounds_lambda_enforce(self, N, reduced=-1, atleast=-1):
        # 2019-12-12: add the "atleast" constraint
        print("Constraint: inferecnce constraint for leaf nodes are used")
        for i in range(1, N + 1):
            lambda_var_name, lambda_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_LAMBDA_VARS, [0, i]))
            if atleast > 0:
                if i % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i + 1]))
                self.wcnf.append([-m_var_id, lambda_id])
                self.s.add_clause([-m_var_id, lambda_id])
            else:
                self.wcnf.append([lambda_id])
                self.s.add_clause([lambda_id])
            # ###print("Refine bounds lambda: add constraint {} = 1".format(lambda_var_name))

        # part of adding the clauses for defining the lambda t, i
        for i in range(1, N + 1):
            if atleast > 0:
                if i % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i + 1]))
            for t in self.get_lambda_bounds(i=i):
                if (t == 0):
                    continue
                assert (i > 0)
                lambda_t_i_var_name, lambda_t_i_id = self.lookup_varid(
                    self.create_indexed_variable_name(LABEL_LAMBDA_VARS, [t, i]))
                lambda_t_m_1_i_m_1_var_name, lambda_t_m_1_i_m_1_id = self.soft_lookup_varid(
                    self.create_indexed_variable_name(LABEL_LAMBDA_VARS, [t - 1, i - 1]))
                lambda_t_i_m_1_var_name, lambda_t_i_m_1_id = self.soft_lookup_varid(
                    self.create_indexed_variable_name(LABEL_LAMBDA_VARS, [t, i - 1]), replace=False)
                v_var_name, v_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [i]))

                temp_t_m_1_i_m_1_v_var_name, temp_t_m_1_i_m_1_v_var_id = self.reified_and(lambda_t_m_1_i_m_1_var_name,
                                                                                          v_var_name)
                # la[t,i] <-> la[t,i-1] \/ temp
                or_names = [lambda_t_i_m_1_var_name, temp_t_m_1_i_m_1_v_var_name]
                if atleast > 0:
                    self.equivalence_lhs_literal_rhs_or(lambda_t_i_var_name, or_names, atleast_var_name=m_var_name)
                else:
                    self.equivalence_lhs_literal_rhs_or(lambda_t_i_var_name, or_names)
                # ###print("Refine bounds lambda: add constraint {} <-> {}".format(lambda_t_i_var_name, or_names))

        # part of clauses for refining the upper bound
        for i in range(1, N + 1):
            if atleast > 0:
                if i % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i + 1]))
            # t = 0 case
            for t in self.get_lambda_bounds(i=i):
                if (t == 0):
                    continue
                lambda_t_i_var_name, lambda_t_i_id = self.lookup_varid(
                    self.create_indexed_variable_name(LABEL_LAMBDA_VARS, [t, i]))
                l_index = 2 * (i - t + 1)
                if (l_index in self.get_l_bounds(i=i, N=N)):
                    if not reduced <= 0:
                        l_i_2it1_var_name, l_i_2it1_id = self.get_p_variable_by_x_l(l_index, i, N)
                    else:
                        l_i_2it1_var_name, l_i_2it1_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_L_VARS, [i, l_index]))
                    if atleast > 0:
                        self.wcnf.append([-m_var_id, -lambda_t_i_id, -l_i_2it1_id])
                        self.s.add_clause([-m_var_id, -lambda_t_i_id, -l_i_2it1_id])
                    else:
                        self.wcnf.append([-lambda_t_i_id, -l_i_2it1_id])
                        self.s.add_clause([-lambda_t_i_id, -l_i_2it1_id])
                    # ###print("Refine bounds lambda:  {} -> -{}".format(lambda_t_i_var_name, l_i_2it1_var_name))
                r_index = 2 * (i - t + 1) + 1
                if (r_index in self.get_r_bounds(i=i, N=N)):
                    if not reduced <= 0:
                        pass
                    else:
                        r_i_2it2_var_name, r_i_2it2_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_R_VARS, [i, r_index]))
                        if atleast > 0:
                            self.wcnf.append([-m_var_id, -lambda_t_i_id, -r_i_2it2_id])
                            self.s.add_clause([-m_var_id, -lambda_t_i_id, -r_i_2it2_id])
                        else:
                            self.wcnf.append([-lambda_t_i_id, -r_i_2it2_id])
                            self.s.add_clause([-lambda_t_i_id, -r_i_2it2_id])
                    # ###print("Refine bounds lambda:  {} -> -{}".format(lambda_t_i_var_name, r_i_2it2_var_name))

    # waited to be confirmed for "atleast" constraint
    def generate_refine_bounds_tau_enforce(self, N, reduced=-1, atleast=-1):
        print("Constraint: inferecnce constraint for non-leaf nodes are used")
        for i in range(1, N + 1):
            tau_var_name, tau_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_TAU_VARS, [0, i]))
            if atleast > 0:
                if i % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i + 1]))
                self.wcnf.append([-m_var_id, tau_id])
                self.s.add_clause([-m_var_id, tau_id])
            else:
                self.wcnf.append([tau_id])
                self.s.add_clause([tau_id])
            # ###print("Refine bounds  tau: add constraint {} = 1".format(tau_var_name))

        for i in range(1, N + 1):
            if atleast > 0:
                if i % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i + 1]))
            for t in self.get_tau_bounds(i=i):
                if (t == 0):
                    continue
                assert (i > 0)
                tau_t_i_var_name, tau_t_i_id = self.lookup_varid(
                    self.create_indexed_variable_name(LABEL_TAU_VARS, [t, i]))
                tau_t_m_1_i_m_1_var_name, tau_t_m_1_i_m_1_id = self.soft_lookup_varid(
                    self.create_indexed_variable_name(LABEL_TAU_VARS, [t - 1, i - 1]))
                tau_t_i_m_1_var_name, tau_t_i_m_1_id = self.soft_lookup_varid(
                    self.create_indexed_variable_name(LABEL_TAU_VARS, [t, i - 1]), replace=False)
                v_var_name, v_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [i]))

                temp_t_m_1_i_m_1_v_var_name, temp_t_m_1_i_m_1_v_var_id = self.reified_and(
                    a_var_name=tau_t_m_1_i_m_1_var_name, b_var_name=v_var_name, a_pos=1, b_pos=-1)
                # la[t,i] <-> la[t,i-1] \/ temp
                or_names = [tau_t_i_m_1_var_name, temp_t_m_1_i_m_1_v_var_name]
                if atleast > 0:
                    self.equivalence_lhs_literal_rhs_or(tau_t_i_var_name, or_names, atleast_var_name=m_var_name)
                else:
                    self.equivalence_lhs_literal_rhs_or(tau_t_i_var_name, or_names)
                # ###print("Refine bounds tau: add constraint {} <-> {}".format(tau_t_i_var_name, or_names))

        for i in range(1, N + 1):
            if atleast > 0:
                if i % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i + 1]))
            # t = 0 case
            for t in range(int(math.ceil(i / 2)), i + 1):
                if (t == 0):
                    continue
                tau_t_i_var_name, tau_t_i_id = self.lookup_varid(
                    self.create_indexed_variable_name(LABEL_TAU_VARS, [t, i]))
                l_index = 2 * (t - 1)
                if (l_index in self.get_l_bounds(i=i, N=N)):
                    if not reduced <= 0:
                        l_i_2it1_var_name, l_i_2it1_id = self.get_p_variable_by_x_l(l_index, i, N)
                    else:
                        l_i_2it1_var_name, l_i_2it1_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_L_VARS, [i, l_index]))
                    if atleast > 0:
                        self.wcnf.append([-tau_t_i_id, -l_i_2it1_id, -m_var_id])
                        self.s.add_clause([-tau_t_i_id, -l_i_2it1_id, -m_var_id])
                    else:
                        self.wcnf.append([-tau_t_i_id, -l_i_2it1_id])
                        self.s.add_clause([-tau_t_i_id, -l_i_2it1_id])
                    # ###print("Refine bounds tau:  {} -> -{}".format(tau_t_i_var_name, l_i_2it1_var_name))
                r_index = 2 * t - 1
                if (r_index in self.get_r_bounds(i=i, N=N)):
                    if not reduced <= 0:
                        pass
                    else:
                        r_i_2it2_var_name, r_i_2it2_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_R_VARS, [i, r_index]))
                        if atleast > 0:
                            self.wcnf.append([-tau_t_i_id, -r_i_2it2_id, -m_var_id])
                            self.s.add_clause([-tau_t_i_id, -r_i_2it2_id, -m_var_id])
                        else:
                            self.wcnf.append([-tau_t_i_id, -r_i_2it2_id])
                            self.s.add_clause([-tau_t_i_id, -r_i_2it2_id])
                    # ###print("Refine bounds tau:  {} -> -{}".format(tau_t_i_var_name, r_i_2it2_var_name))

    def generate_bin_tree_constraints(self, N, max_depth=-1, depth=-1, reduced=-1, atleast=-1):
        """
            Hao HU: 2019-5-2: Function of generating new variables for constructing the binary tree in reduced version
            2019-12-5: add the option of H indicating controlling the maximum depth or not. H = -1, not control.
        """
        # bin tree encoding
        # self.generate_constants_fixed()
        self.generate_root_is_fixed(N=N)
        if atleast > 0:
            self.generate_atleast_three_node_used(N=N)
            self.generate_atleast_increasing_relationship(N=N)
        self.generate_leaf_has_no_children(N=N, reduced=reduced, atleast=atleast)
        if reduced <= 0:
            self.generate_leaf_and_right(N=N, atleast=atleast)
            self.generate_parent_has_child(N=N, atleast=atleast)
            self.generate_node_has_one_parent(N=N, atleast=atleast)
        else:
            self.generate_node_has_one_parent_reduced(N=N, atleast=atleast)

        self.generate_nonleaf_at_least_one_child(N=N, atleast=atleast, reduced=reduced)

        # add the constraints of controlling the maximum depth if it needs
        if max_depth > 0 or depth > 0:
            assert not (max_depth > 0 and depth > 0)
            self.generate_root_in_layer_zero(N=N)
            self.generate_node_be_in_exact_layer(N=N, atleast=atleast)
            self.generate_relationship_increase_depth(N=N, atleast=atleast, reduced=reduced)
            if max_depth > 0:
                self.generate_contrlloing_maximum_depth_constraint(N=N, H=max_depth, atleast=atleast)
            if depth > 0:
                self.generate_controlling_depth_constraint(N=N, H=depth, atleast=atleast)
        self.generate_refine_bounds_lambda_enforce(N=N, atleast=atleast, reduced=reduced)
        self.generate_refine_bounds_tau_enforce(N=N, atleast=atleast, reduced=reduced)

    # --------------------------------------------------------#
    # Hao HU: 2019-4-12
    def generate_bin_tree_constraints_without_la_and_tau(self, N):
        self.generate_constants_fixed()
        self.generate_root_is_fixed(N=N)
        self.generate_leaf_has_no_children(N=N)
        self.generate_leaf_and_right(N=N)
        self.generate_nonleaf_at_least_one_child(N=N)
        self.generate_parent_has_child(N=N)
        self.generate_node_has_one_parent(N=N)

    # --------------------------------------------------------#

    # --------------------------------------------------------#

    def reified_and(self, a_var_name, b_var_name, a_pos=1, b_pos=1):
        # add the option of "clause_hard" for identifying the clause is hard or not
        # create temp var
        temp_var_name = self.create_indexed_variable_name(
            LABEL_T_VARS + "_and_" + str(a_pos) + "_" + a_var_name + "_" + str(b_pos) + "_" + b_var_name, [])
        temp_id = self.get_varid(temp_var_name)
        _, a_id = self.lookup_varid(a_var_name)
        _, b_id = self.lookup_varid(b_var_name)

        # add a AND b <-> c
        clauses_f = []
        clauses_f.append([-a_id * a_pos, -b_id * b_pos, temp_id])
        clauses_f.append([-temp_id, a_id * a_pos])
        clauses_f.append([-temp_id, b_id * b_pos])
        self.wcnf.extend(clauses_f)
        self.s.append_formula(clauses_f)
        # self.s.add_clause([-a_id * a_pos, -b_id * b_pos, temp_id])
        # self.s.add_clause([-temp_id, a_id * a_pos])
        # self.s.add_clause([-temp_id, b_id * b_pos])
        return temp_var_name, temp_id

    def equivalence_lhs_literal_rhs_or(self, a_var_name, b_var_names, atleast_var_name=None):
        # a <-> b or .. or b_n
        # 2019-12-9: add the option for using constraint of atleast
        _, a_id = self.lookup_varid(a_var_name)
        rhs_or = []
        for b_var_name in b_var_names:
            _, b_id = self.lookup_varid(b_var_name)
            rhs_or.append(b_id)

        if not atleast_var_name == None:
            _, m_var_id = self.lookup_varid(atleast_var_name)

        # -a or b or .. or b_n
        # change into -m \/ -a \/ b etc
        clause_f = rhs_or[:]
        clause_f.append(-a_id)
        if not atleast_var_name == None:
            clause_f.append(-m_var_id)
        self.wcnf.append(clause_f)
        self.s.add_clause(clause_f)

        for key, lit in enumerate(rhs_or):
            # a  \/ -b_i
            # change into (-m \/ a \/ -b_i)
            if not atleast_var_name == None:
                self.wcnf.append([-m_var_id, a_id, -lit])
                self.s.add_clause([-m_var_id, a_id, -lit])
            else:
                self.wcnf.append([a_id, -lit])
                self.s.add_clause([a_id, -lit])

    # ---------------------------------------------------------------------#
    # constraints for discriminating examples correctly

    def generate_discriminate_feature_0(self, N, K, reduced=-1, atleast=-1):
        # 2019-5-8 Hao HU: add the option of reduced for encoding reduced.
        # Discriminating a feature for value 0 at node $j$, with
        # $1< j\le N$:
        # \begin{equation} \label{eq:d0}
        # d^0_{rj}\leftrightarrow
        # \left(\bigvee_{i=\lfloor\frac{j}{2}\rfloor}^{j-1}
        # \left((p_{ji}\land d^0_{ri}) \lor (a_{ri}\land r_{ij})\right)
        # \right)
        # \end{equation}
        # And $d^{0}_{r,1}=0$.
        print("Constraint: discriminating feature 0")
        for r in range(1, K + 1):
            # d^{0}_{r,1}=0
            d0_var_name, d0_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_D_ZERO_VARS, [r, 1]))
            self.wcnf.append([-d0_var_id])
            self.s.add_clause([-d0_var_id])
            # ###print("Discriminate feature 0: add constraint -{}".format(d0_var_name))
            for j in range(2, N + 1):
                rhs_or_names = []

                if atleast > 0:
                    if j % 2 == 1:
                        m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j]))
                    else:
                        m_var_name, m_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))

                for i in self.get_p_bounds(j=j, N=N):
                    # (p_{ji}\land d^0_{ri})
                    if reduced <= 0:
                        p_var_name, p_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_P_VARS, [j, i]))
                    # if it is reduced and p_ji is valid & useful, which means it can be found by x_ij, it will get, or it will directly pass this i
                    elif j in self.get_l_bounds(i=i, N=N) and j % 2 == 0:
                        p_var_name, p_var_id = self.get_p_variable_by_x_l(j, i, N)
                    elif j in self.get_r_bounds(i=i, N=N) and j % 2 == 1:
                        p_var_name, p_var_id = self.get_p_variable_by_x_r(j, i, N)
                    else:
                        # directly pass this i for j selected because p_ji is not useful here
                        continue

                    d0_i_var_name, d0_i_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_D_ZERO_VARS, [r, i]))
                    temp_p_d_var_name, temp_p_d_var_id = self.reified_and(p_var_name, d0_i_var_name)
                    rhs_or_names.append(temp_p_d_var_name)

                    # ###print("Discriminate feature 0: add constraint {}<-> {} AND {} ".format(temp_p_d_var_name, p_var_name, d0_i_var_name))

                    # (a_{ri}\land r_{ij})
                    if (j % 2 == 1 and j in self.get_r_bounds(i=i, N=N)):
                        a_var_name, a_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_A_VARS, [r, i]))
                        if reduced <= 0:
                            r_var_name, r_var_id = self.lookup_varid(
                                self.create_indexed_variable_name(LABEL_R_VARS, [i, j]))
                        else:
                            # get the id in reduced type
                            r_var_name, r_var_id = self.get_r_variable_by_x(i, j)
                        temp_a_r_var_name, temp_a_r_var_id = self.reified_and(a_var_name, r_var_name)
                        rhs_or_names.append(temp_a_r_var_name)
                        # ###print("Discriminate feature 0: add constraint {}<-> {} AND {} ".format(temp_a_r_var_name, a_var_name, r_var_name))

                # d0_{r,j} --> \/ of tem vars
                d0_r_var_name, d0_r_var_id = self.lookup_varid(
                    self.create_indexed_variable_name(LABEL_D_ZERO_VARS, [r, j]))
                if atleast > 0:
                    self.equivalence_lhs_literal_rhs_or(d0_r_var_name, rhs_or_names, atleast_var_name=m_var_name)
                else:
                    self.equivalence_lhs_literal_rhs_or(d0_r_var_name, rhs_or_names)
                # ###print("Discriminate feature 0: add constraint {}<-> {}  ".format(d0_r_var_name, rhs_or_names))

    def generate_discriminate_feature_1(self, N, K, reduced=-1, atleast=-1):
        #  2019-5-8 Hao HU: Add the option of reduced for encoding reduced
        #  Discriminating a feature for value 1 at node $j$, with
        #     $1< j\le N$:
        #     \begin{equation} \label{eq:d1}
        #       d^1_{rj}\leftrightarrow
        #       \left(\bigvee_{i=\lfloor\frac{j}{2}\rfloor}^{j-1}
        #       \left((p_{ji}\land d^1_{ri}) \lor (a_{ri}\land l_{ij})\right)
        #       \right)
        #     \end{equation}
        #     And $d^{1}_{r,1}=0$.
        print("Constraint: discriminating feature 1")
        for r in range(1, K + 1):
            # d^{0}_{r,1}=0
            d1_var_name, d1_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_D_ONE_VARS, [r, 1]))
            self.wcnf.append([-d1_var_id])
            self.s.add_clause([-d1_var_id])
            # ###print("Discriminate feature 1: add constraint -{}".format(d1_var_name))
            for j in range(2, N + 1):
                rhs_or_names = []

                if atleast > 0:
                    if j % 2 == 1:
                        m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j]))
                    else:
                        m_var_name, m_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))

                for i in self.get_p_bounds(j=j, N=N):
                    # (p_{ji}\land d^0_{ri})
                    # same as the previous function
                    if reduced <= 0:
                        p_var_name, p_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_P_VARS, [j, i]))
                    elif j in self.get_l_bounds(i=i, N=N) and j % 2 == 0:
                        p_var_name, p_var_id = self.get_p_variable_by_x_l(j, i, N)
                    elif j in self.get_r_bounds(i=i, N=N) and j % 2 == 1:
                        p_var_name, p_var_id = self.get_p_variable_by_x_r(j, i, N)
                    else:
                        continue

                    d1_i_var_name, d1_i_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_D_ONE_VARS, [r, i]))
                    temp_p_d_var_name, temp_p_d_var_id = self.reified_and(p_var_name, d1_i_var_name)
                    rhs_or_names.append(temp_p_d_var_name)

                    # ###print("Discriminate feature 1: add constraint {}<-> {} AND {} ".format(temp_p_d_var_name, p_var_name, d1_i_var_name))

                    # (a_{ri}\land r_{ij})
                    if j % 2 == 0 and j in self.get_l_bounds(i=i, N=N):
                        a_var_name, a_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_A_VARS, [r, i]))
                        if reduced <= 0:
                            l_var_name, l_var_id = self.lookup_varid(
                                self.create_indexed_variable_name(LABEL_L_VARS, [i, j]))
                        else:
                            l_var_name, l_var_id = self.get_l_variablle_by_x(i, j)
                        temp_a_l_var_name, temp_a_l_var_id = self.reified_and(a_var_name, l_var_name)

                        rhs_or_names.append(temp_a_l_var_name)
                        # ###print("Discriminate feature 1: add constraint {}<-> {} AND {} ".format(temp_a_l_var_name, a_var_name, l_var_name))

                # d1_{r,j} --> \/ of tem vars
                d1_r_var_name, d1_r_var_id = self.lookup_varid(
                    self.create_indexed_variable_name(LABEL_D_ONE_VARS, [r, j]))
                if atleast > 0:
                    self.equivalence_lhs_literal_rhs_or(d1_r_var_name, rhs_or_names, atleast_var_name=m_var_name)
                else:
                    self.equivalence_lhs_literal_rhs_or(d1_r_var_name, rhs_or_names)
                # ###print("Discriminate feature 1: add constraint {}<-> {}  ".format(d1_r_var_name, rhs_or_names))

    def generate_feature_used_in_node(self, N, K, reduced=-1, atleast=-1):
        #  2019-5-8 Hao HU: add the option of reduced
        #         \item Using a feature $r$ at node $j$, with
        #     $1\le r\le K$ and $1\le j\le N$:
        #     \begin{equation} \label{eq:uf1}
        #       \begin{array}{l}
        #         \bigwedge\limits_{i=\lfloor\frac{j}{2}\rfloor}^{j-1}
        #         \left(u_{ri}\land p_{ji}\limply \neg a_{rj}\right) \\[4pt]
        #         %
        #         %
        #         u_{rj} \leftrightarrow \left(a_{rj} \lor
        #         \bigvee\limits_{i=\lfloor\frac{j}{2}\rfloor}^{j-1}
        #         (u_{ri}\land p_{ji}) \right)\\
        #       \end{array}
        #     \end{equation}
        print("Constraint: feature used in nodes")
        for r in range(1, K + 1):
            for j in range(1, N + 1):

                if atleast > 0:
                    if j % 2 == 1:
                        m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j]))
                    else:
                        m_var_name, m_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))

                for i in self.get_p_bounds(j=j, N=N):
                    # \left(u_{ri}\land p_{ji}\limply \neg a_{rj}\right)
                    u_var_name, u_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_U_VARS, [r, i]))
                    if reduced <= 0:
                        p_var_name, p_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_P_VARS, [j, i]))
                    elif j in self.get_l_bounds(i=i, N=N) and j % 2 == 0:
                        p_var_name, p_var_id = self.get_p_variable_by_x_l(j, i, N)
                    elif j in self.get_r_bounds(i=i, N=N) and j % 2 == 1:
                        p_var_name, p_var_id = self.get_p_variable_by_x_r(j, i, N)
                    else:
                        continue
                    a_var_name, a_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_A_VARS, [r, j]))

                    # add for atleast condition
                    if atleast > 0:
                        self.wcnf.append([-m_var_id, -u_var_id, -p_var_id, -a_var_id])
                        self.s.add_clause([-m_var_id, -u_var_id, -p_var_id, -a_var_id])
                    else:
                        self.wcnf.append([-u_var_id, -p_var_id, -a_var_id])
                        self.s.add_clause([-u_var_id, -p_var_id, -a_var_id])

                #         u_{rj} \leftrightarrow \left(a_{rj} \lor
                #         \bigvee\limits_{i=\lfloor\frac{j}{2}\rfloor}^{j-1}
                #         (u_{ri}\land p_{ji}) \right)\\
                u_j_var_name, u_j_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_U_VARS, [r, j]))
                a_var_name, a_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_A_VARS, [r, j]))

                rhs_or_names = []
                for i in self.get_p_bounds(j=j, N=N):
                    u_var_name, u_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_U_VARS, [r, i]))
                    if reduced <= 0:
                        p_var_name, p_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_P_VARS, [j, i]))
                    elif j in self.get_l_bounds(i=i, N=N) and j % 2 == 0:
                        p_var_name, p_var_id = self.get_p_variable_by_x_l(j, i, N)
                    elif j in self.get_r_bounds(i=i, N=N) and j % 2 == 1:
                        p_var_name, p_var_id = self.get_p_variable_by_x_r(j, i, N)
                    else:
                        continue
                    temp_u_p_var_name, temp_u_p_var_id = self.reified_and(u_var_name, p_var_name)
                    rhs_or_names.append(temp_u_p_var_name)
                rhs_or_names.append(a_var_name)
                if atleast > 0:
                    self.equivalence_lhs_literal_rhs_or(u_j_var_name, rhs_or_names, atleast_var_name=m_var_name)
                else:
                    self.equivalence_lhs_literal_rhs_or(u_j_var_name, rhs_or_names)

    def generate_nonleaf_has_one_feature(self, N, K, atleast=-1):
        #         \item For a non-leaf node $j$, exactly one feature is used:
        #     \begin{equation} \label{eq:uf2}
        #       \begin{array}{lcr}
        #         %\neg v_j\rightarrow\left(\sum_{r=1}^{K}b_{rj}=1\right)
        #         \neg v_j\rightarrow\left(\sum\limits_{r=1}^{K}a_{rj}=1\right)
        #         & \quad\quad &
        #         \tn{with~} j=1\ldots,N\\
        #       \end{array}
        #     \end{equation}
        #
        print("Constraint: nonleaf node has one feature")
        for j in range(1, N + 1):
            v_var_name, v_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [j]))

            if atleast > 0:
                if j % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))

            a_names = []
            for r in range(1, K + 1):
                a_var_name, a_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_A_VARS, [r, j]))
                a_names.append(a_var_name)
            if atleast > 0:
                # print(j)
                # print(str(len(a_names)) + '\n')
                self.generate_reified_card(v_var_name, a_names, atleast_var_name=m_var_name)
            else:
                self.generate_reified_card(v_var_name, a_names)
            # ###print("Non leaf has one feature: card : add constraint -{} -> {} = 1".format(v_var_name, a_names))

    def generate_leaf_has_no_features(self, N, K, atleast=-1):
        # \item For a leaf node $j$, no feature is used:
        #     \begin{equation} \label{eq:uf3}
        #       \begin{array}{lcr}
        #         %v_j\rightarrow\left(\sum_{r=1}^{K}b_{rj}=0\right)
        #         v_j\rightarrow\left(\sum\limits_{r=1}^{K}a_{rj}=0\right)
        #         & \quad\quad &
        #         \tn{with~} j=1\ldots,N\\
        #       \end{array}
        #     \end{equation}
        print("Constraint: leaf has no features")
        for j in range(1, N + 1):
            v_var_name, v_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [j]))

            if atleast > 0:
                if j % 2 == 1:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j]))
                else:
                    m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))
            for r in range(1, K + 1):
                a_var_name, a_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_A_VARS, [r, j]))
                if atleast > 0:
                    self.wcnf.append([-m_var_id, -v_var_id, -a_var_id])
                    self.s.add_clause([-m_var_id, -v_var_id, -a_var_id])
                else:
                    self.wcnf.append([-v_var_id, -a_var_id])
                    self.s.add_clause([-v_var_id, -a_var_id])
                # ###print("Leaf has no features: card : add constraint -{} \/ -{}".format(v_var_name, a_var_name))

    def generate_account_for_examples(self, N, K, out_samp_index=[], option=1, atleast=-1):
        # -------------------------------------------------------------------------------#
        # 2019-3-8 Hao HU: add the option of out_samp_index for indicating the list of 
        #                  index of examples will not used in training data. By default, let it vide
        # 2019-3-15 Hao HU: add the option of getting the wcnf or cnf!
        #                   option == 1 -> wcnf/ others -> cnf
        # -------------------------------------------------------------------------------#
        print("Constraint: accounts for examples")
        sample_index = 1

        # out_samp_index is reserved for k-fold cross-validation
        samps_train = [self.data.samps[i] for i in range(len(self.data.samps)) if i not in out_samp_index]

        # 2020-1-16: for handling the problem of consitent, we give every example the same weight no matter it is duplicated or with different label
        # it could be much useful in our MaxSAT encoding because MaxSAT could find an assignment to minimizing the misclassification examples
        for sample in samps_train:
            pos = sample[-1]
            sample = sample[:-1]
            # ###print(sample)
            sample = [1 if x > 0 else 0 for x in sample]
            # ###print("binary", sample)
            for j in range(1, N + 1):
                v_var_name, v_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [j]))
                c_var_name, c_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_C_VARS, [j]))

                # get the atleast variable
                if atleast > 0:
                    if j % 2 == 1:
                        m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [j]))
                    else:
                        m_var_name, m_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))

                r_or_names = []
                r_or = []
                for r in range(1, K + 1):
                    # get the \sigma(r, q)
                    if (sample[r - 1] == 1):
                        d1_var_name, d1_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_D_ONE_VARS, [r, j]))
                        r_or_names.append(d1_var_name)
                        r_or.append(d1_var_id)

                    if (sample[r - 1] == 0):
                        d0_var_name, d0_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_D_ZERO_VARS, [r, j]))
                        r_or_names.append(d0_var_name)
                        r_or.append(d0_var_id)

                    assert ((sample[r - 1] == 1) or (sample[r - 1] == 0))

                if (pos == 1):
                    temp_v_c_var_name, temp_v_c_var_id = self.reified_and(a_var_name=v_var_name, b_var_name=c_var_name,
                                                                          a_pos=1, b_pos=-1)
                else:
                    temp_v_c_var_name, temp_v_c_var_id = self.reified_and(a_var_name=v_var_name, b_var_name=c_var_name,
                                                                          a_pos=1, b_pos=1)

                r_or.append(-temp_v_c_var_id)

                # do not change the clauses for SAT-Solver
                self.s.add_clause(r_or)
                if option == 1:
                    # --------------------------------------#
                    # 2019-3-7: Hao HU - change the hard clause to soft clauses
                    # add the blocking literal of each sample
                    b_var_name, b_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_B_VARS, [sample_index]))
                    r_or.append(-b_var_id)
                    if atleast > 0:
                        r_or.append(-m_var_id)
                    # making sure that the clauses with blocking literals are hard
                    self.wcnf.append(r_or)
                    # --------------------------------------#
                else:
                    self.wcnf.append(r_or)
            # making the blocking literals are soft clauses for each examples
            # Here firstly, considering all clauses of blocking literals with same weights -> final, considering the duplicate examples with same weight
            if option == 1:
                self.wcnf.append([b_var_id], weight=1)
            sample_index = sample_index + 1

    def useful_feature_selection(self, N, atleast=-1):
        """
            2019-6-24: add a new constraint for avoid useless feature selection
            For every internal node has two leaf nodes, those leaf nodes must be different class
            A practical way is to set for every potenial leaf child, the right child must have a different class 
            without considering it is internal node or leaf node 
        """
        print("Constraint: two child of same non-leaf node must have different node value")
        for i in range(1, N + 1):
            for j in self.get_l_bounds(i, N):
                if j % 2 == 0:
                    clause = []
                    c_var_name, c_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_C_VARS, [j]))
                    c_next_var_name, c_next_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_C_VARS, [j + 1]))

                    # add the constraint c_{j} \to \neg c_{j+1} and \neg c_{j} \to c_{j+1}
                    if atleast > 0:
                        m_var_name, m_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_M_VARS, [j + 1]))
                        clause.append([-c_id, -c_next_id, -m_var_id])
                        clause.append([c_id, c_next_id, -m_var_id])
                    else:
                        clause.append([-c_id, -c_next_id])
                        clause.append([c_id, c_next_id])

                    self.wcnf.extend(clause)
                    self.s.append_formula(clause)

    def generate_atmost_one4non_binary_features(self, N, K):
        for j in range(1, N + 1):
            for nonbin, binfeats in self.data.nonbin2bin.items():
                # #print(nonbin, binfeats)
                d0_vars = []
                d0_names = []
                for binfeat in binfeats:
                    r = self.data.nm2id[binfeat] + 1
                    if (r <= K):
                        # #print("column ", index)
                        d0_var_name, d0_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_D_ZERO_VARS, [r, j]))
                        d0_vars.append(d0_var_id)
                        d0_names.append(d0_var_name)
                # #print(d0_vars, d0_names)
                if (len(d0_vars) > 1):
                    at_most = d0_vars[:]
                    one_out = CardEnc().atmost(lits=at_most, top_id=self.max_id, encoding=self.options.enc)
                    self.max_id = one_out.nv
                    # copying cardinality constraint
                    for cl in one_out.clauses:
                        self.wcnf.append(cl)
                        self.s.add_clause(cl)

    # 2019-5-8 Hao HU: add option reduced as using the reduced encoding
    def generate_classifer_constraints(self, N, K, out_index=[], option=1, reduced=-1, atleast=-1):
        self.generate_discriminate_feature_0(N=N, K=K, reduced=reduced, atleast=atleast)
        self.generate_discriminate_feature_1(N=N, K=K, reduced=reduced, atleast=atleast)
        self.generate_feature_used_in_node(N=N, K=K, reduced=reduced, atleast=atleast)
        self.generate_nonleaf_has_one_feature(N=N, K=K, atleast=atleast)
        self.generate_leaf_has_no_features(N=N, K=K, atleast=atleast)
        self.generate_account_for_examples(N=N, K=K, out_samp_index=out_index, option=option, atleast=atleast)
        # self.generate_atmost_one4non_binary_features(N=N, K=K)
        self.useful_feature_selection(N=N, atleast=atleast)
        print("All constraints are generated!!")

    def print_solution_c(self, N, model, K=None):
        # ###print("c")
        for i in range(1, N + 1):
            c_var_name, c_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_C_VARS, [i]))
            if (model[c_var_id - 1] > 0):
                print(c_var_name + "= {}".format(1))
            else:
                print(c_var_name + "= {}".format(0))

    def print_solution_v(self, N, model, K=None):
        # ###print("v")
        for i in range(1, N + 1):
            v_var_name, v_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [i]))
            if (model[v_var_id - 1] > 0):
                print(v_var_name + "= {}".format(1))
            else:
                print(v_var_name + "= {}".format(0))

    def print_solution_d_zero(self, N, model, K=None):
        # ###print("D0")
        for r in range(1, K + 1):
            for j in range(1, N + 1):
                d0_i_j_var_name, d0_i_j_var_id = self.lookup_varid(
                    self.create_indexed_variable_name(LABEL_D_ZERO_VARS, [r, j]))
                if (model[d0_i_j_var_id - 1] > 0):
                    print(d0_i_j_var_name + "= {}".format(1))
                else:
                    print(d0_i_j_var_name + "= {}".format(0))

    def print_solution_d_one(self, N, model, K=None):
        # ###print("D1")
        for r in range(1, K + 1):
            for j in range(1, N + 1):
                d1_i_j_var_name, d1_i_j_var_id = self.lookup_varid(
                    self.create_indexed_variable_name(LABEL_D_ONE_VARS, [r, j]))
                if (model[d1_i_j_var_id - 1] > 0):
                    print(d1_i_j_var_name + "= {}".format(1))
                else:
                    print(d1_i_j_var_name + "= {}".format(0))

    def get_number_nodes_really_used(self, N, model):
        """
            get the number of nodes really used 
        """
        assert N % 2 == 1
        nb_node_used = 1
        for i in range(1, N + 1):
            if i % 2 == 0:
                continue
            m_var_name, m_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_M_VARS, [i]))
            if model[m_var_id - 1] > 0:
                # if atleast i nodes are used
                nb_node_used = i
            else:
                break
        return nb_node_used

    def build_graph(self, N, model, filename="graph.png", K=None, labeled=False, reduced=-1, atleast=-1):
        # 2019-5-8 Hao HU: add the option reduced for encoding reduced
        # 2019-12-10: add option "atleast" for using the atleast constraint
        self.s.file_dt = open(filename, "w")
        nodes = []
        labeldict = None

        # in "atleast" condition, get the number of nodes really used
        if atleast > 0:
            nb_node_used = self.get_number_nodes_really_used(N=N, model=model)
        else:
            nb_node_used = N
        print("Number of nodes used: " + str(nb_node_used))
        self.s.file_dt.write("NODES\n")
        if (labeled == True):
            labeldict = {}
            for r in range(1, K + 1):
                for j in range(1, nb_node_used + 1):
                    a_r_j_var_name, a_r_j_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_A_VARS, [r, j]))
                    if (model[a_r_j_var_id - 1] > 0):
                        labeldict[j] = "             " + self.data.names[r - 1] + "_" + str(r)
                        nodes.append((j, self.data.names[r - 1]))
                        self.s.file_dt.write("{} {}\n".format(j, self.data.names[r - 1]))
                        # ###print("r = {}, j = {}".format(r, j))
            for j in range(1, nb_node_used + 1):
                v_var_name, v_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [j]))
                c_var_name, c_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_C_VARS, [j]))
                if (model[v_var_id - 1] > 0):
                    # terminal
                    v = 1 if (model[c_var_id - 1] > 0) else 0
                    labeldict[j] = "             c_" + str(v)
                    self.s.file_dt.write("{} {}\n".format(j, v))

        self.s.file_dt.write("EDGES\n")
        ##print(nodes)
        edges = []
        for i in range(1, nb_node_used + 1):
            for j in self.get_l_bounds(i=i, N=nb_node_used):
                if (j % 2 == 0):
                    if reduced <= 0:
                        l_i_j_var_name, l_i_j_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_L_VARS, [i, j]))
                    else:
                        l_i_j_var_name, l_i_j_var_id = self.get_l_variablle_by_x(i, j)
                    if (model[l_i_j_var_id - 1] > 0):
                        # we know that i is parent of j
                        # G.add_edge(i, j, weight=0)
                        edges.append((i, j, 1))
                        self.s.file_dt.write("{} {} {}\n".format(i, j, 1))

            for j in self.get_r_bounds(i=i, N=nb_node_used):
                if (j % 2 == 1):
                    if reduced <= 0:
                        r_i_j_var_name, r_i_j_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_R_VARS, [i, j]))
                    else:
                        r_i_j_var_name, r_i_j_var_id = self.get_r_variable_by_x(i, j)
                    if (model[r_i_j_var_id - 1] > 0):
                        # we know that i is parent of j
                        edges.append((i, j, 0))
                        self.s.file_dt.write("{} {} {}\n".format(i, j, 0))

                        # G.add_edge(i, j, weight=1)
        self.s.file_dt.close()

    # ---------------------------------------------------------#
    # Hao HU: 2019-4-9 Add a function for only building the graph of binary tree
    # without using the feature
    # UPDATE: Hao HU 2019-5-7 Add the option of reduced, which suits for the condition of reduced literal
    def build_binary_tree(self, N, model, filename="bin_tree_graph.png", reduced=-1, atleast=-1):

        if atleast > 0:
            nb_node_used = self.get_number_nodes_really_used(N=N, model=model)
        else:
            nb_node_used = N

        file_bin_tree = open(filename, 'w')
        # because we know the root of binary tree is node 1, so just store the edges
        file_bin_tree.write("EDGES\n")
        for i in range(1, nb_node_used + 1):
            for j in self.get_l_bounds(i=i, N=nb_node_used):
                if (j % 2 == 0):
                    if reduced <= 0:
                        l_i_j_var_name, l_i_j_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_L_VARS, [i, j]))
                    else:
                        l_i_j_var_name, l_i_j_var_id = self.get_l_variablle_by_x(i, j)
                    if (model[l_i_j_var_id - 1] > 0):
                        file_bin_tree.write("{} {} {}\n".format(i, j, 1))

            for j in self.get_r_bounds(i=i, N=nb_node_used):
                if (j % 2 == 1):
                    if reduced <= 0:
                        r_i_j_var_name, r_i_j_var_id = self.lookup_varid(
                            self.create_indexed_variable_name(LABEL_R_VARS, [i, j]))
                    else:
                        r_i_j_var_name, r_i_j_var_id = self.get_r_variable_by_x(i, j)
                    if (model[r_i_j_var_id - 1] > 0):
                        file_bin_tree.write("{} {} {}\n".format(i, j, 0))
        file_bin_tree.close()

    # ---------------------------------------------------------#
    def prepare_cnf_file_path(self, basepath, dataset_name):
        """
            return the updated basepath for storing the .wcnf file
        """
        path = basepath
        reduced_version = 'original' if self.options.reduced <= 0 else 'reduced'
        test_way_folder = 'rest' if self.options.rest else 'kfold' if self.options.k_fold > 1 else 'percentage'
        # add the sub-folder for sampling ways
        folder_names = [dataset_name, reduced_version, test_way_folder, self.options.sampling]
        for sub_path in folder_names:
            path = path + '/' + sub_path
            if not os.path.exists(path):
                os.mkdir(path)
        return path

    def generate_formula(self, N, K=None, option=1, reduced=0):
        if K is None:
            K = len(self.data.samps[0]) - 1

        # generate different formula by using the reduced version or not
        if reduced == 0:
            self.generate_variables(N=N, K=K)
            self.generate_bin_tree_constraints(N=N)
        else:
            self.generate_variables_reduced(N=N, K=K)
            self.generate_bin_tree_constraints_reduced(N=N)
        self.generate_classifer_constraints(N=N, K=K, option=option, out_index=self.options.out_index, reduced=reduced)

        # -----------------------------------------------------------#
        # add the part of transforming the wcnf formula to .wcnf file
        # -----------------------------------------------------------#
        cnf = self.wcnf.unweighed()
        cnf_basepath = 'cnf'
        file_name = self.options.files[0].split('/')[-1].split('.')[0]
        cnf_path = self.prepare_cnf_file_path(cnf_basepath, file_name)

        cnf.to_file(
            cnf_path + '/formula_' + str(N) + '_' + str(self.options.seed) + '_' + str(self.options.ratio) + '.cnf')

        sol_opt = None

        t = time.time()
        res = self.s.solve()

        if res == True:
            reduce_name = 'orig' if reduced == 0 else 'redu'
            sol_path = cnf_path + '/' + file_name + '_' + str(N) + '_' + str(self.options.seed) + '_' + str(
                self.options.ratio) + '_' + reduce_name + '.sol'
            self.build_graph(N=N, model=self.s.get_model(), K=K, filename=sol_path, labeled=True, reduced=reduced)
            sol_opt = sol_path
        else:
            print("UNSAT")
        sat_time = time.time() - t
        # ###print(sat_time)
        return res, sat_time, sol_opt

    # -----------------------------------------------------------------------------------------------#
    # 2019-4-10 Hao HU: because of my stupid thought of if all concerned id of var for constructing the graph or binary tree are different,
    # it must result in different graph or tree. But in fact, there are some useless literals, which means it still has some potenial constraints:
    # when constructing the graph, for variable c_j, which means that node j is class 1 or 0. However, if node j is not leaf, it makes no sense
    # for the value c_j, but it will give two different solutions.
    #   So, finally, here I realize the function of comparing the dt files generated by different solutions, which I always want to avoid considering the 
    # effecifiency of reading many files. But now, whatever, there are thoudands of files, not too many.
    def compare_solution_file(self, s_path, s_c_path):
        """
            For the atomity of function, this will return true or false of comparasion two 
            solution files
        """
        equal = True
        with open(s_path, 'r+') as f:
            lines_f = f.readlines()
            with open(s_c_path, 'r+') as f_c:
                lines_f_c = f_c.readlines()

                for i in range(len(lines_f)):
                    if cmp(lines_f[i], lines_f_c[i]):
                        # print(lines_f[i])
                        # print(lines_f_c[i])
                        equal = False
                        break
        if equal:
            # delete the solution file of s_path
            os.remove(s_path)
        return equal

        # 2019-3-18 Hao HU: Add the function of getting serveral solutions for one specific cnf file

    def get_n_solutions(self, N, K, cnf, u_bound=10000):
        """
            This function will only be used in case of having solution for selected cnf file!

            because the cnf formula is stored in the solver object, so we can directly add some extra clauses.
            u_bound -> indicate the upper bound of solution found, in default 500, to avoid calculating so long time
        """
        cnt = 0
        sol_files = []
        # store the solution
        dir_path = 'cnf/' + str(N) + "_" + str(self.options.percentage)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if not os.path.exists(dir_path + '/solutions'):
            os.mkdir(dir_path + '/solutions')
        if not os.path.exists(dir_path + '/dtsol'):
            os.mkdir(dir_path + '/dtsol')
        if not os.path.exists(dir_path + '/selected_var'):
            os.mkdir(dir_path + '/selected_var')

        with Solver(bootstrap_with=cnf.clauses) as solver_temp:
            # ideas for reducing the number of files -> compare is there any changes for
            # variables concerned in constructing the graph
            id_var_dict = self.get_all_id_of_var_concerned_tree(N=N, K=K)
            # print(id_var_dict)
            id_value_dicts = []
            for m in solver_temp.enum_models():
                if cnt > u_bound:
                    break
                with open(dir_path + '/solutions/soltion_' + str(cnt), 'w') as f:
                    for item in m:
                        f.write("%s\n" % str(item))

                # build the decision tree by the solution found
                sol_file = dir_path + '/dtsol/solution_' + str(cnt) + '.sol'

                if cnt == 0:
                    self.build_graph(N=N, K=K, model=m, filename=sol_file, labeled=True)
                    sol_files.append(sol_file)
                    id_value_dict = {}
                    for id_index in id_var_dict.keys():
                        id_value_dict[id_index] = m[id_index - 1]
                    id_value_dicts.append(id_value_dict.copy())
                # compare the value of key 
                else:
                    flag = False
                    cnt_diff = 0
                    ##print(id_value_dicts)
                    for dic in id_value_dicts:
                        for id_index, value in dic.items():
                            ##print(str(value) + " " + str(m[id_index - 1]))
                            if not value == m[id_index - 1]:
                                cnt_diff = cnt_diff + 1
                                break
                    if cnt_diff == len(id_value_dicts):
                        flag = True

                    if flag:
                        flag_new_sol = True
                        self.build_graph(N=N, K=K, model=m, filename=sol_file, labeled=True)
                        # add also the part of comparing the solution files
                        for sol_c_file in sol_files:
                            if self.compare_solution_file(sol_file, sol_c_file):
                                flag_new_sol = False
                                break
                        if flag_new_sol:
                            # print("INININ")
                            sol_files.append(sol_file)
                            id_value_dict = {}
                            for id_index in id_var_dict.keys():
                                id_value_dict[id_index] = m[id_index - 1]
                            id_value_dicts.append(id_value_dict.copy())
                cnt = cnt + 1
            s_cnt = 0
            for dic in id_value_dicts:
                with open(dir_path + '/selected_var/s_' + str(s_cnt), 'w') as f:
                    for index, value in dic.items():
                        f.write("%s\n" % (str(index) + ":" + str(value)))
                s_cnt = s_cnt + 1
        return cnt, sol_files

    # get all id of concerned of changing the structure of binary decision tree,
    # then judge is there any differences between different solution
    def get_all_id_of_var_concerned_tree(self, N, K):
        id_var_dict = {}
        for r in range(1, K + 1):
            for j in range(1, N + 1):
                a_r_j_var_name, a_r_j_var_id = self.lookup_varid(
                    self.create_indexed_variable_name(LABEL_A_VARS, [r, j]))
                id_var_dict[a_r_j_var_id] = a_r_j_var_name
        for j in range(1, N + 1):
            v_var_name, v_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [j]))
            c_var_name, c_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_C_VARS, [j]))
            id_var_dict[v_var_id] = v_var_name
            id_var_dict[c_var_id] = c_var_name
        for i in range(1, N + 1):
            for j in self.get_l_bounds(i=i, N=N):
                if (j % 2 == 0):
                    l_i_j_var_name, l_i_j_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_L_VARS, [i, j]))
                    id_var_dict[l_i_j_var_id] = l_i_j_var_name
            for j in self.get_r_bounds(i=i, N=N):
                if (j % 2 == 1):
                    r_i_j_var_name, r_i_j_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_R_VARS, [i, j]))
                    id_var_dict[r_i_j_var_id] = r_i_j_var_name
        return id_var_dict

    # -----------------------------------------------------------------------------------------------#

    # -----------------------------------------------------------------------------------------------#
    # 2019-4-24 Hao HU: Add the function of preparing the valid path of wcnf files, and .sol files
    def prepare_wcnf_file_path(self, basepath, dataset_name):
        """
            return the updated basepath for storing the .wcnf file
        """
        path = basepath
        test_way_folder = 'rest' if self.options.rest else str(
            self.options.k_fold) + '-fold' if self.options.k_fold > 1 else 'percentage'
        # add the sub-folder for sampling ways
        folder_names = [dataset_name, test_way_folder, self.options.sampling]
        for sub_path in folder_names:
            path = path + '/' + sub_path
            if not os.path.exists(path):
                os.mkdir(path)
        return path

    # 2019-10-17 Hao HU: the function of extracting the optimal info and model from the output text
    def extract_info_solver_output(self, output_info):
        """
            Input: the file of output information file
            Output: The optimization value o, the model m(assignment), and the status s
            
            line starting with 'c' means comments, 'o' means the optimal solution, 's' means the solution status
            'v' means the model of assignment for each variable => There are some problem with grep when the variable is in large size
            
            There is a comment line containing the total time of MaxSAT solver, we extract this as the maxsat_time
        """
        print("Extracting the info from solver....")
        status = 'UNSATISFIABLE'
        var_model = []
        o = -1
        total_time = 0
        for info_line in output_info.readlines():
            if info_line.startswith('c'):
                # 2019-10-18 Hao HU: considering extracting the total time of the solver
                if not info_line.find('Total time') == -1:
                    # for Loandra solver format
                    total_time = float(info_line.strip().replace(' ', '').split(':')[-1][:-1])
                elif not info_line.find('oracle time') == -1:
                    # for RC2 solver format
                    total_time = float(info_line.strip().replace(' ', '').split(':')[-1])
                else:
                    continue
            elif info_line.startswith('o'):
                o_current = int(info_line.split(' ')[1])
                if o == -1:
                    o = o_current
                if o_current < o:
                    o = o_current
            elif info_line.startswith('s'):
                print(info_line)
                status = info_line.split(' ')[1]
            elif info_line.startswith('v'):
                var_model = [int(i) for i in info_line.strip(' \n').split(' ')[1:]]
            else:
                print("Something wrong in the output of incomplete MaxSAT solver!!")
                exit()
        return o, var_model, status, total_time

    def get_training_example_classification_result(self, var_model):
        """
            function of getting the classification results based on current model.
            0 -> wrongly classified, 1 -> well classified
            Input: var_model: the model from solver
            Output: the array of classification results based on var_model for soft clauses in self.wcnf
        """
        c_results = []
        for s_clause in self.wcnf.soft:
            wrong_f = False
            for v_s_cluse in s_clause:
                if var_model[v_s_cluse - 1] < 0:
                    c_results.append(0)
                    wrong_f = True
                    break
            if not wrong_f:
                c_results.append(1)

        return c_results

    def generate_decision_tree_of_maxSAT(self, N, K=None, reduced=-1, max_depth=-1, depth=-1, atleast=-1, sol_path=None,
                                         complete=1, maxsat_solver='RC2', gtimeout=900, \
                                         complete_tlimit=0, u_wght_soft=[], call_m='', core_ptime=600,
                                         wcnf_timeout=1800):
        """
            2019-3-8 Hao HU: Add the function of getting the structure of decision tree build by the MaxSAT
                a kind of rewriting the function of "generate_formula", but get the value from the MaxSAT solver
            2019-5-8 Hao HU: add the new option reduced indicating using the reduced encoding
            2019-10-17 Hao HU: change the option 'solver_type' into three options indicating the type and name of MaxSAT solver
               complete -> complete or incomplete MaxSAT solver / gtimeout -> only useful when using incomplete solver, global timeout for solver in seconds
            2019-10-23 Hao HU: add option 'complete_tlimit' for setting time limit for both complete MaxSAT solver and incomplete, so that they are fair
                0 -> global timeout is not for both, 1 -> gtimeout is for both complete and incomplete
            2019-12-5 Hao HU: add option 'max_depth' for controlling the maximum depth
            2020-1-17: add the option of remembering the call method for avoiding the conflit of name of wcnf
        """
        assert not (max_depth > 0 and depth > 0)
        if K is None:  # number of features
            K = len(self.data.samps[0]) - 1
        reduced_name = "" if reduced <= 0 else "_reduced"
        complete_name = "" if complete == 1 else "_incomplete"
        incomplete_time = "" if complete == 1 else '_' + str(gtimeout)
        wcnf_time = -1

        def handler(signum, frame):
            raise TimeoutError("Timeout in WCNF generation: writing to file!")

        # create or update the wcnf file
        if len(u_wght_soft) == 0:
            # create the wcnf file at first -> generating the values in self.wcnf
            time_begin = time.time()
            self.generate_variables(N=N, K=K, max_depth=max_depth, depth=depth, atleast=atleast, reduced=reduced)
            if (time.time() - time_begin) > wcnf_timeout:
                raise TimeoutError("Timeout in WCNF generation: variable!")

            signal.signal(signal.SIGALRM, handler)
            print(
                "Time left for preparing constraint binary tree: " + str(wcnf_timeout - int(time.time() - time_begin)))
            signal.alarm(wcnf_timeout - int(time.time() - time_begin))
            self.generate_bin_tree_constraints(N=N, max_depth=max_depth, depth=depth, reduced=reduced, atleast=atleast)
            signal.alarm(0)

            signal.signal(signal.SIGALRM, handler)
            print("Time left for preparing constraint classification: " + str(
                wcnf_timeout - int(time.time() - time_begin)))
            signal.alarm(wcnf_timeout - int(time.time() - time_begin))
            self.generate_classifer_constraints(N=N, K=K, out_index=self.options.out_index, reduced=reduced,
                                                atleast=atleast)
            signal.alarm(0)

            # add the wcnf formula to the .wcnf file -> making the hard clause always been hard
            self.wcnf.topw = sum(self.wcnf.wght) + 1
            file_name = self.options.files[0].split('/')[-1].split('.')[0]
            wcnf_base_path = self.prepare_wcnf_file_path('binarytree/wcnf', file_name)
            # define the wcnf file path -> add the name for different max_depth
            # 2019-12-11: add the part of using "atleast" constraint or not
            atleast_version = 'atleast_' if self.options.atleast > 0 else ''
            if max_depth > 0:
                wcnf_file_path = wcnf_base_path + '/formula_' + str(self.options.ratio) + '_' + str(
                    self.options.seed) + '_' + atleast_version \
                                 + str(N) + '_max-' + str(max_depth) + reduced_name + complete_name
            elif depth > 0:
                wcnf_file_path = wcnf_base_path + '/formula_' + str(self.options.ratio) + '_' + str(
                    self.options.seed) + '_' + atleast_version \
                                 + str(N) + '_exact-' + str(depth) + reduced_name + complete_name
            else:
                wcnf_file_path = wcnf_base_path + '/formula_' + str(self.options.ratio) + '_' + str(
                    self.options.seed) + '_' \
                                 + atleast_version + str(N) + reduced_name + complete_name
            wcnf_file_path = wcnf_file_path + incomplete_time
            # do not store the wcnf file because of the size of disk -> 2019-10-17 store the wcnf file as temporary
            self.wcnf_file_path = wcnf_file_path + call_m + '.wcnf'

            signal.signal(signal.SIGALRM, handler)
            print("Time left for writing wcnf file: " + str(wcnf_timeout - int(time.time() - time_begin)))
            signal.alarm(wcnf_timeout - int(time.time() - time_begin))
            self.wcnf.to_file(self.wcnf_file_path)
            signal.alarm(0)
            wcnf_time = time.time() - time_begin
        else:
            # update the weights of soft clauses in exisiting wcnf formula
            assert len(u_wght_soft) == len(self.wcnf.wght)
            assert not os.path.isfile(self.wcnf_file_path)
            # 2021-4-14:a small change so that all wcnf in different iterations could be stored
            self.wcnf_file_path = '_'.join(self.wcnf_file_path.split('_')[:-2]) + '_' + call_m + '.wcnf'
            self.wcnf.wght = u_wght_soft
            self.wcnf.topw = sum(u_wght_soft) + 1
            self.wcnf.to_file(self.wcnf_file_path)

        # print the size information of wcnf formula
        nb_var = self.wcnf.nv
        nb_hard = len(self.wcnf.hard)
        nb_soft = len(self.wcnf.soft)
        count_literal = lambda l: sum([len(i) for i in l])
        nb_literals = count_literal(self.wcnf.soft) + count_literal(self.wcnf.hard)
        print("nb variables: " + str(nb_var))
        print("nb soft clauses: " + str(nb_soft))
        print("nb hard clauses: " + str(nb_hard))
        print("nb literals: " + str(nb_literals))
        print("nb features used: " + str(K))

        var_model = []
        o = -1

        if complete == 1:
            # complete MaxSAT solver
            if maxsat_solver == 'RC2':
                # 2019-10-23 Add the option of setting global timeout for the complete solver
                if complete_tlimit == 0:
                    t = time.time()
                    with RC2(self.wcnf) as rc2:
                        var_model = rc2.compute()
                        o = rc2.cost
                    maxsat_time = time.time() - t
                else:
                    # //TODO: need be improved considering the multi-process in python
                    # Because the difficult implementation in timelimite for CPU-concerning function in Python
                    # Use command line with timeout for simplifying the use of time limit
                    file_temp = os.popen('timeout ' + str(
                        gtimeout) + 's python ~/Software/pysat/examples/rc2.py -vv ' + self.wcnf_file_path)
                    # extract the solutions, and some other information
                    o, var_model, status, maxsat_time = self.extract_info_solver_output(file_temp)
                    file_temp.close()
            else:
                print("This MaxSAT solver is not available here!")
                exit()
        else:
            # incomplete MaxSAT solver, where gtimout is useful 
            if maxsat_solver == 'RC2' or maxsat_solver == 'Loandra':
                # considering the default of complete solver. -> the default incomplete solver used is Loandra, the chammpion of 2019 MaxSAT

                # command line -> it depends the path and mode of installing the Loandra + the global timeout used
                # timeout 15s ./loandra_static -pmreslin-cglim=30 -print-model ../example_input
                # the command line option -> print the model found + core-guided time limit = 60s + weight-based strategy
                os.chdir(f"Loandra/bin")
                commandline = "timeout " + str(gtimeout) + "s ./loandra_static -pmreslin-cglim=" + str(core_ptime) + \
                              " -weight-strategy=1 -print-model " + "../../" + self.wcnf_file_path
                file_temp = os.popen(commandline)
                # extract the best solution, models, optimality and global time from the output information text
                o, var_model, status, maxsat_time = self.extract_info_solver_output(file_temp)
                file_temp.close()
                os.chdir("../..")
            else:
                print("This incomplete MaxSAT solver is not available here!")

        # get the results of classification in training samples
        if len(var_model) > 0:
            c_results = self.get_training_example_classification_result(var_model)
        else:
            print("No solution found in limited time!!")

        # should delete the wcnf file when considering the real case
        assert os.path.isfile(self.wcnf_file_path)
        os.remove(self.wcnf_file_path)

        # prepare the sol file
        if sol_path is None:
            sol_file_name = wcnf_file_path + "{}".format("_best") + ".sol"
        else:
            sol_file_name = sol_path
        # self.build_binary_tree(N=N, model=var_model, filename=sol_file_name, reduced=reduced, atleast=atleast)
        if len(var_model) == 0:
            if call_m == 'adaboost' or call_m == 'bagging':
                return -1, _, _, _, _
            else:
                exit()
        else:
            self.build_graph(N=N, model=var_model, filename=sol_file_name, K=K, labeled=True, reduced=reduced,
                             atleast=atleast)

        return o, maxsat_time, sol_file_name, c_results, self.wcnf.wght, status, nb_var, nb_soft, nb_hard, nb_literals, wcnf_time

    def generate_decision_tree_of_maxSAT_using_min_features(self, N, K=None, reduced=-1, max_depth=-1, depth=-1, atleast=-1, sol_path=None,
                                         complete=1, maxsat_solver='RC2', gtimeout=900, \
                                         complete_tlimit=0, u_wght_soft=[], call_m='', core_ptime=600,
                                         wcnf_timeout=1800):
        """
            2019-3-8 Hao HU: Add the function of getting the structure of decision tree build by the MaxSAT
                a kind of rewriting the function of "generate_formula", but get the value from the MaxSAT solver
            2019-5-8 Hao HU: add the new option reduced indicating using the reduced encoding
            2019-10-17 Hao HU: change the option 'solver_type' into three options indicating the type and name of MaxSAT solver
               complete -> complete or incomplete MaxSAT solver / gtimeout -> only useful when using incomplete solver, global timeout for solver in seconds
            2019-10-23 Hao HU: add option 'complete_tlimit' for setting time limit for both complete MaxSAT solver and incomplete, so that they are fair
                0 -> global timeout is not for both, 1 -> gtimeout is for both complete and incomplete
            2019-12-5 Hao HU: add option 'max_depth' for controlling the maximum depth
            2020-1-17: add the option of remembering the call method for avoiding the conflit of name of wcnf
        """
        assert not (max_depth > 0 and depth > 0)
        if K is None:  # number of features
            K = len(self.data.samps[0]) - 1
        reduced_name = "" if reduced <= 0 else "_reduced"
        complete_name = "" if complete == 1 else "_incomplete"
        incomplete_time = "" if complete == 1 else '_' + str(gtimeout)
        wcnf_time = -1

        def handler(signum, frame):
            raise TimeoutError("Timeout in WCNF generation: writing to file!")

        # create or update the wcnf file
        if len(u_wght_soft) == 0:
            # create the wcnf file at first -> generating the values in self.wcnf
            time_begin = time.time()
            self.generate_variables(N=N, K=K, max_depth=max_depth, depth=depth, atleast=atleast, reduced=reduced)
            if (time.time() - time_begin) > wcnf_timeout:
                raise TimeoutError("Timeout in WCNF generation: variable!")

            signal.signal(signal.SIGALRM, handler)
            print(
                "Time left for preparing constraint binary tree: " + str(wcnf_timeout - int(time.time() - time_begin)))
            signal.alarm(wcnf_timeout - int(time.time() - time_begin))
            self.generate_bin_tree_constraints(N=N, max_depth=max_depth, depth=depth, reduced=reduced, atleast=atleast)
            signal.alarm(0)

            signal.signal(signal.SIGALRM, handler)
            print("Time left for preparing constraint classification: " + str(
                wcnf_timeout - int(time.time() - time_begin)))
            signal.alarm(wcnf_timeout - int(time.time() - time_begin))
            self.generate_classifer_constraints(N=N, K=K, out_index=self.options.out_index, reduced=reduced,
                                                atleast=atleast)
            signal.alarm(0)

            # add the wcnf formula to the .wcnf file -> making the hard clause always been hard
            self.wcnf.topw = sum(self.wcnf.wght) + 1
            file_name = self.options.files[0].split('/')[-1].split('.')[0]
            wcnf_base_path = self.prepare_wcnf_file_path('binarytree/wcnf', file_name)
            # define the wcnf file path -> add the name for different max_depth
            # 2019-12-11: add the part of using "atleast" constraint or not
            atleast_version = 'atleast_' if self.options.atleast > 0 else ''
            if max_depth > 0:
                wcnf_file_path = wcnf_base_path + '/min_features_formula_' + str(self.options.ratio) + '_' + str(
                    self.options.seed) + '_' + atleast_version \
                                 + str(N) + '_max-' + str(max_depth) + reduced_name + complete_name
            elif depth > 0:
                wcnf_file_path = wcnf_base_path + '/min_features_formula_' + str(self.options.ratio) + '_' + str(
                    self.options.seed) + '_' + atleast_version \
                                 + str(N) + '_exact-' + str(depth) + reduced_name + complete_name
            else:
                wcnf_file_path = wcnf_base_path + '/min_features_formula_' + str(self.options.ratio) + '_' + str(
                    self.options.seed) + '_' \
                                 + atleast_version + str(N) + reduced_name + complete_name
            wcnf_file_path = wcnf_file_path + incomplete_time
            # do not store the wcnf file because of the size of disk -> 2019-10-17 store the wcnf file as temporary
            self.wcnf_file_path = wcnf_file_path + call_m + '.wcnf'

            signal.signal(signal.SIGALRM, handler)
            print("Time left for writing wcnf file: " + str(wcnf_timeout - int(time.time() - time_begin)))
            signal.alarm(wcnf_timeout - int(time.time() - time_begin))
            self.wcnf.to_file(self.wcnf_file_path)
            signal.alarm(0)
            wcnf_time = time.time() - time_begin
        else:
            # update the weights of soft clauses in exisiting wcnf formula
            assert len(u_wght_soft) == len(self.wcnf.wght)
            assert not os.path.isfile(self.wcnf_file_path)
            # 2021-4-14:a small change so that all wcnf in different iterations could be stored
            self.wcnf_file_path = '_'.join(self.wcnf_file_path.split('_')[:-2]) + '_' + call_m + '.wcnf'
            self.wcnf.wght = u_wght_soft
            self.wcnf.topw = sum(u_wght_soft) + 1
            self.wcnf.to_file(self.wcnf_file_path)

        # print the size information of wcnf formula
        nb_var = self.wcnf.nv
        nb_hard = len(self.wcnf.hard)
        nb_soft = len(self.wcnf.soft)
        count_literal = lambda l: sum([len(i) for i in l])
        nb_literals = count_literal(self.wcnf.soft) + count_literal(self.wcnf.hard)
        print("nb variables: " + str(nb_var))
        print("nb soft clauses: " + str(nb_soft))
        print("nb hard clauses: " + str(nb_hard))
        print("nb literals: " + str(nb_literals))
        print("nb features used: " + str(K))

        var_model = []
        o = -1

        if complete == 1:
            # complete MaxSAT solver
            if maxsat_solver == 'RC2':
                # 2019-10-23 Add the option of setting global timeout for the complete solver
                if complete_tlimit == 0:
                    t = time.time()
                    with RC2(self.wcnf) as rc2:
                        var_model = rc2.compute()
                        o = rc2.cost
                    maxsat_time = time.time() - t
                else:
                    # //TODO: need be improved considering the multi-process in python
                    # Because the difficult implementation in timelimite for CPU-concerning function in Python
                    # Use command line with timeout for simplifying the use of time limit
                    file_temp = os.popen('timeout ' + str(
                        gtimeout) + 's python ~/Software/pysat/examples/rc2.py -vv ' + self.wcnf_file_path)
                    # extract the solutions, and some other information
                    o, var_model, status, maxsat_time = self.extract_info_solver_output(file_temp)
                    file_temp.close()
            else:
                print("This MaxSAT solver is not available here!")
                exit()
        else:
            # incomplete MaxSAT solver, where gtimout is useful
            if maxsat_solver == 'RC2' or maxsat_solver == 'Loandra':
                # considering the default of complete solver. -> the default incomplete solver used is Loandra, the chammpion of 2019 MaxSAT

                # command line -> it depends the path and mode of installing the Loandra + the global timeout used
                # timeout 15s ./loandra_static -pmreslin-cglim=30 -print-model ../example_input
                # the command line option -> print the model found + core-guided time limit = 60s + weight-based strategy
                os.chdir(f"Loandra/bin")
                commandline = "timeout " + str(gtimeout) + "s ./loandra_static -pmreslin-cglim=" + str(core_ptime) + \
                              " -weight-strategy=1 -print-model " + "../../" + self.wcnf_file_path
                file_temp = os.popen(commandline)
                # extract the best solution, models, optimality and global time from the output information text
                o, var_model, status, maxsat_time = self.extract_info_solver_output(file_temp)
                file_temp.close()
                os.chdir("../..")
            else:
                print("This incomplete MaxSAT solver is not available here!")

        # get the results of classification in training samples
        if len(var_model) > 0:
            c_results = self.get_training_example_classification_result(var_model)
        else:
            print("No solution found in limited time!!")

        # should delete the wcnf file when considering the real case
        assert os.path.isfile(self.wcnf_file_path)
        os.remove(self.wcnf_file_path)

        # prepare the sol file
        if sol_path is None:
            sol_file_name = wcnf_file_path + "{}".format("_best") + ".sol"
        else:
            sol_file_name = sol_path
        # self.build_binary_tree(N=N, model=var_model, filename=sol_file_name, reduced=reduced, atleast=atleast)
        if len(var_model) == 0:
            if call_m == 'adaboost' or call_m == 'bagging':
                return -1, _, _, _, _
            else:
                exit()
        else:
            self.build_graph(N=N, model=var_model, filename=sol_file_name, K=K, labeled=True, reduced=reduced,
                             atleast=atleast)

        return o, maxsat_time, sol_file_name, c_results, self.wcnf.wght, status, nb_var, nb_soft, nb_hard, nb_literals, wcnf_time

    # -----------------------------------------------------------------------------------------------#

    # ---------------------------------------------------------------------------------#
    def get_all_id_of_var_concerned_binary_tree(self, N):
        id_var_dict = {}
        for j in range(1, N + 1):
            v_var_name, v_var_id = self.lookup_varid(self.create_indexed_variable_name(LABEL_V_VARS, [j]))
            id_var_dict[v_var_id] = v_var_name

        for i in range(1, N + 1):
            for j in self.get_l_bounds(i=i, N=N):
                if (j % 2 == 0):
                    l_i_j_var_name, l_i_j_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_L_VARS, [i, j]))
                    id_var_dict[l_i_j_var_id] = l_i_j_var_name
            for j in self.get_r_bounds(i=i, N=N):
                if (j % 2 == 1):
                    r_i_j_var_name, r_i_j_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_R_VARS, [i, j]))
                    id_var_dict[r_i_j_var_id] = r_i_j_var_name
        return id_var_dict

    def find_write_all_diff_between_solution_file_and_model(self, s_file_path, model, cnt, var_id_dict):
        """
            s_file_path: the first solution found for constructing specific binary tree
            model: the model found which constructs the same binary tree
            cnt: indicate the number of model found
            var_id_dict: the dict where variable id are keys, the name of variables are values

            It will create the file with same name as s_file_path but concated with '.diff' to record the difference 
        """
        file_comparer = FileComparer(s_file_path)
        var_changed = file_comparer.find_diff_between_solution_file_and_model(model, var_id_dict)

        with open(s_file_path + '.diff', 'a+') as f:
            f.write("Solution " + str(cnt) + ":")
            for item in var_changed:
                f.write("%s, " % item)
            f.write("\n")

    def get_all_solutions(self, N, cnf, base_path, id_var_name_dict, reduced=0):
        """
            get all solutions for cnf selected. Will not be used when using the dataset.
            It will be only used for getting all solutions for different binary trees for specific node 
            base_path: indicating the base path for storing the files.
        """
        cnt = 0
        sol_files = []
        sol_s_path_dict = {}

        s_path = base_path + '/solutions_' + str(N)
        dtsol_path = base_path + '/dtsol_' + str(N)

        if not os.path.exists(s_path):
            os.mkdir(s_path)
        if not os.path.exists(dtsol_path):
            os.mkdir(dtsol_path)

        id_value_dicts = {}
        if reduced == 0:
            id_var_dict = self.get_all_id_of_var_concerned_binary_tree(N=N)
        else:
            # considering the reduced condition, only x_ij and v_i, which are all counting literals when constructing the binary tree
            id_var_dict = {id: var for var, id in self.var2ids.items()}
        with Solver(bootstrap_with=cnf.clauses) as solver_temp:
            for m in solver_temp.enum_models():
                # build the binary tree by the solution found
                sol_file = dtsol_path + '/solution_' + str(cnt) + '.sol'

                # for finding the variables with different values, which 
                # construct the same binary tree, so store all solution in N <= 5
                if N <= 5:
                    with open(s_path + '/solution_' + str(cnt), 'w') as f:
                        for item in m:
                            f.write("%s\n" % str(item))

                if cnt == 0:
                    self.build_binary_tree(N=N, model=m, filename=sol_file, reduced=reduced)
                    with open(s_path + '/solution_' + str(cnt), 'w') as f:
                        for item in m:
                            f.write("%s\n" % str(item))
                    sol_s_path_dict[sol_file] = s_path + '/solution_' + str(cnt)
                    sol_files.append(sol_file)
                    id_value_dict = {}
                    for id_index in id_var_dict.keys():
                        id_value_dict[id_index] = m[id_index - 1]
                    id_value_dicts[s_path + '/solution_' + str(cnt)] = id_value_dict.copy()
                else:
                    flag = True
                    for key_s_path, dic in id_value_dicts.items():
                        cnt_same = 0
                        for id_index, value in dic.items():
                            if value == m[id_index - 1]:
                                cnt_same = cnt_same + 1
                        # which means it will construct the same binary tree
                        if cnt_same == len(dic.keys()):
                            flag = False
                            # print(key_s_path + ": " + str(cnt))
                            self.find_write_all_diff_between_solution_file_and_model(key_s_path, m, cnt,
                                                                                     id_var_name_dict)

                    # if flag == True, which means it found a totally different solution which will probabaly change the binary tree
                    # but it can not guarntee for constructing a new binary tree
                    if flag:
                        flag_new_sol = True
                        self.build_binary_tree(N=N, model=m, filename=sol_file, reduced=reduced)
                        # compare with the files of binary tree before to see is there some differences or not
                        for sol_c_file in sol_files:
                            if self.compare_solution_file(sol_file, sol_c_file):
                                flag_new_sol = False
                                break

                        if flag_new_sol:
                            # print("New binary tree found")
                            # then store the solution too
                            if N > 5:
                                with open(s_path + '/solution_' + str(cnt), 'w') as f:
                                    for item in m:
                                        f.write("%s\n" % str(item))
                            sol_s_path_dict[sol_file] = s_path + '/solution_' + str(cnt)
                            sol_files.append(sol_file)
                            id_value_dict = {}
                            for id_index in id_var_dict.keys():
                                id_value_dict[id_index] = m[id_index - 1]
                            id_value_dicts[s_path + '/solution_' + str(cnt)] = id_value_dict.copy()
                cnt = cnt + 1
        return cnt, sol_files

    def print_all_var_names(self, cnf, base_path):
        id_var_names = {}
        for i in range(cnf.nv):
            var_name = self.lookup_var_name_by_id(i + 1)
            # print(str(i+1) + ":" + var_name)
            id_var_names[i + 1] = var_name
        with open(base_path + '/var_name.txt', 'w') as f:
            for id_var, name_var in id_var_names.items():
                f.write("%s\n" % (str(id_var) + ":" + name_var))
        return id_var_names

        # Hao HU: 2019-4-12 Change the function of getting all solutions of Nina or new into a base function

    # which can get the results of different combinations
    def generate_all_solutions_of_binary_tree(self, N, nina_l_t, new, base_path):
        """
            nina_l_t -> int, 0 indicates not use the lambda and tau constraints in Nina's encoding; others, use it
            new -> same idea as the past. The constraints used is defined in the following function
            base_path -> indicates the base path used for storing the files generated
        """

        # data_name = self.data.fname.split('/')[-1]
        folder_names = [str(N), 'non_nina_non_new', 'non_nina_new', "nina_non_new", "nina_new"]

        for i in range(2):
            if i == 0:
                base_path = base_path + '/' + folder_names[i]
            if i == 1:
                base_path = base_path + '/' + folder_names[1 + 2 * nina_l_t + new]
            if not os.path.exists(base_path):
                os.mkdir(base_path)

        if nina_l_t == 0:
            self.generete_constant_variables()
            self.generate_binary_tree_variables_without_l_t(N=N)
            self.generate_bin_tree_constraints_without_la_and_tau(N=N)

            if new == 0:
                # some way of storing the files generated
                print("Without lambda and tau constraints of Nina, also without the use of new constraints!")
            else:
                print("Without lambda and tau constraints of Nina, but with the use of new constraints!")
                self.generate_childern_of_brother_constraint(N=N)
        else:
            # generate the varaible for the lamda and tau
            self.generete_lambda_variables(N=N)
            self.generete_tau_variables(N=N)
            self.generete_constant_variables()
            self.generate_binary_tree_variables_without_l_t(N=N)

            self.generate_bin_tree_constraints(N=N)
            if new == 0:
                print("With lambda and tau constraints of Nina, but without the use of new constraints!")
            else:
                self.generate_childern_of_brother_constraint(N=N)
                print("With lambda and tau constraints of Nina, also with the use of new constraints!")

        cnf = self.wcnf.unweighed()
        id_var_name_dict = self.print_all_var_names(cnf, base_path)
        cnf.to_file(base_path + '/formula_' + str(N) + '.cnf')
        cnf_num_solution, sol_files = self.get_all_solutions(N, cnf, base_path, id_var_name_dict)
        return cnf_num_solution, sol_files

    # Hao HU: 2019-5-7 Add the function of getting all solutions in reduced condition.
    # here the reduced condition will use only the original constraints of Nina
    def generate_all_solutions_of_binary_tree_reduced(self, N, base_path):
        folder_names = [str(N), 'nina_non_new_reduced']
        for item in folder_names:
            base_path = base_path + '/' + item
            if not os.path.exists(base_path):
                os.mkdir(base_path)

        # generate the reduced variables and original constraints of Nina for reduced condition
        self.generate_binary_tree_variables_reduced(N=N)
        self.generate_bin_tree_constraints_reduced(N=N)

        cnf = self.wcnf.unweighed()
        id_var_name_dict = self.print_all_var_names(cnf, base_path)
        cnf.to_file(base_path + '/formula_reduced_' + str(N) + '.cnf')
        cnf_num_solution, sol_files = self.get_all_solutions(N, cnf, base_path, id_var_name_dict, reduced=1)
        return cnf_num_solution, sol_files

    # Hao HU: 2019-4-11 Add the new constraints of ourselves
    def generate_childern_of_brother_constraint(self, N):
        """
            all i, for j in l_bound(i), l_i_j -> (not v_(i+1) -> l_(a+1)_(K+2))
            4-12: add a constraint for the last situation when the brother can not have the left child greater
             than the N-1
        """
        for i in range(1, N):
            # i from 1 to N-1 to make sure that it must has the last one
            for j in self.get_l_bounds(i, N):
                if (j % 2 == 0):
                    v_p_1_var_name, v_p_1_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_V_VARS, [i + 1]))
                    l_i_j_var_name, l_i_j_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_L_VARS, [i, j]))
                    if not j + 2 in self.get_l_bounds(i + 1, N):
                        # the situation of last node, which will directly indicate the next node will be a leaf
                        self.wcnf.append([-l_i_j_var_id, v_p_1_var_id])
                        self.s.add_clause([-l_i_j_var_id, v_p_1_var_id])
                        continue
                    assert (j + 2 in self.get_l_bounds(i + 1, N))
                    l_i_p_1_j_p_2_var_name, l_i_p_1_j_p_2_var_id = self.lookup_varid(
                        self.create_indexed_variable_name(LABEL_L_VARS, [i + 1, j + 2]))
                    self.wcnf.append([-l_i_j_var_id, v_p_1_var_id, l_i_p_1_j_p_2_var_id])
                    self.s.add_clause([-l_i_j_var_id, v_p_1_var_id, l_i_p_1_j_p_2_var_id])

    # ---------------------------------------------------------------------------------#

    # create and initialize primer
    # if self.options.indprime == False:
#    self.init_solver()
#
#     def prepare_formula(self):
#         """
#             Prepare a MaxSAT formula for prime enumeration.
#         """
#
#         # creating a formula
#         self.formula = WCNF()
#
#         # formula's variables
#         self.orig_vars = max(self.data.fvmap.opp.keys())
#         self.formula.nv = self.orig_vars * 2
#
#         # soft clauses, p clauses and soft weights
#         self.id2var = {}
#         for v in xrange(1, self.orig_vars + 1):
#             if v not in self.data.deleted:
#                 self.formula.soft.append([-v])
#                 self.id2var[len(self.formula.soft)] = v
#
#                 self.formula.soft.append([-v - self.orig_vars])
#                 self.id2var[len(self.formula.soft)] = -v
#
#                 self.formula.hard.append([-v, -v - self.orig_vars])  # p clauses
#
#         self.formula.wght = [1 for cl in self.formula.soft]
#         self.formula.topw = len(self.formula.soft) + 1
#
#         # hard clauses
#         for sample in self.data.samps:
#             cl = map(lambda l: -l if l < 0 else l + self.orig_vars, sample[:-1]) + [sample[-1]]
#             self.formula.hard.append(cl)
#
#         # additional hard clauses (exactly one outcome possible)
#         outs = [self.data.fvmap.dir[(self.data.names[-1], l)] for l in self.data.feats[-1]]
#         one_out = CardEnc().equals(lits=outs, top_id=2 * self.orig_vars, encoding=self.options.enc)
#
#         # updating top id (take Tseitin variable into account)
#         self.formula.nv = one_out.nv
#
#         # copying cardinality constraint
#         for cl in one_out.clauses:
#             cl_dr = map(lambda x: -x + self.orig_vars if abs(x) <= self.orig_vars and x < 0 else x, cl)
#             self.formula.hard.append(cl_dr)
#
#         # add hard clauses filtering out unnecessary combinations
#         if self.options.filter:
#             self.filter_combinations()
#
#         if self.options.pdump:
#             fname = 'primes.{0}@{1}.wcnf'.format(os.getpid(), socket.gethostname())
#             self.formula.to_file(fname)
#
#     def init_solver(self):
#         """
#             Create an initialize a solver for prime enumeration.
#         """
#
#         # initializing prime enumerator
#         if self.options.primer == 'lbx':
#             self.mcsls = LBX(self.formula, use_cld=self.options.use_cld,
#                     solver_name=self.options.solver, use_timer=True)
#         elif self.options.primer == 'mcsls':
#             self.mcsls = MCSls(self.formula, use_cld=self.options.use_cld,
#                     solver_name=self.options.solver, use_timer=True)
#         else:  # sorted or maxsat
#             self.hitman = HitMinimum()
#
#             if self.options.trim:
#                 self.hitman.trim(self.options.trim)
#
#             # hard clauses are modeled as hard
#             for cl in self.formula.hard:
#                 self.hitman.hard(cl)
#
#             # soft clauses
#             for cl in self.formula.soft:
#                 self.hitman.soft(-cl[0])
#
#             # mapping is not really needed when working with Hitman
#             self.id2var = {}
#             for v in xrange(1, self.orig_vars + 1):
#                 self.id2var[v] = v
#                 self.id2var[v + self.orig_vars] = -v
#
#     def filter_combinations(self):
#         """
#             Filter out unnecessary combinations of a given size.
#         """
#
#         if self.options.filter == -1:
#             self.block_noncovers()
#             return
#
#         samps = []
#         for s in self.data.samps:
#             samps.append(set(s))
#
#         filtered = 0
#         for sz in xrange(1, self.options.filter + 1):
#             if sz == 1:
#                 for i, f in enumerate(self.data.feats[:-1]):
#                     if len(f) == 1:
#                         filtered += 1
#                         var = self.data.fvmap.dir[(self.data.names[i], list(f)[0])]
#                         if var > 0:
#                             self.formula.hard.append([-var - self.orig_vars])
#                         else:
#                             self.formula.hard.append([var])
#             else:  # currently a mess, but does the job
#                 for tup in itertools.combinations(xrange(len(self.data.feats[:-1])), sz):
#
#                     to_hit = []
#                     for i in tup:
#                         set_ = []
#
#                         for l in self.data.feats[i]:
#                             v = self.data.fvmap.dir[(self.data.names[i], l)]
#                             set_.extend([v, -v])
#
#                         to_hit.append(set_)
#
#                     # enumerating combinations with hitman
#                     # they are all trivial and of size == sz
#                     # because all sets to hit are disjoint
#                     # with HitMinimum(bootstrap_with=to_hit) as h:
#                     with HitMinimal(bootstrap_with=to_hit) as h:
#                         for hs in h.enum():
#                             hs = set(hs)
#
#                             # we have to check all samples
#                             for s in samps:
#                                 if hs.issubset(s):
#                                     break
#                             else:  # no sample covered by this combination
#                                 filtered += 1
#
#                                 cl = [-l if l > 0 else l - self.orig_vars for l in hs]
#                                 self.formula.hard.append(cl)
#
#         if self.options.verb:
#             #####print('c1 blocked {0} prime combs'.format(filtered))
#
#     def block_noncovers(self):
#         """
#             Add hard constraints to block all non-covering primes.
#         """
#
#         topv = self.formula.nv
#         ncls = len(self.formula.hard)
#         tvars = []  # auxiliary variables
#
#         allv = []
#         for v in xrange(1, self.data.fvars + 1):
#             allv.append(v)
#             allv.append(v + self.orig_vars)
#         allv = set(allv)
#
#         self.tcls = {self.data.fvmap.dir[(self.data.names[-1], c)]: [] for c in self.data.feats[-1]}
#
#         for sample in self.data.samps:
#             s = set([l if l > 0 else -l + self.orig_vars for l in sample[:-1]])
#
#             # computing the complement of the sample
#             compl = allv.difference(s)
#
#             # encoding the complement (as a term) into a set of clauses
#             if compl:
#                 topv += 1
#                 tvars.append(topv)
#                 self.tcls[sample[-1]].append(topv)
#
#                 for l in compl:
#                     self.formula.hard.append([-l, -topv])
#
#                 self.formula.hard.append(list(compl) + [topv])
#
#         if tvars:
#             if self.options.nooverlap:
#                 # at most one class can be covered by each prime
#                 classes = []
#                 for cls in self.tcls.itervalues():
#                     if len(cls) > 1:
#                         topv += 1
#                         classes.append(topv)
#                         for v in cls:
#                             self.formula.hard.append([-v, topv])
#                         self.formula.hard.append(cls + [-topv])
#                     else:
#                         classes.append(cls[0])
#
#                 one_c = CardEnc().atmost(lits=classes, top_id=topv, encoding=self.options.enc)
#                 self.formula.hard.extend(one_c.clauses)
#                 topv = one_c.nv
#
#             if self.options.indprime:
#                 self.tvars = tvars
#             else:
#                 # add final clause forcing to cover at least one sample
#                 self.formula.hard.append(tvars)
#
#             #####print('c1 blocked all non-covering primes')
#             #####print('c1 added more {0} vars and {1} clauses'.format(
#                 topv - self.formula.nv, len(self.formula.hard) - ncls))
#
#             self.formula.nv = topv
#
#     def compute(self):
#         """
#             Enumerate all prime implicants.
#         """
#
#         if self.options.primer in ('lbx', 'mcsls'):
#             return self.compute_mcsls()
#         else:  # sorted or maxsat
#             return self.compute_sorted()
#
#     def compute_mcsls(self):
#         """
#             Call an MCS enumerator.
#         """
#
#         if self.options.verb:
#             #####print('c1 enumerating primes (mcs-based)')
#
#         self.primes = collections.defaultdict(lambda: [])
#
#         if self.options.indprime:
#             self.formula.hard.append([])
#
#             mcses = []
#             for t in self.tvars:
#                 # set sample to cover
#                 self.formula.hard[-1] = [t]
#
#                 # initialize solver from scratch
#                 self.init_solver()
#
#                 # block all previosly computed MCSes
#                 for mcs in mcses:
#                     self.mcsls.block(mcs)
#
#                 # enumerate MCSes covering sample t
#                 for i, mcs in enumerate(self.mcsls.enumerate()):
#                     prime, out = self.process_mcs(mcs)
#
#                     # recording prime implicant
#                     self.primes[out].append(prime)
#
#                     # block
#                     self.mcsls.block(mcs)
#
#                     # record mcs to block later
#                     mcses.append(mcs)
#
#                     if self.options.plimit and i + 1 == self.options.plimit:
#                         break
#
#                 self.mcsls.delete()
#         else:
#             for mcs in self.mcsls.enumerate():
#                 prime, out = self.process_mcs(mcs)
#
#                 # recording prime implicant
#                 self.primes[out].append(prime)
#
#                 # block
#                 self.mcsls.block(mcs)
#
#             self.mcsls.delete()
#
#         # recording time
#         self.stime = resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.init_stime
#         self.ctime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime - self.init_ctime
#         self.time = self.stime + self.ctime
#
#         return self.primes
#
#     def compute_sorted(self):
#         """
#             MaxSAT-based prime implicant enumeration.
#         """
#
#         if self.options.verb:
#             #####print('c1 enumerating primes (maxsat-based)')
#
#         self.primes = collections.defaultdict(lambda: [])
#
#         if self.options.indprime:
#             self.formula.hard.append([])
#
#             mcses = []
#             for t in self.tvars:
#                 # set sample to cover
#                 self.formula.hard[-1] = [t]
#
#                 # initialize solver from scratch
#                 self.init_solver()
#
#                 # block all previosly computed MCSes
#                 for mcs in mcses:
#                     self.hitman.block(mcs)
#
#                 # enumerate MCSes covering sample t
#                 for i, mcs in enumerate(self.hitman.enum()):
#                     prime, out = self.process_mcs(mcs)
#
#                     # recording prime implicant
#                     self.primes[out].append(prime)
#
#                     # record mcs to block later
#                     mcses.append(mcs)
#
#                     if self.options.plimit and i + 1 == self.options.plimit:
#                         break
#
#                 self.hitman.delete()
#         else:
#             for mcs in self.hitman.enum():
#                 prime, out = self.process_mcs(mcs)
#
#                 # recording prime implicant
#                 self.primes[out].append(prime)
#
#             self.hitman.delete()
#
#         # recording time
#         self.stime = resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.init_stime
#         self.ctime = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime - self.init_ctime
#         self.time = self.stime + self.ctime
#
#         return self.primes
#
#     def process_mcs(self, mcs):
#         """
#             Extract a prime implicant from MCS.
#         """
#
#         prime, out = [], None
#
#         for i in mcs:
#             # getting the corresponding variable
#             v_orig = self.id2var[i]
#
#             # filtering out all labels
#             if self.data.nm2id[self.data.fvmap.opp[abs(v_orig)][0]] == len(self.data.names) - 1:
#                 if v_orig > 0:
#                     out = v_orig
#
#                 continue
#
#             prime.append(v_orig)
#
#         # #####printing prime implicant
#         if self.options.verb > 2:
#             premise = []
#
#             for l in prime:
#                 feat, val = self.data.fvmap.opp[abs(l)]
#                 premise.append('{0}\'{1}: {2}\''.format('' if l > 0 else 'not ', feat, val))
#
#             if self.options.verb > 3:
#                 #####print('c1 mcs: {0}'.format(' '.join([str(l) for l in mcs])))
#
#             #####print('c1 prime: {0} => \'{1}: {2}\''.format(', '.join(premise), *self.data.fvmap.opp[out]))
#
#         return prime, out
