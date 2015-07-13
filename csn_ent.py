#!/usr/bin/python

"""
Chow-Liu Trees

Chow, C. K. and Liu, C. N. (1968), Approximating discrete probability distributions with dependence trees,
IEEE Transactions on Information Theory IT-14 (3): 462-467.

"""
import numpy as np
import random
import math
import csv
import sys

import logging
import scipy
import numba
import itertools

from scipy import sparse
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order


@numba.njit
def logr(x):
    if x > 0.0:
        return np.log(x)
    else:
        return -1000.0

DATA_PATH = "data/"


@numba.jit
def log_p(data, alpha):
    features = data.shape[1]
    instances = data.shape[0]
    log_probs = np.zeros((features, 2))
    # p = (data.sum(axis=0) + self.alpha) / (instances + 2.0*self.alpha)
    p = (data.sum(axis=0) + 2.0 * alpha) / (instances + 4.0 * alpha)
    # FIX:         p = (data.sum(axis=0) + 2.0*self.alpha) / (instances +
    # 4.0*self.alpha)
    for i in range(features):
        log_probs[i, 0] = logr(1 - p[i])
        log_probs[i, 1] = logr(p[i])
    return log_probs


@numba.jit
def feature_avg_entropy(log_probs):
    """
    log_probs = [n_features X feature_vas]
    """
    n_features = log_probs.shape[0]
    exp_probs = np.exp(log_probs)
    avg_entropy = - (exp_probs * log_probs).sum() / n_features
    # x_log_x = (exp_probs * log_probs)
    # avg_entropy = - x_log_x.sum() / n_features
    return avg_entropy


def compute_feature_entropy(data, alpha):
    """
    WRITEME
    """
    log_ps = log_p(data, alpha)
    avg_entropy = feature_avg_entropy(log_ps)
    return avg_entropy


def csv_2_numpy(file, path=DATA_PATH, sep=',', type='int'):
    """
    WRITEME
    """
    file_path = path + file
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    dataset = np.array(x).astype(type)
    return dataset


@numba.njit
def cMI_numba(features, log_probs, log_j_probs, MI):
    for i in range(features):
        for j in range(i + 1, features):
            for v0 in range(2):
                for v1 in range(2):
                    MI[i, j] = MI[i, j] + np.exp(log_j_probs[i, j, v0, v1]) * (
                        log_j_probs[i, j, v0, v1] - log_probs[i, v0] - log_probs[j, v1])
                    MI[j, i] = MI[i, j]
    return MI


@numba.njit
def log_p_jp_numba(features, instances, alpha, log_probs, log_j_probs, cond, p):
    for i in range(features):
        log_probs[i, 0] = logr(1 - p[i])
        log_probs[i, 1] = logr(p[i])

    for i in range(features):
        for j in range(i + 1, features):
            log_j_probs[i, j, 1, 1] = cond[i, j]
            log_j_probs[i, j, 0, 1] = cond[j, j] - cond[i, j]
            log_j_probs[i, j, 1, 0] = cond[i, i] - cond[i, j]
            log_j_probs[i, j, 0, 0] = instances - log_j_probs[i, j,
                                                              1, 1] - log_j_probs[i, j, 0, 1] - log_j_probs[i, j, 1, 0]

            log_j_probs[i, j, 1, 1] = logr(
                (log_j_probs[i, j, 1, 1] + alpha) / (instances + 4.0 * alpha))
            log_j_probs[i, j, 0, 1] = logr(
                (log_j_probs[i, j, 0, 1] + alpha) / (instances + 4.0 * alpha))
            log_j_probs[i, j, 1, 0] = logr(
                (log_j_probs[i, j, 1, 0] + alpha) / (instances + 4.0 * alpha))
            log_j_probs[i, j, 0, 0] = logr(
                (log_j_probs[i, j, 0, 0] + alpha) / (instances + 4.0 * alpha))

            log_j_probs[j, i, 1, 1] = log_j_probs[i, j, 1, 1]
            log_j_probs[j, i, 0, 1] = log_j_probs[i, j, 1, 0]
            log_j_probs[j, i, 1, 0] = log_j_probs[i, j, 0, 1]
            log_j_probs[j, i, 0, 0] = log_j_probs[i, j, 0, 0]

    return (log_probs, log_j_probs)


@numba.njit
def log_cp_numba(features, log_probs, log_j_probs, log_c_probs):
    for i in range(features):
        for j in range(features):
            if i != j:
                for v0 in range(2):
                    for v1 in range(2):
                        log_c_probs[i, j, v0, v1] = log_j_probs[
                            i, j, v0, v1] - log_probs[j, v1]

    return log_c_probs


def minimum_spanning_tree_np(X):
    X = X.copy()
    n_vertices = X.shape[0]
    spanning_edges = []

    visited_vertices = [0]
    num_visited = 1
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf

    spanning_trees = []
    st = minimum_spanning_tree_npr(
        X, n_vertices, spanning_edges, visited_vertices, num_visited, spanning_trees)
    for i in st:
        if i not in spanning_trees:
            spanning_trees.append(i)
    return spanning_trees


def minimum_spanning_tree_npr(X, n_vertices, spanning_edges, visited_vertices, num_visited, spanning_trees):

    if num_visited != n_vertices:
        spanning_trees = []
        minimum_value = X[visited_vertices].min()
        new_edges = np.where(X[visited_vertices] == minimum_value)
        if len(new_edges[0]) > (n_vertices - num_visited):
            new_edges = ([new_edges[0][0]], [new_edges[1][0]])
        for e in range(len(new_edges[0])):
            new_edge = (visited_vertices[new_edges[0][e]], new_edges[1][e])
            new_spanning_edges = spanning_edges + [new_edge]
            new_visited_vertices = visited_vertices + [new_edge[1]]

            X_copy = X.copy()
            X_copy[visited_vertices, new_edge[1]] = np.inf
            X_copy[new_edge[1], visited_vertices] = np.inf
            tree = minimum_spanning_tree_npr(
                X_copy, n_vertices, new_spanning_edges, new_visited_vertices, num_visited + 1, spanning_trees)
            spanning_trees = spanning_trees + tree
        return spanning_trees
    else:
        tree = [sorted(spanning_edges)]
        return tree


@numba.njit
def compute_log_factors(tree,
                        n_features,
                        log_probs,
                        log_c_probs,
                        log_factors):

    log_factors[0, 0, 0] = log_probs[0, 0]
    log_factors[0, 0, 1] = log_probs[0, 0]
    log_factors[0, 1, 0] = log_probs[0, 1]
    log_factors[0, 1, 1] = log_probs[0, 1]

    for feature in range(1, n_features):
        parent = int(tree[feature])
        for feature_val in range(2):
            for parent_val in range(2):
                log_factors[feature, feature_val, parent_val] = \
                    log_c_probs[feature, parent, feature_val, parent_val]

    return log_factors


class cltree:

    """

    Parameters
    ----------
    alpha : float, optional (default=1.0)
    Additive (Laplace/Lidstone) smoothing parameter
    (0 for no smoothing).

    """

    " it works only for boolean variables "

    " computing marginal distribution and marginal pair distribution "

    def log_p_jp(self, data, features, instances):
        log_probs = np.zeros((features, 2))
        log_j_probs = np.zeros((features, features, 2, 2))
#        cond = np.dot(data.T.astype(np.int), data.astype(np.int))
# cond0 =
# sparse.csr_matrix(data.T.astype(np.int)).dot(sparse.csr_matrix(data.astype(np.int)))
        sparse_cond = sparse.csr_matrix(data)
        cond0 = sparse_cond.T.dot(sparse_cond)
        cond = np.array(cond0.todense())
        p = (cond.diagonal() + 2 * self.alpha) / (instances + 4 * self.alpha)
        # FIX:         p = (data.sum(axis=0) + 2.0*self.alpha) / (instances +
        # 4.0*self.alpha)
        return log_p_jp_numba(features, instances, self.alpha, log_probs, log_j_probs, cond, p)

    " computing conditional distribution "

    def log_cp(self, features, log_probs, log_j_probs):
        log_c_probs = np.zeros((features, features, 2, 2))
        return log_cp_numba(features, log_probs, log_j_probs, log_c_probs)

    def cMI(self, features, log_probs, log_j_probs):
        MI = np.zeros((self.n_features, self.n_features))
        return cMI_numba(features, log_probs, log_j_probs, MI)

    def __init__(self, data, features_name=None, alpha=1.0):

        self.data = data
        self.alpha = alpha

        self.n_features = data.shape[1]
        if features_name is None:
            self.features = [i for i in range(self.n_features)]
        else:
            self.features = features_name

        self.num_instances = data.shape[0]

        (self.log_probs, self.log_j_probs) = self.log_p_jp(
            self.data, self.n_features, self.num_instances)
        self.log_c_probs = self.log_cp(
            self.n_features, self.log_probs, self.log_j_probs)

        self.MI = self.cMI(self.n_features, self.log_probs, self.log_j_probs)

        " the tree is represented as a sequence of parents"

        mst = minimum_spanning_tree(-(self.MI + 1))
        dfs_tree = depth_first_order(mst, directed=False, i_start=0)

        self.tree = np.zeros(self.n_features)
        self.tree[0] = -1
        for p in range(1, self.n_features):
            self.tree[p] = dfs_tree[1][p]

        # computing the factored represetation
        self.factors = np.zeros((self.n_features, 2, 2))
        self.factors = self.log_factors()

        self.data = None
        self.MI = None
        self.log_j_probs = None
        self.log_probs = None
        self.log_c_probs = None
        mst = None
        dfs_tree = None

#     def loglik(self, data):
#         #        Prob = self.log_probs[0,data[:,0].astype(np.int)]
#         Prob = self.log_probs[0, data[:, 0]]
#         for feature in range(1, self.n_features):
#             parent = self.tree[feature]
# # Prob = Prob + self.log_c_probs[feature, parent,
# # data[:,feature].astype(np.int), data[:,parent].astype(np.int)]
#             Prob = Prob + \
#                 self.log_c_probs[
#                     feature, parent, data[:, feature], data[:, parent]]
#         m = Prob.mean()
#         return m

#     def ll_instance_from_csn(self, x):
#         prob = self.log_probs[0, x[0]]
#         for feature in range(1, self.n_features):
#             parent = self.tree[feature]
#             prob = prob + \
#                 self.log_c_probs[feature, parent, x[feature], x[parent]]
#         return prob
    def log_factors(self):

        return compute_log_factors(self.tree,
                                   self.n_features,
                                   self.log_probs,
                                   self.log_c_probs,
                                   self.factors)

    def ll(self, data):
        Prob = self.factors[0, data[:, 0], 0]
        for feature in range(1, self.n_features):
            parent = self.tree[feature]
            Prob = Prob + \
                self.factors[feature, data[:, feature], data[:, parent]]
        m = Prob.mean()
        return m

    def ll_instance_from_csn(self, x):
        prob = self.factors[0, x[0], 0]
        for feature in range(1, self.n_features):
            parent = self.tree[feature]
            prob = prob + self.factors[feature, x[feature], x[parent]]
        return prob


# prun_dict = {}
# prun_dict['prune'] = 0


class csn_entropy:

    """

    Parameters
    ----------
    alpha : float, optional (default=1.0)
    Additive (Laplace/Lidstone) smoothing parameter
    (0 for no smoothing).

    """

    def __init__(self,
                 data,
                 ID=1,
                 # clt=None,
                 # ll=0.0,
                 avg_entropy=None,
                 features=None,
                 min_instances=10,
                 min_features=3,
                 max_depth=3,
                 alpha=1.0,
                 greedy=1.0,
                 random_forest=False,
                 d=None):

        # parameters:
        self.min_instances = min_instances
        self.min_features = min_features
        self.max_depth = max_depth

        self.alpha = alpha
        self.greedy = greedy

        self.cut_nodes = 0
        self.leaf_nodes = 0

        self.data = data
        self.cltree = None
        self.avg_entropy = None
        self.random_forest = random_forest
        #
        # in case it the the first data block, create a CLTree on it
        # if clt is None:
        # self.cltree = cltree(data, alpha=self.alpha)
        # self.orig_ll = self.cltree.loglik(self.data) = \
        if avg_entropy is None:
            print('First node, computing entropy ')
            self.avg_entropy = \
                compute_feature_entropy(self.data,
                                        # n_features=self.data.shape[1],
                                        # n_instances=data.shape[0],
                                        alpha=self.alpha)
            self.features = [i for i in range(self.data.shape[1])]
            self.d = int(math.sqrt(self.data.shape[1]))

        else:
            self.avg_entropy = avg_entropy
            self.features = features
            self.d = d
        # else:
        # self.cltree = clt
        # self.orig_ll = ll
        # self.avg_entropy

        self.independence = False

        self.root_feature = None
        self.left_child = None
        self.right_child = None

        self.left_weight = 0.0
        self.right_weight = 0.0

        self.id = ID
        print("Block ", self.id, "on features",
              self.features, "ent:", self.avg_entropy)

        if self.data.shape[0] > self.min_instances:
            if self.id < pow(2, self.max_depth) + 1:
                if self.data.shape[1] > self.min_features and self.avg_entropy > 0.01:
                    self.cut()
                else:
                    print(" > no cutting due to few features")
                    self.cltree = cltree(data=self.data, alpha=self.alpha)

                    # check the iid factorization
                    # self.check_independence()
            else:
                print(" > no cutting due to depth limit")
                self.cltree = cltree(data=self.data, alpha=self.alpha)

                # self.check_independence()

        else:
            print(" > no cutting due to few instances")
            self.cltree = cltree(data=self.data, alpha=self.alpha)

            # check the iid factorization
            # self.check_independence()

        if self.root_feature is not None:
            self.cut_nodes = 1 + self.left_child.cut_nodes + \
                self.right_child.cut_nodes
            self.leaf_nodes = self.left_child.leaf_nodes + \
                self.right_child.leaf_nodes
        else:
            self.leaf_nodes = 1

    # def check_independence(self):
    #     iid_ll = self.cltree.iid_loglik(self.data)
    #     if iid_ll > self.orig_ll:
    #         self.independence = True
    # print "--------------------------------------------\nBlock with
    # independent features! cltree ll:", self.orig_ll, "iid ll:", iid_ll

    def show(self):
        self.showl(0)
        print("Cut nodes:", self.cut_nodes)
        print("Leaf nodes:", self.leaf_nodes)

    def showl(self, level):
        print (''.ljust(level), end="")
        print ("id: ", self.id, end=" ")
        if self.root_feature is not None:
            print ("split on", end=" ")
            # print self.cltree.features[self.root_feature],
            self.features[self.root_feature]
            print(self.left_weight, end=" ")
        # print self.cltree.features,
        # print self.cltree.tree
        print(self.features, end=" ")
        if self.root_feature is not None:
            self.left_child.showl(level + 1)
            self.right_child.showl(level + 1)

    def instance_ll(self, x):
        prob = 0.0
        if self.root_feature is None:
            # if self.independence:
            #     prob = prob + self.cltree.iid_ll_instance_from_csn(x)
            # else:
            prob = prob + self.cltree.ll_instance_from_csn(x)
        else:
            x1 = np.concatenate(
                (x[0:self.root_feature], x[self.root_feature + 1:]))
            if x[self.root_feature] == 0:
                prob = prob + logr(self.left_weight) + \
                    self.left_child.instance_ll(x1)
            else:
                prob = prob + logr(self.right_weight) + \
                    self.right_child.instance_ll(x1)
        return prob

    def ll(self, data):
        mean = 0.0
        for x in data:
            prob = self.instance_ll(x)
            mean = mean + prob
        return mean / data.shape[0]

    def cut(self):
        print (" > trying to cut ...", end=" ")
        found = False

        # bestlik = top_ll
        # best_clt_l = None
        # best_clt_r = None
        best_feature_cut = None
        best_left_weight = 0.0
        best_right_weight = 0.0
        best_right_data = None
        best_left_data = None
        best_left_avg_entropy = None
        best_right_avg_entropy = None
        # best_list = []

        best_info_gain = 0.0

        if self.random_forest:
            if self.d > len(self.features):
                selected = range(len(self.features))
            else:
                selected = sorted(random.sample(range(len(self.features)),
                                                self.d))
        else:
            selected = range(len(self.features))

        # for feature in range(self.cltree.n_features):
        # for feature in range(len(self.features)):
        for feature in selected:
            # try to cut on the feature
            print(feature, end=" ")
            sys.stdout.flush()

            condition = self.data[:, feature] == 0
            new_features = np.ones(self.data.shape[1], dtype=bool)
            new_features[feature] = False
            left_data = self.data[condition, :][:, new_features]
            right_data = self.data[~condition, :][:, new_features]
            left_weight = 1.0 * left_data.shape[0] / self.data.shape[0]
            right_weight = 1.0 * right_data.shape[0] / self.data.shape[0]

            info_gain = -np.inf
            right_avg_entropy = np.Inf
            left_avg_entropy = np.Inf

            # avoiding partitions with zero instances
            # TODO: check whether to have cut with single child
            if left_data.shape[0] > 0 and right_data.shape[0] > 0:

                # left_features_name = \

                # right_features_name = \
                #     self.features[0:feature] + self.features[feature + 1:]

                #
                # computing the two entropies
                left_avg_entropy = \
                    compute_feature_entropy(left_data,
                                            # n_features=left_data.shape[1],
                                            # n_instances=left_data.shape[0],
                                            alpha=self.alpha)
                right_avg_entropy = \
                    compute_feature_entropy(right_data,
                                            # n_features=right_data.shape[1],
                                            # n_instances=right_data.shape[0],
                                            alpha=self.alpha)

                split_entropy = (left_weight * left_avg_entropy +
                                 right_weight * right_avg_entropy)
                #
                # compute the information gain in doing this kind of split
                info_gain = self.avg_entropy - split_entropy

                # CL_l = cltree(left_data, left_features_name, alpha=self.alpha)
                # CL_r = cltree(
                #     right_data, right_features_name, alpha=self.alpha)
                # l_ll = CL_l.loglik(left_data)
                # r_ll = CL_r.loglik(right_data)
                # ll = ((l_ll + logr(left_weight)) * left_data.shape[0] + (
                # r_ll + logr(right_weight)) * right_data.shape[0]) /
                # self.data.shape[0]
            # else:
            #     ll = -np.inf

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                found = True
                best_feature_cut = feature
                best_left_weight = left_weight
                best_right_weight = right_weight
                best_right_data = right_data
                best_left_data = left_data
                best_left_avg_entropy = left_avg_entropy
                best_right_avg_entropy = right_avg_entropy

            # if ll > bestlik:
            #     bestlik = ll
            #     best_clt_l = CL_l
            #     best_clt_r = CL_r
            #     best_feature_cut = feature
            #     best_left_weight = left_weight
            #     best_right_weight = right_weight
            #     best_right_data = right_data
            #     best_left_data = left_data
            #     best_l_ll = l_ll
            #     best_r_ll = r_ll
            #     found = True
            # best_list.append((bestlik, best_clt_l, best_clt_r,
            # best_feature_cut, best_left_weight,
            # best_right_weight, best_right_data, best_left_data, best_l_ll,
            # best_r_ll))

        if found:
            # best_list.sort(key=lambda bestll: bestll[0])
            # choosen=int((len(best_list) - 1) * self.greedy)
            # (bestlik, best_clt_l, best_clt_r, best_feature_cut,
            # best_left_weight, best_right_weight,
            # best_right_data, best_left_data, best_l_ll,
            # best_r_ll)=best_list[choosen]

            features_name =\
                self.features[0:best_feature_cut] + \
                self.features[best_feature_cut + 1:]

            self.root_feature = best_feature_cut
            print ("cutting on feature ", self.root_feature)

            self.left_child = csn_entropy(best_left_data, self.id * 2,
                                          # best_clt_l, best_l_ll,
                                          avg_entropy=best_left_avg_entropy,
                                          features=features_name,
                                          max_depth=self.max_depth,
                                          min_instances=self.min_instances,
                                          min_features=self.min_features,
                                          alpha=self.alpha, greedy=self.greedy,
                                          random_forest=self.random_forest,
                                          d=self.d)
            self.right_child = csn_entropy(best_right_data, self.id * 2 + 1,
                                           # best_clt_r, best_r_ll,
                                           avg_entropy=best_right_avg_entropy,
                                           features=features_name,
                                           max_depth=self.max_depth,
                                           min_instances=self.min_instances,
                                           min_features=self.min_features,
                                           alpha=self.alpha, greedy=self.greedy,
                                           random_forest=self.random_forest,
                                           d=self.d)
            self.left_weight = best_left_weight
            self.right_weight = best_right_weight
        else:
            print (">>>> no cutting, putting a CL tree")
            # check the iid factorization
            # self.check_independence()

            #
            # create cl tree here
            self.cltree = cltree(data=self.data, alpha=self.alpha)
            print(self.cltree)

    def prune(self, valid):
        """
        Prune on valid data checking the ll
        """

        print ('Checking node', self.id, '(on feature)', self.root_feature)

        DEFAULT_VALID_LL = -1000
        #
        # if it is a leaf return
        if self.root_feature is None:

            print ('it is a leaf', valid.shape[0], valid.shape[1])

            # self.leaf_nodes = 1
            # self.cut_nodes = 0

            if valid.shape[0] > 0:
                # print ('greater than 0')
                return self.cltree.ll(valid)
            else:
                #
                # here I am saying I do not know what to do
                # print ('defaulting')
                return DEFAULT_VALID_LL
        else:
            print ('it is not a leaf, checking children')
            #
            # calling it on the childrens on splitted version of the valid
            condition = valid[:, self.root_feature] == 0
            new_features = np.ones(valid.shape[1], dtype=bool)
            new_features[self.root_feature] = False
            valid_left = valid[condition, :][:, new_features]
            valid_right = valid[~condition, :][:, new_features]

            left_pruned_ll = self.left_child.prune(valid_left)
            right_pruned_ll = self.right_child.prune(valid_right)

            split_ll = DEFAULT_VALID_LL
            no_split_ll = DEFAULT_VALID_LL

            #
            # computing a cltree on this node
            curr_cl_tree = cltree(data=self.data, alpha=self.alpha)

            if valid.shape[0] > 0:

                #
                # I evaluate it on the validation set
                no_split_ll = curr_cl_tree.ll(valid)
                #
                # check if it is worth it splitting
                # left_valid_ll = self.left_child.cltree.loglik(valid_left)
                # right_valid_ll = self.right_child.cltree.loglik(valid_right)
            #    split_ll = DEFAULT_VALID_LL

            # if valid.shape[0] > 0:
                # if valid_left.shape[0] > 0 and valid_right.shape[0] > 0:
                # print('x', valid_left.shape[0], valid_right.shape[0])
                split_ll = (((left_pruned_ll +
                              logr(self.left_weight)) * valid_left.shape[0] +
                             (right_pruned_ll +
                              logr(self.right_weight)) * valid_right.shape[0])
                            / valid.shape[0])

            #
            # any improvement?
            if no_split_ll < split_ll:
                print ('leaving it as it is', no_split_ll, split_ll)

                self.leaf_nodes = (self.left_child.leaf_nodes +
                                   self.right_child.leaf_nodes)
                self.cut_nodes = (1 + self.left_child.cut_nodes +
                                  self.right_child.cut_nodes)
                # print('prune', prun_dict['prune'])
                return split_ll
            else:
                print ('pruning ', no_split_ll, split_ll)
                # prun_dict['prune'] += (1 + self.left_child.cut_nodes +
                #                        self.right_child.cut_nodes)
                # print('prune', prun_dict['prune'], self.cut_nodes)

                #
                # making it a leaf node
                self.root_feature = None
                self.cltree = curr_cl_tree
                self.left_child = None
                self.right_child = None
                self.left_weight = 0.0
                self.right_weight = 0.0

                self.leaf_nodes = 1
                self.cut_nodes = 0

                return no_split_ll


class csnm_entropy:

    """

    Parameters
    ----------

    """

    def __init__(self,
                 training_data,
                 max_components=1,
                 p=1.0,
                 min_instances=10,
                 min_features=3,
                 max_depth=3,
                 alpha=1.0,
                 random_forest=False,
                 prune=False,
                 valid_data=None):

        # parameters:
        self.max_components = max_components

        self.training_data = training_data
        # self.min_instances = int(self.training_data.shape[0]*min_instances/100)
        self.min_instances = min_instances
        self.min_features = min_features
        self.max_depth = max_depth

        self.prune = prune
        self.valid = valid_data

        #
        # this shall be included?
        # self.alpha = int(self.training_data.shape[0] * alpha / 100)
        self.alpha = alpha
        logging.info("Setting alpha to %d", self.alpha)

        self.p = p
        self.random_forest = random_forest

        self.bags = [None] * self.max_components
        self.csns = [None] * self.max_components
        self.weights = [1 / self.max_components] * self.max_components
        self.lls = [0.0] * self.max_components

        self.create_bags()
        self.learn()

    def create_bags(self):
        if self.max_components == 1:
            self.bags[0] = self.training_data
        else:
            for i in range(self.max_components):
                self.bags[i] = self.create_bag()

    def create_bag(self):
        n_instances = int(self.p * self.training_data.shape[0])
        bag = np.zeros((n_instances, self.training_data.shape[1]), dtype='int')
        for i in range(n_instances):
            choice = random.randint(0, self.training_data.shape[0] - 1)
            bag[i] = self.training_data[choice]
        return bag

    def learn(self):
        for i in range(self.max_components):
            C = csn_entropy(data=self.bags[i],
                            max_depth=self.max_depth,
                            min_instances=self.min_instances,
                            min_features=self.min_features,
                            alpha=self.alpha,
                            random_forest=self.random_forest)
            if self.prune:
                C.prune(self.valid)

            self.csns[i] = C

            self.lls[i] = self.csns[i].ll(self.training_data)
            print ("Cut nodes: ", self.csns[i].cut_nodes)

    def compute_weights(self, n_c):
        sum_ll = 0.0
        for i in range(n_c):
            sum_ll += self.lls[i]
        for i in range(n_c):
            self.weights[i] = self.lls[i] / sum_ll
        print (self.weights)

    def ll(self, data, n_c, out_filename):
        with open(out_filename, 'w') as out_log:
            self.compute_weights(n_c)
            mean = 0.0
            for x in data:
                prob = 0.0
                for k in range(n_c):
                    prob = prob + \
                        np.exp(self.csns[k].instance_ll(x)) * self.weights[k]
                mean = mean + logr(prob)
                out_log.write('%.10f\n' % logr(prob))
        out_log.close()
        return mean / data.shape[0]

    def nodes(self, n_c):
        mean = 0.0
        for i in range(n_c):
            mean = mean + self.csns[i].cut_nodes
        return mean / n_c

    def check_correctness(self):
        mean = 0.0
        for world in itertools.product([0, 1], repeat=self.training_data.shape[1]):
            prob = 0.0
            for k in range(self.n_components):
                prob = prob + np.exp(self.csns[k].instance_ll(world))
            prob = prob / self.n_components
            mean = mean + prob
        return mean
