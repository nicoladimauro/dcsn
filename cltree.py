#!/usr/bin/python3

"""
Chow-Liu Trees

Chow, C. K. and Liu, C. N. (1968), Approximating discrete probability distributions with dependence trees, 
IEEE Transactions on Information Theory IT-14 (3): 462-467. 

"""

import numpy as np
import numba
from scipy import sparse
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order

from logr import logr

@numba.njit
def cMI_numba(n_features, 
              log_probs, 
              log_j_probs, 
              MI):
    for i in range(n_features):
        for j in range(i+1,n_features):
            for v0 in range(2):
                for v1 in range(2):
                    MI[i,j] = MI[i,j] + np.exp(log_j_probs[i,j,v0,v1])*( log_j_probs[i,j,v0,v1] - log_probs[i,v0] - log_probs[j,v1])
                    MI[j,i] = MI[i,j]
    return MI

@numba.njit
def log_probs_numba(n_features, 
                    features_id, 
                    n_instances, 
                    alpha, 
                    mpriors, 
                    priors, 
                    log_probs, 
                    log_j_probs, 
                    log_c_probs, 
                    cond, 
                    p):
    for i in range(n_features):
        id_i = features_id[i]
        prob = (p[i] + alpha*mpriors[id_i,1])/(n_instances + alpha)
        log_probs[i,0] = logr(1-prob)
        log_probs[i,1] = logr(prob)

    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                id_i = features_id[i]
                id_j = features_id[j]
                log_j_probs[i,j,1,1] = cond[i,j] 
                log_j_probs[i,j,0,1] = cond[j,j] - cond[i,j] 
                log_j_probs[i,j,1,0] = cond[i,i] - cond[i,j] 
                log_j_probs[i,j,0,0] = n_instances - log_j_probs[i,j,1,1] - log_j_probs[i,j,0,1] - log_j_probs[i,j,1,0]

                log_j_probs[i,j,1,1] = logr((log_j_probs[i,j,1,1] + alpha*priors[id_i,id_j,1,1]) / ( n_instances + alpha))
                log_j_probs[i,j,0,1] = logr((log_j_probs[i,j,0,1] + alpha*priors[id_i,id_j,0,1]) / ( n_instances + alpha))
                log_j_probs[i,j,1,0] = logr((log_j_probs[i,j,1,0] + alpha*priors[id_i,id_j,1,0]) / ( n_instances + alpha))
                log_j_probs[i,j,0,0] = logr((log_j_probs[i,j,0,0] + alpha*priors[id_i,id_j,0,0]) / ( n_instances + alpha))

                log_c_probs[i,j,1,1] = log_j_probs[i,j,1,1] - log_probs[j,1]
                log_c_probs[i,j,0,1] = log_j_probs[i,j,0,1] - log_probs[j,1]
                log_c_probs[i,j,1,0] = log_j_probs[i,j,1,0] - log_probs[j,0]
                log_c_probs[i,j,0,0] = log_j_probs[i,j,0,0] - log_probs[j,0]

    return (log_probs, log_j_probs, log_c_probs)


@numba.njit
def compute_log_factors(tree,
                        n_features,
                        log_probs,
                        log_c_probs,
                        log_factors):

    for feature in range(0,n_features):
        if tree[feature]==-1:
            log_factors[feature, 0, 0] = log_probs[feature, 0]
            log_factors[feature, 0, 1] = log_probs[feature, 0]
            log_factors[feature, 1, 0] = log_probs[feature, 1]
            log_factors[feature, 1, 1] = log_probs[feature, 1]
        else:
            parent = int(tree[feature])
            for feature_val in range(2):
                for parent_val in range(2):
                    log_factors[feature, feature_val, parent_val] = log_c_probs[feature, parent, feature_val, parent_val]

    return log_factors


class Cltree:
    """

    Args:
    
    data: a numpy matrix of 0s and 1s (the data)
    m_priors: the marginal priors for each feature
    j_priors: the joint priors for each couple of features
    alpha: the constant for the smoothing
    features_id (int): unique identifiers for the features
    is_root (bool): indicates whether the cltree corresponds to the root node of the csn

    Attributes:

    """

    " it works only for boolean variables "


    def __init__(self, 
                 data, 
                 m_priors, 
                 j_priors, 
                 alpha=1.0, 
                 beta = 1.0,
                 features_id=None, 
                 and_leaves=False,
                 is_root=False):


        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.m_priors = m_priors
        self.j_priors = j_priors
        self.is_root = is_root
        self.num_trees = 1
        self.num_edges = 0
        self._forest = False

        self.and_leaves = and_leaves


        self.n_features = data.shape[1]
        if features_id is None:
            self.features_id = np.array([i for i in range(self.n_features)])
        else:
            self.features_id = features_id


        self.num_instances = data.shape[0]

        (self.log_probs, self.log_j_probs, self.log_c_probs) = self.compute_log_probs(self.data, self.n_features, self.num_instances)


        self.MI = self.cMI(self.log_probs, self.log_j_probs)

        " the tree is represented as a sequence of parents"

        mst = minimum_spanning_tree(-(self.MI))
        dfs_tree = depth_first_order(mst, directed=False, i_start=0)

        self.df_order = dfs_tree[0]
        self.tree = np.zeros(self.n_features, dtype=np.int)
        self.tree[0] = -1
        for p in range(1, self.n_features):
            self.tree[p]=dfs_tree[1][p]
        
#        penalization = np.inf
        penalization = logr(self.data.shape[0])/(2*self.data.shape[0])
#        penalization = self.beta

        """
        min_v = np.inf
        min_p = -1
        for p in range(1,self.n_features):
            if self.MI[self.tree[p],p] < min_v:
                min_v = self.MI[self.tree[p],p]
                min_p = p

        if self.and_leaves == True:
            if self.MI[self.tree[min_p],min_p] < penalization:
                    self.tree[min_p]=-1
                    self.num_trees = self.num_trees + 1
            if self.num_trees > 1:
                self._forest = True
        """


        if self.and_leaves == True:
            for p in range(1,self.n_features):
                if self.MI[self.tree[p],p]<penalization:
                    self.tree[p]=-1
                    self.num_trees = self.num_trees + 1
            if self.num_trees > 1:
                self._forest = True


        """
        if self.and_leaves == True:
            # check for a forest
            eps = pow(self.data.shape[0],-self.beta)
            for p in range(1,self.n_features):
                if self.MI[self.tree[p],p]<eps:
                    self.tree[p]=-1
                    self.num_trees = self.num_trees + 1
            if self.num_trees > 1:
                self._forest = True
        """

        self.num_edges = self.n_features - self.num_trees
        # computing the factored represetation
        self.factors = np.zeros((self.n_features, 2, 2))
        self.factors = self.log_factors()

        self.data = None
        self.MI = None
        self.log_j_probs = None
        self.log_probs = None
        self.log_c_probs = None
        self.m_priors = None
        self.j_priors = None
        mst = None
        dfs_tree = None

    def log_factors(self):

        return compute_log_factors(self.tree,
                                   self.n_features,
                                   self.log_probs,
                                   self.log_c_probs,
                                   self.factors)

    def compute_log_probs(self, 
                          data, 
                          n_features, 
                          n_instances):

        log_probs = np.zeros((n_features,2))
        log_c_probs = np.zeros((n_features,n_features,2,2))
        log_j_probs = np.zeros((n_features,n_features,2,2))

        sparse_cooccurence = sparse.csr_matrix(data)
        cooccurence0 = sparse_cooccurence.T.dot(sparse_cooccurence)
        cooccurence = np.array(cooccurence0.todense())

        p = cooccurence.diagonal() 

        return log_probs_numba(n_features, 
                               self.features_id, 
                               n_instances, 
                               self.alpha, 
                               self.m_priors, 
                               self.j_priors, 
                               log_probs, 
                               log_j_probs, 
                               log_c_probs, 
                               cooccurence, 
                               p)

    def cMI(self, 
            log_probs, 
            log_j_probs):

        MI = np.zeros((self.n_features, self.n_features))
        return cMI_numba(self.n_features, log_probs, log_j_probs, MI)

    def ll(self, 
           data):

        Prob = data[:,0]*0.0
        for feature in range(0,self.n_features):
            parent = self.tree[feature]
            if parent == -1:
                Prob = Prob + self.factors[feature,data[:,feature],0]
            else:
                Prob = Prob + self.factors[feature, data[:,feature], data[:,parent]]
        m = Prob.mean()
        return m

    def ll_instance_from_csn(self, 
                             x):
        prob = 0.0
        for feature in range(0,self.n_features):
            parent = self.tree[feature]
            if parent == -1:
                prob = prob + self.factors[feature,x[feature],0]
            else:
                prob = prob + self.factors[feature, x[feature], x[parent]]
        return prob


    """
    In case of a forest, this procedure compute the ll of a single tree of the forest.
    The features parameter is the list of the features of the corresponding tree.
    """
    def sub_ll(self, 
               data,
               features):

        Prob = data[:,0]*0.0
        for feature in features:
            parent = self.tree[feature]
            if parent == -1:
                Prob = Prob + self.factors[feature,data[:,feature],0]
            else:
                Prob = Prob + self.factors[feature, data[:,feature], data[:,parent]]
        m = Prob.mean()
        return m

    def sub_ll_instance_from_csn(self, 
                                 x,
                                 features):
        prob = 0.0
        for feature in features:
            parent = self.tree[feature]
            if parent == -1:
                prob = prob + self.factors[feature,x[feature],0]
            else:
                prob = prob + self.factors[feature, x[feature], x[parent]]
        return prob


