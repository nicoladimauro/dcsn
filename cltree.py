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
from utils import check_is_fitted

###############################################################################

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
                    scope, 
                    n_samples, 
                    alpha, 
                    mpriors, 
                    priors, 
                    log_probs, 
                    log_j_probs, 
                    log_c_probs, 
                    cond, 
                    p):
    for i in range(n_features):
        id_i = scope[i]
        prob = (p[i] + alpha*mpriors[id_i,1])/(n_samples + alpha)
        log_probs[i,0] = logr(1-prob)
        log_probs[i,1] = logr(prob)

    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                id_i = scope[i]
                id_j = scope[j]
                log_j_probs[i,j,1,1] = cond[i,j] 
                log_j_probs[i,j,0,1] = cond[j,j] - cond[i,j] 
                log_j_probs[i,j,1,0] = cond[i,i] - cond[i,j] 
                log_j_probs[i,j,0,0] = n_samples - log_j_probs[i,j,1,1] - log_j_probs[i,j,0,1] - log_j_probs[i,j,1,0]

                log_j_probs[i,j,1,1] = logr((log_j_probs[i,j,1,1] + alpha*priors[id_i,id_j,1,1]) / ( n_samples + alpha))
                log_j_probs[i,j,0,1] = logr((log_j_probs[i,j,0,1] + alpha*priors[id_i,id_j,0,1]) / ( n_samples + alpha))
                log_j_probs[i,j,1,0] = logr((log_j_probs[i,j,1,0] + alpha*priors[id_i,id_j,1,0]) / ( n_samples + alpha))
                log_j_probs[i,j,0,0] = logr((log_j_probs[i,j,0,0] + alpha*priors[id_i,id_j,0,0]) / ( n_samples + alpha))

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

###############################################################################

class Cltree:

    def __init__(self):
        self.num_trees = 1
        self.num_edges = 0
        self._forest = False

    def is_forest(self):
        return self._forest

    def fit(self, X, m_priors, j_priors, alpha=1.0, beta = 1.0,
                 sample_weight=None, scope=None, and_leaves=False):
        """Fit the model to the data.

        Parameters
        ----------
        X : ndarray, shape=(n, m)
        The data array.

        m_priors: 
        the marginal priors for each feature
        
        j_priors: 
        the joint priors for each couple of features

        alpha: float, default=1.0
        the constant for the smoothing

        beta: float, default=1.0

        sample_weight: ndarray, shape=(n,)
        The weight of each sample.

        scope: 
        unique identifiers for the features

        and_leaves: boolean, default=False

        """

        self.alpha = alpha
        self.beta = beta
        self.and_leaves = and_leaves
        self.n_features = X.shape[1]

        if scope is None:
            self.scope = np.array([i for i in range(self.n_features)])
        else:
            self.scope = scope

        if sample_weight is None:
            self.n_samples = X.shape[0]
            sample_weight = np.ones(self.n_samples, dtype=np.float32)
        else:
            self.n_samples = np.sum(sample_weight)

        (log_probs, log_j_probs, log_c_probs) = self.compute_log_probs(X, sample_weight, m_priors, j_priors)

        MI = self.cMI(log_probs, log_j_probs)
        " the tree is represented as a sequence of parents"
        mst = minimum_spanning_tree(-(MI))
        dfs_tree = depth_first_order(mst, directed=False, i_start=0)

        self.df_order = dfs_tree[0]
        self.tree = np.zeros(self.n_features, dtype=np.int)
        self.tree[0] = -1
        for p in range(1, self.n_features):
            self.tree[p]=dfs_tree[1][p]
        
        penalization = logr(X.shape[0])/(2*X.shape[0])

        if self.and_leaves == True:
            for p in range(1,self.n_features):
                if MI[self.tree[p],p]<penalization:
                    self.tree[p]=-1
                    self.num_trees = self.num_trees + 1
            if self.num_trees > 1:
                self._forest = True

        self.num_edges = self.n_features - self.num_trees
        # computing the factored represetation
        self.log_factors = np.zeros((self.n_features, 2, 2))
        self.log_factors = compute_log_factors(self.tree, self.n_features, log_probs, log_c_probs, self.log_factors)

    def compute_log_probs(self, X, sample_weight, m_priors, j_priors):
        """ WRITEME """
        log_probs = np.zeros((self.n_features,2))
        log_c_probs = np.zeros((self.n_features,self.n_features,2,2))
        log_j_probs = np.zeros((self.n_features,self.n_features,2,2))

        weighted_X = np.einsum('ij,i->ij', X, sample_weight)

        sparse_cooccurences = sparse.csr_matrix(X)
        cooccurences = sparse_cooccurences.T.dot(weighted_X)
        p = cooccurences.diagonal() 

        return log_probs_numba(self.n_features, 
                               self.scope, 
                               self.n_samples, 
                               self.alpha, 
                               m_priors, 
                               j_priors, 
                               log_probs, 
                               log_j_probs, 
                               log_c_probs, 
                               cooccurences, 
                               p)

    def cMI(self, log_probs, log_j_probs):
        """ WRITEME """
        MI = np.zeros((self.n_features, self.n_features))
        return cMI_numba(self.n_features, log_probs, log_j_probs, MI)

    def score_samples_log_proba(self, X):
        """ WRITEME """
        check_is_fitted(self, "tree")

        Prob = X[:,0]*0.0
        for feature in range(0,self.n_features):
            parent = self.tree[feature]
            if parent == -1:
                Prob = Prob + self.log_factors[feature, X[:,feature],0]
            else:
                Prob = Prob + self.log_factors[feature, X[:,feature], X[:,parent]]
        m = Prob.mean()
        return m

    def score_sample_log_proba(self, x):
        """ WRITEME """
        prob = 0.0
        for feature in range(0,self.n_features):
            parent = self.tree[feature]
            if parent == -1:
                prob = prob + self.log_factors[feature, x[feature], 0]
            else:
                prob = prob + self.log_factors[feature, x[feature], x[parent]]
        return prob


    def score_samples_scope_log_proba(self, X, features):
        """
        In case of a forest, this procedure compute the ll of a single tree of the forest.
        The features parameter is the list of the features of the corresponding tree.
        """
        Prob = X[:,0]*0.0
        for feature in features:
            parent = self.tree[feature]
            if parent == -1:
                Prob = Prob + self.log_factors[feature, X[:,feature], 0]
            else:
                Prob = Prob + self.log_factors[feature, X[:,feature], X[:,parent]]
        m = Prob.mean()
        return m

    def score_sample_scope_log_proba(self, x, features):
        """ WRITEME """
        prob = 0.0
        for feature in features:
            parent = self.tree[feature]
            if parent == -1:
                prob = prob + self.log_factors[feature, x[feature], 0]
            else:
                prob = prob + self.log_factors[feature, x[feature], x[parent]]
        return prob

