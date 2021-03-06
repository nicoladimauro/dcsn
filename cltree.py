"""
Tree Bayesian Networks: a probability distribution factored according to a tree 
whose structure is learned using the Chow-Liu algorithm

Chow, C. K. and Liu, C. N. (1968), Approximating discrete probability distributions 
with dependence trees, IEEE Transactions on Information Theory IT-14 (3): 462-467. 

"""

import numpy as np
import numba
from scipy import sparse
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order
from logr import logr
from utils import check_is_fitted
import itertools

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
                    cond, 
                    p):
    for i in range(n_features):
        id_i = scope[i]
        prob = (p[i] + alpha*mpriors[id_i,1])/(n_samples + alpha)
        log_probs[i,0] = logr(1-prob)
        log_probs[i,1] = logr(prob)

    for i in range(n_features):
        for j in range(n_features):
            id_i = scope[i]
            id_j = scope[j]

            log_j_probs[i,j,1,1] = logr((cond[i,j] + alpha*priors[id_i,id_j,1,1]) / ( n_samples + alpha))
            log_j_probs[i,j,0,1] = logr((cond[j,j] - cond[i,j] + alpha*priors[id_i,id_j,0,1]) / ( n_samples + alpha))
            log_j_probs[i,j,1,0] = logr((cond[i,i] - cond[i,j] + alpha*priors[id_i,id_j,1,0]) / ( n_samples + alpha))
            log_j_probs[i,j,0,0] = logr((n_samples - cond[j,j] - cond[i,i] + cond[i,j] + alpha*priors[id_i,id_j,0,0]) / ( n_samples + alpha))

            log_j_probs[j,i,1,1] = log_j_probs[i,j,1,1]
            log_j_probs[j,i,1,0] = log_j_probs[i,j,0,1]
            log_j_probs[j,i,0,1] = log_j_probs[i,j,1,0]
            log_j_probs[j,i,0,0] = log_j_probs[i,j,0,0]

    return (log_probs, log_j_probs)


@numba.njit
def compute_log_factors(tree,
                        n_features,
                        log_probs,
                        log_j_probs,
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
                    log_factors[feature, feature_val, parent_val] = log_j_probs[feature,parent,feature_val, parent_val] - log_probs[parent, parent_val] 

    return log_factors

###############################################################################

@numba.njit
def compute_cooccurences_numba(X, C, NZ, r, c):
    for k in range(r):
        non_zeros = 0
        for i in range(c):
            if X[k,i]:
                NZ[non_zeros]=i
                non_zeros += 1
                for j in range(non_zeros):
                    v = NZ[j]
                    C[v,i] += 1
    for i in range(1,c):
        for j in range(i):
            C[i,j] = C[j,i]



class Cltree:

    def __init__(self):
        self.num_trees = 1
        self.num_edges = 0
        self._forest = False

    def is_forest(self):
        return self._forest

    def fit(self, X, m_priors, j_priors, alpha=1.0, sample_weight=None, scope=None, and_leaves=False, multilabel = False, n_labels=0, ml_tree_structure=0):
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

        sample_weight: ndarray, shape=(n,)
        The weight of each sample.

        scope: 
        unique identifiers for the features

        and_leaves: boolean, default=False

        multilabel: boolean, default=False
        its value indicates whether the cltree are used for multilabel classification 
        problems when imported by mlcsn.py

        n_labels: integer, default=0
        in case of multilabel classification problem indicates the number of labels,
        assumed to be the n_labels rows of X

        ml_tree_structure: integer, default=0
        in case of multilabel classification problem indicates the structure of the tree 
        to be learned. The set of features F corresponds to the union of A (the attributes)
        and Y (the labels):
        - 0, no constraint on the resulting tree
        - 1, the parent of each variable in Y must have the parent in Y, while the parent of each
        variable in A can have the parent in A or in Y. A label variable depends on a label 
        variable; an attribute variable can depend on a label variable or on an attribute variable
        - 2, the parent of each variable in Y must have the parent in Y, and the parent of each
        variable in A can have the parent in Y. A label variable depends on a label variable; an 
        attribute variable depends on a label variable
        
        """


        self.alpha = alpha
        self.and_leaves = and_leaves
        self.n_features = X.shape[1]

        rootTree = False
        if scope is None:
            self.scope = np.array([i for i in range(self.n_features)])
            rootTree = True
        else:
            self.scope = scope

        if sample_weight is None:
            self.n_samples = X.shape[0]
        else:
            self.n_samples = np.sum(sample_weight)


        (log_probs, log_j_probs) = self.compute_log_probs(X, sample_weight, m_priors, j_priors)


        MI = self.cMI(log_probs, log_j_probs)


        if multilabel == True:
            if ml_tree_structure == 1:
                MI[-n_labels:,-n_labels:] += np.max(MI)
            elif ml_tree_structure == 2:
                MI[-n_labels:,-n_labels:] += np.max(MI)
                MI[:-n_labels,:-n_labels] = 0
            elif ml_tree_structure == 3:
                MI[:-n_labels,:-n_labels] = 0
        
        " the tree is represented as a sequence of parents"

        mst = minimum_spanning_tree(-(MI))
        dfs_tree = depth_first_order(mst, directed=False, i_start=0)

        self.df_order = dfs_tree[0]
        self.post_order = dfs_tree[0][::-1]
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

        """
        selected_MI = []
        for p in range(1,self.n_features):
            selected_MI.append((p,MI[self.tree[p],p]))
        selected_MI.sort(key=lambda mi: mi[1], reverse=True)
        for p in range(10,self.n_features-1):
            self.tree[selected_MI[p][0]]=-1
        """

        if multilabel == True and rootTree:
            pX = 0
            for i in range(self.n_features-n_labels):
                if self.tree[i]>=(self.n_features-n_labels):
                    pX += 1
            pY = 0
            for i in range(self.n_features-n_labels,self.n_features):
                if self.tree[i]>=(self.n_features-n_labels):
                    pY += 1
                    
            print("Xs with Y parent: ", pX)
            print("Ys with Y parent: ", pY)            

        self.num_edges = self.n_features - self.num_trees
        # computing the factored represetation
        self.log_factors = np.zeros((self.n_features, 2, 2))
        self.log_factors = compute_log_factors(self.tree, self.n_features, log_probs, log_j_probs, self.log_factors)




    def compute_log_probs(self, X, sample_weight, m_priors, j_priors):
        """ WRITEME """
        log_probs = np.zeros((self.n_features,2))
        log_j_probs = np.zeros((self.n_features,self.n_features,2,2))

        cooccurences = np.zeros((X.shape[1], X.shape[1]))
        NZ = np.zeros(X.shape[1],dtype='int')

        compute_cooccurences_numba(X, cooccurences, NZ, X.shape[0], X.shape[1])
        
        """
        sparse_cooccurences = sparse.csr_matrix(X)
        if sample_weight is None:
            cooccurences_ = sparse_cooccurences.T.dot(sparse_cooccurences)
            cooccurences = np.array(cooccurences_.todense())
        else:
            weighted_X = np.einsum('ij,i->ij', X, sample_weight)
            cooccurences = sparse_cooccurences.T.dot(weighted_X)
        """
        p = cooccurences.diagonal() 

        return log_probs_numba(self.n_features, 
                               self.scope, 
                               self.n_samples, 
                               self.alpha, 
                               m_priors, 
                               j_priors, 
                               log_probs, 
                               log_j_probs, 
                               cooccurences, 
                               p)




    def cMI(self, log_probs, log_j_probs):
        """ WRITEME """
        MI = np.zeros((self.n_features, self.n_features))
        return cMI_numba(self.n_features, log_probs, log_j_probs, MI)





    def score_samples_log_proba(self, X, sample_weight = None):
        """ WRITEME """
        check_is_fitted(self, "tree")

        Prob = X[:,0]*0.0
        for feature in range(0,self.n_features):
            parent = self.tree[feature]
            if parent == -1:
                Prob = Prob + self.log_factors[feature, X[:,feature],0]
            else:
                Prob = Prob + self.log_factors[feature, X[:,feature], X[:,parent]]

        if sample_weight is None:
            m = Prob.mean()
        else:
            Prob = sample_weight * Prob
            m = np.sum(Prob) / np.sum(sample_weight)
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


    def score_samples_scope_log_proba(self, X, features, sample_weight=None):
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

        if sample_weight is None:
            m = Prob.mean()
        else:
            Prob = sample_weight * Prob
            m = np.sum(Prob) / np.sum(sample_weight)

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

    def mpe(self, evidence = {}):

        messages = np.zeros((self.n_features, 2))
        states = [ [0,0] for i in range(self.n_features) ] 
        MAP = {}

        for i in self.post_order:
            if i != 0:
                state_evidence = evidence.get(self.scope[i])
                if state_evidence != None:
                    states[i][0] = state_evidence
                    states[i][1] = state_evidence
                    messages[self.tree[i],0]+= self.log_factors[i,state_evidence,0]+messages[i,state_evidence]
                    messages[self.tree[i],1]+= self.log_factors[i,state_evidence,1]+messages[i,state_evidence]
                else:
                    state_evidence_parent = evidence.get(self.scope[self.tree[i]])
                    if state_evidence_parent != None:
                        if (self.log_factors[i,0,state_evidence_parent] +messages[i,0] > self.log_factors[i,1,state_evidence_parent] + messages[i,1]):
                            states[i][state_evidence_parent] = 0
                            messages[self.tree[i],state_evidence_parent]+= self.log_factors[i,0,state_evidence_parent]+messages[i,0]
                        else:
                            states[i][state_evidence_parent] = 1
                            messages[self.tree[i],state_evidence_parent]+= self.log_factors[i,1,state_evidence_parent]+messages[i,1]
                    else:
                        for parent in range(2):
                            if (self.log_factors[i,0,parent]+messages[i,0] > self.log_factors[i,1,parent]+messages[i,1]):
                                states[i][parent] = 0
                                messages[self.tree[i],parent]+= self.log_factors[i,0,parent]+messages[i,0]

                            else:
                                states[i][parent] = 1
                                messages[self.tree[i],parent]+= self.log_factors[i,1,parent]+messages[i,1]
        logprob = 0.0
        for i in self.df_order:
            if self.tree[i]==-1:
                state_evidence = evidence.get(i)
                if state_evidence != None:
                    MAP[self.scope[i]] = state_evidence
                    logprob += self.log_factors[i,MAP[self.scope[i]],0]
                else:
                    if self.log_factors[i,0,0]+messages[i,0]>self.log_factors[i,1,0]+messages[i,1]:
                        MAP[self.scope[i]] = 0
                    else:
                        MAP[self.scope[i]] = 1
                    logprob += self.log_factors[i,MAP[self.scope[i]],0]
            else:
                MAP[self.scope[i]] = states[i][MAP[self.scope[self.tree[i]]]]
                logprob += self.log_factors[i,MAP[self.scope[i]],MAP[self.scope[self.tree[i]]]]
        return (MAP, logprob)


    def marginal_inference(self, evidence = {}):

        messages = np.zeros((self.n_features, 2))
        logprob = 0.0
        for i in self.post_order:
            if i != 0:
                state_evidence = evidence.get(self.scope[i])
                if state_evidence != None:
                    messages[self.tree[i],0] += self.log_factors[i,state_evidence,0] + messages[i,state_evidence]
                    messages[self.tree[i],1] += self.log_factors[i,state_evidence,1] + messages[i,state_evidence]
                else:
                    # marginalization
                    messages[self.tree[i], 0] += logr(np.exp(self.log_factors[i, 0, 0] + messages[i,0]) + np.exp(self.log_factors[i, 1, 0] + messages[i,1]))
                    messages[self.tree[i], 1] += logr(np.exp(self.log_factors[i, 0, 1] + messages[i,0]) + np.exp(self.log_factors[i, 1, 1] + messages[i,1]))
            else:
                state_evidence = evidence.get(self.scope[i])
                if state_evidence != None:
                    logprob = self.log_factors[i,state_evidence,0] + messages[0,state_evidence]
                else:
                    # marginalization
                    logprob = logr(np.exp(self.log_factors[i,0,0]+messages[0,0])+np.exp(self.log_factors[i,1,0]+messages[0,1]))
        return logprob


    def naiveMPE(self, evidence = {}):
        maxprob = -np.inf
        maxstate = []

        worlds = list(itertools.product([0, 1], repeat=self.n_features))

        for w in worlds:
            ver = True
            for var, state in evidence.items():
                if w[var] != state:
                    ver = False
                    break

            if ver:
                prob = self.log_factors[0, w[0], 0]
                for i in range(1,self.n_features):
                    prob = prob + self.log_factors[i, w[i], w[self.tree[i]]]
                if prob > maxprob:
                    maxprob = prob
                    maxstate = w
        return (maxstate, maxprob)


    def naive_marginal(self, evidence = {}):
        probm = 0.0
        
        M = {}
        for i in range(self.n_features):
            if evidence.get(i) == None:
                M[i] = [0,1]

        A = [dict(zip(M,prod)) for prod in itertools.product(*(M[param] for param in M))]

        for D in A:
            D.update(evidence)
            prob = self.log_factors[0, D[0], 0]
            for i in range(1,self.n_features):
                prob = prob + self.log_factors[i, D[i], D[self.tree[i]]]
            probm += np.exp(prob)
        return logr(probm)


"""
C = Cltree()
X = np.random.choice([0,1], size=(2000,15))
m_priors = np.ones((15,2))/2
j_priors = np.ones((15,15,2,2))/4
C.fit(X, m_priors, j_priors)
print (C.mpe())
print(C.naiveMPE())

evidence = {}
evidence[2]=0
evidence[3]=0
print(np.exp(C.marginal_inference(evidence=evidence)))
print(np.exp(C.naive_marginal(evidence=evidence)))
evidence[2]=0
evidence[3]=1
print(np.exp(C.marginal_inference(evidence=evidence)))
print(np.exp(C.naive_marginal(evidence=evidence)))
evidence[2]=1
evidence[3]=0
print(np.exp(C.marginal_inference(evidence=evidence)))
print(np.exp(C.naive_marginal(evidence=evidence)))
evidence[2]=1
evidence[3]=1
print(np.exp(C.marginal_inference(evidence=evidence)))
print(np.exp(C.naive_marginal(evidence=evidence)))
"""
