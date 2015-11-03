import numpy as np
from scipy import sparse
import math 
import logging
import sys
import itertools 
import random

from nodes import Node, OrNode, AndNode, TreeNode, is_or_node, is_and_node, is_tree_node
from logr import logr
from cltree import Cltree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################

class Csn:

    _id_node_counter = 1
    _or_nodes = 0
    _leaf_nodes = 0
    _or_edges = 0
    _clt_edges = 0
    _and_nodes = 0
    _cltrees = 0
    _clforests = 0
    _depth = 0
    _mean_depth = 0

    @classmethod
    def init_stats(cls):
        Csn._id_node_counter = 1
        Csn._or_nodes = 0
        Csn._leaf_nodes = 0
        Csn._or_edges = 0
        Csn._clt_edges = 0
        Csn._and_nodes = 0
        Csn._cltrees = 0
        Csn._clforests = 0
        Csn._depth = 0
        Csn._mean_depth = 0
    
    def __init__(self, data, clt = None, ll = 0.0,  min_instances = 5, min_features = 3, 
                 alpha = 1.0, beta = 1.0, d = None, n_original_samples = None,
                 random_forest = False, m_priors = None, j_priors = None, 
                 and_leaves=False, and_inners=False, min_gain = None, depth = 1,
                 sample_weight=None):

        self.min_instances = min_instances
        self.min_features = min_features
        self.and_leaves = and_leaves
        self.and_inners = and_inners
        self.alpha = alpha
        self.beta = beta
        self.depth = depth
        self.data = data
        self.node = TreeNode()
        self.sample_weight = sample_weight

        if n_original_samples is None:
            self.n_original_samples = self.data.shape[0]
        else:
            self.n_original_samples = n_original_samples

        if min_gain is None:
            self.min_gain = np.log(self.data.shape[0])/(2*self.data.shape[0])
        else:
            self.min_gain = min_gain

        self.random_forest = random_forest

        self.lprior = 1 # laplace prior

        if m_priors is None:
            self.m_priors = np.zeros((self.data.shape[1],2))
            for i in range(self.data.shape[1]):
                self.m_priors[i,1] = (self.data[:,i].sum() + self.lprior/2)/ (self.data.shape[0] + self.lprior)
                self.m_priors[i,0] = 1 - self.m_priors[i][1]

            self.j_priors = np.zeros((self.data.shape[1],self.data.shape[1],2,2))

            sparse_cond = sparse.csr_matrix(self.data)
            cond0 = sparse_cond.T.dot(sparse_cond)
            cond = np.array(cond0.todense())
            """ FIXME to consider the weights """
            for i in range(self.data.shape[1]):
                for j in range(self.data.shape[1]):
                    if i != j:
                        self.j_priors[i,j,1,1] = cond[i,j] 
                        self.j_priors[i,j,0,1] = cond[j,j] - cond[i,j] 
                        self.j_priors[i,j,1,0] = cond[i,i] - cond[i,j] 
                        self.j_priors[i,j,0,0] = self.data.shape[0] - self.j_priors[i,j,1,1] - self.j_priors[i,j,0,1] - self.j_priors[i,j,1,0]
                        self.j_priors[i,j,1,1] = (self.j_priors[i,j,1,1] + self.lprior/4)/ (self.data.shape[0] + self.lprior)
                        self.j_priors[i,j,0,1] = (self.j_priors[i,j,0,1] + self.lprior/4)/ (self.data.shape[0] + self.lprior)
                        self.j_priors[i,j,1,0] = (self.j_priors[i,j,1,0] + self.lprior/4)/ (self.data.shape[0] + self.lprior)
                        self.j_priors[i,j,0,0] = (self.j_priors[i,j,0,0] + self.lprior/4)/ (self.data.shape[0] + self.lprior)

            sparse_cond = None
            cond0 = None
            cond = None
        else:
            self.m_priors = m_priors
            self.j_priors = j_priors

        if clt is None:
            self.node.cltree = Cltree()
            self.node.cltree.fit(data, self.m_priors, self.j_priors, alpha=self.alpha, beta=self.beta, 
                                 and_leaves=self.and_leaves, sample_weight=self.sample_weight)
            self.orig_ll = self.node.cltree.score_samples_log_proba(self.data, self.sample_weight)
            self.d = int(math.sqrt(self.data.shape[1]))
            sparsity = 0.0
            sparsity = len(self.data.nonzero()[0])
            sparsity /= (self.data.shape[1] * self.data.shape[0])
            logger.info("Dataset sparsity: %f", sparsity)
        else:
            self.node.cltree = clt
            self.orig_ll = ll
            self.d = d

        self.scope = self.node.cltree.scope

        self.id = Csn._id_node_counter
        Csn._id_node_counter = Csn._id_node_counter + 1
        print("Block", self.id, "on", len(self.node.cltree.scope), "features",  "local ll:", self.orig_ll)

        if self.data.shape[0] > self.min_instances:
            if self.data.shape[1] >= self.min_features:
                if not self.node.cltree.is_forest():
                    self.or_cut()
                else:
                    if self.and_inners == True:
                        self.and_cut()
                    else:
                        self.or_cut()                            
            else:
                print( " > no cutting due to few features")
        else:
            print(" > no cutting due to few instances")

        if is_tree_node(self.node):
            if self.depth > Csn._depth:
                Csn._depth = self.depth
            Csn._mean_depth = Csn._mean_depth + self.depth
            Csn._leaf_nodes = Csn._leaf_nodes + 1
            if self.node.cltree.is_forest():
                Csn._clforests = Csn._clforests + 1
            else:
                Csn._cltrees = Csn._cltrees + 1
            Csn._clt_edges = Csn._clt_edges + self.node.cltree.num_edges

    def check_correctness(self,k):
        mean = 0.0
        for world in itertools.product([0,1], repeat=k):
            prob = np.exp(self.instance_ll(world))
            mean = mean + prob
        return mean 


    def show(self):
        """ WRITEME """
        print ("Learned Cut Set Network")
        self._showl(0)
        print("OR nodes:", Csn._or_nodes)
        print("And nodes:", Csn._and_nodes)
        print("Leaves:", Csn._leaf_nodes)
        print("Cltrees:", Csn._cltrees)
        print("Clforests:", Csn._clforests)
        print("Edges outgoing OR nodes:", Csn._or_edges)
        print("Edges in CLtrees:", Csn._clt_edges)
        print("Total edges:", Csn._or_edges + Csn._clt_edges)
        print("Total nodes:", Csn._or_nodes + Csn._leaf_nodes + Csn._and_nodes)
        print("Depth:", Csn._depth)
        print("Mean Depth:", Csn._mean_depth / Csn._leaf_nodes)

    def _showl(self,level):
        """ WRITEME """
        if is_or_node(self.node):
            print(self.id,"OR", self.node.left_weight,self.node.left_child.id,self.node.right_child.id,"on",self.scope[self.node.or_feature])
            self.node.left_child._showl(level+1)
            self.node.right_child._showl(level+1)
        elif is_and_node(self.node):
            print(self.id, "AND", end="")
            for i in range(len(self.tree_forest)):
                if self.node.or_features[i] == None:
                    print("()", end="")
                else:
                    print("(",self.node.children_left[i].id,self.node.children_right[i].id,"on",self.node.cltree.scope[self.tree_forest[i][self.node.or_features[i]]],")", end="")
            print("")
            for i in range(len(self.tree_forest)):
                if self.node.or_features[i] is not None:
                    self.node.children_left[i]._showl(level+1)
                    self.node.children_right[i]._showl(level+1)
        else:
            print(self.id, "LEAF", end=" ")
            if self.node.cltree.is_forest():
                print("Forest")
            else:
                print("Tree")
                print(self.node.cltree.tree)
                print(self.node.cltree.scope)


    def mpe(self, x):
        """ WRITEME """
        if self._leaf:
            self.cltree.mpe(x)
        elif self._and:
            print("TODO")
        else:
            self.left_child.mpe(x)
            self.right_child.mpe(x)
            lv = self.left_child.mpe_value * left_weight
            rv = self.right_child.mpe_value * right_weight
            # maximization
            if lv > rv:
                self.mpe_value = lv
                self.mpe_state = 0
            else:
                self.mpe_value = rv
                self.mpe_state = 1
            
    def _score_sample_log_proba(self,x):
        """ WRITEME """
        prob = 0.0
        if is_tree_node(self.node):
            prob = prob + self.node.cltree.score_sample_log_proba(x)
        elif is_and_node(self.node):
            for i in range(len(self.tree_forest)):
                if self.node.or_features[i] == None:
                    prob = prob + self.node.cltree.score_sample_scope_log_proba(x,self.tree_forest[i])
                else:
                    x0 = x[self.tree_forest[i]]
                    x1 = np.concatenate((x0[0:self.node.or_features[i]],x0[self.node.or_features[i]+1:]))
                    if x0[self.node.or_features[i]] == 0:
                        prob = prob + logr(self.node.left_weights[i]) + self.node.children_left[i]._score_sample_log_proba(x1)
                    else:
                        prob = prob + logr(self.node.right_weights[i]) + self.node.children_right[i]._score_sample_log_proba(x1)
        else:
            x1 = np.concatenate((x[0:self.node.or_feature],x[self.node.or_feature+1:]))
            if x[self.node.or_feature] == 0:
                prob = prob + logr(self.node.left_weight) + self.node.left_child._score_sample_log_proba(x1)
            else:
                prob = prob + logr(self.node.right_weight) + self.node.right_child._score_sample_log_proba(x1)
        return prob

        
    def score_samples_log_proba(self, X, sample_weight=None):
        """ WRITEME """

        Prob = X[:,0]*0.0
        for i in range(X.shape[0]):
            Prob[i] = self._score_sample_log_proba(X[i])

        if sample_weight is None:
            m = np.sum(Prob) / X.shape[0]
        else:
            Prob = sample_weight * Prob
            m = np.sum(Prob) / np.sum(sample_weight)
        return m
        

    def and_cut(self):
        """ WRITEME """
        n_features = self.data.shape[1]
        self.forest = np.zeros(n_features, dtype=np.int)
        self.roots = []

        # naive approach to build the tree_forest
        for i in range(n_features):        
            if self.node.cltree.tree[i] == -1:
                self.roots.append(i)
        for i in range(n_features):        
            if self.node.cltree.tree[i] != -1:
                parent = self.node.cltree.tree[i]
                while self.node.cltree.tree[parent] != -1:
                    parent = self.node.cltree.tree[parent]
                self.forest[i] = parent
            else:
                self.forest[i] = i

        self.tree_forest = []
        for r in self.roots:
            t_forest = []
            for i in range(n_features):
                if self.forest[i] == r:
                    t_forest.append(i)
            self.tree_forest.append(t_forest)

        print ("AND node")
        print (self.tree_forest)

        for i in range(self.node.cltree.num_trees):

            print(" tree", self.tree_forest[i])
            sys.stdout.flush()

            tree_n_features = len(self.tree_forest[i])

            if self.data.shape[0] > self.min_instances:
                if tree_n_features >= self.min_features:

                    tree_data = self.data[:,self.tree_forest[i]]

                    found = False

                    orig_ll = self.node.cltree.score_samples_scope_log_proba(self.data, self.tree_forest[i], self.sample_weight)

                    bestlik = orig_ll
                    best_clt_l = None
                    best_clt_r = None
                    best_feature_cut = None
                    best_left_weight = 0.0
                    best_right_weight = 0.0
                    best_right_data = None
                    best_left_data = None
                    best_v_ll = 0.0
                    best_gain = -np.inf

                    if self.random_forest:
                        if self.d > tree_n_features:
                            selected = range(tree_n_features)
                        else:
                            selected = sorted(random.sample(range(tree_n_features), self.d))
                    else:
                        selected = range(tree_n_features)


                    for feature in selected:
                        condition = tree_data[:,feature]==0
                        new_features = np.ones(tree_data.shape[1], dtype=bool)
                        new_features[feature] = False
                        left_data = tree_data[condition,:][:, new_features]
                        right_data = tree_data[~condition,:][:, new_features]
                        left_weight = (left_data.shape[0] ) / (tree_data.shape[0] )
                        right_weight = (right_data.shape[0] ) / (tree_data.shape[0] )        

                        if self.sample_weight is not None:
                            left_sample_weight = self.sample_weight[condition]
                            right_sample_weight = self.sample_weight[~condition]
                        else:
                            left_sample_weight = None
                            right_sample_weight = None
                            

                        if left_data.shape[0]>self.min_instances and right_data.shape[0]>self.min_instances:
                            # compute the tree features id
                            tree_scope = np.zeros(tree_n_features, dtype=np.int)
                            for f in range(tree_n_features):
                                tree_scope[f] = self.node.cltree.scope[self.tree_forest[i][f]]

                            left_scope = np.concatenate((tree_scope[0:feature],tree_scope[feature+1:]))
                            right_scope = np.concatenate((tree_scope[0:feature],tree_scope[feature+1:]))

                            CL_l = Cltree()
                            CL_r = Cltree()

                            CL_l.fit(left_data,self.m_priors,self.j_priors,scope=left_scope,alpha=self.alpha*left_weight, beta=self.beta,
                                          and_leaves=self.and_leaves, sample_weight = left_sample_weight)
                            CL_r.fit(right_data,self.m_priors,self.j_priors,scope=right_scope,alpha=self.alpha*right_weight, beta=self.beta,
                                          and_leaves=self.and_leaves, sample_weight = right_sample_weight)

                            l_ll = CL_l.score_samples_log_proba(left_data, left_sample_weight)
                            r_ll = CL_r.score_samples_log_proba(right_data, right_sample_weight)

                            ll = ((l_ll+logr(left_weight))*left_data.shape[0] + (r_ll+logr(right_weight))*right_data.shape[0])/self.data.shape[0]
                        else:
                            ll = -np.inf

                        if ll>bestlik:

                            bestlik = ll
                            best_clt_l = CL_l
                            best_clt_r = CL_r
                            best_feature_cut = feature
                            best_left_weight = left_weight
                            best_right_weight = right_weight
                            best_right_data = right_data
                            best_left_data = left_data
                            best_l_ll = l_ll
                            best_r_ll = r_ll

                            best_left_sample_weight = left_sample_weight
                            best_right_sample_weight = right_sample_weight

                            found = True

                    gain = (bestlik - orig_ll)
                    print (" gain:", gain, end = " ")

                    if gain <= self.min_gain:
                        print("no improvement")

                    if found==True and gain > self.min_gain:

                        if not is_and_node(self.node):
                            clt = self.node.cltree
                            self.node = AndNode()
                            self.node.cltree = clt
                            self.node.children_left = [None] * self.node.cltree.num_trees
                            self.node.children_right = [None] * self.node.cltree.num_trees        
                            self.node.or_features = [None] * self.node.cltree.num_trees
                            self.node.left_weights = [None] * self.node.cltree.num_trees
                            self.node.right_weights = [None] * self.node.cltree.num_trees        
                   
                        Csn._or_nodes = Csn._or_nodes + 1
                        Csn._or_edges = Csn._or_edges + 2
                        
                        
                        self.node.or_features[i] = best_feature_cut
                        print(" cutting on feature ", self.node.or_features[i])

                        instances = self.data.shape[0]

                        self.node.left_weights[i] = best_left_weight
                        self.node.right_weights[i] = best_right_weight

                        self.node.children_left[i] = Csn(data=best_left_data, 
                                                         clt=best_clt_l, ll=best_l_ll, 
                                                         min_instances=self.min_instances, 
                                                         min_features=self.min_features, alpha=self.alpha*best_left_weight, 
                                                         d=self.d, random_forest=self.random_forest,
                                                         m_priors = self.m_priors, j_priors = self.j_priors,
                                                         n_original_samples = self.n_original_samples,
                                                         and_leaves=self.and_leaves, and_inners=self.and_inners,
                                                         min_gain = self.min_gain, beta=self.beta, depth=self.depth+1,
                                                         sample_weight = best_left_sample_weight)
                        self.node.children_right[i] = Csn(data=best_right_data, 
                                                          clt=best_clt_r, ll=best_r_ll, 
                                                          min_instances=self.min_instances, 
                                                          min_features=self.min_features, alpha=self.alpha*best_right_weight, d=self.d, 
                                                          random_forest=self.random_forest,
                                                          m_priors = self.m_priors, j_priors = self.j_priors,
                                                          n_original_samples = self.n_original_samples,
                                                          and_leaves=self.and_leaves, and_inners=self.and_inners,
                                                          min_gain = self.min_gain, beta=self.beta, depth=self.depth+1,
                                                          sample_weight = best_right_sample_weight)


                else:
                    print( " > no cutting due to few features")
            else:
                print(" > no cutting due to few instances")
        if is_and_node(self.node):
            Csn._and_nodes = Csn._and_nodes + 1

        # free memory before to recurse
        self.free_memory()



    def or_cut(self):
        """ WRITEME """
        print(" > trying to cut ... ", end = "")
        sys.stdout.flush()

        found = False

        bestlik = self.orig_ll
        best_clt_l = None
        best_clt_r = None
        best_feature_cut = None
        best_left_weight = 0.0
        best_right_weight = 0.0
        best_right_data = None
        best_left_data = None
        best_v_ll = 0.0
        best_gain = -np.inf
        best_left_sample_weight = None
        best_right_sample_weight = None
                            

        if self.random_forest:
            if self.d > self.node.cltree.n_features:
                selected = range(self.node.cltree.n_features)
            else:
                selected = sorted(random.sample(range(self.node.cltree.n_features), self.d))

        else:
            selected = range(self.node.cltree.n_features)


        for feature in selected:
            condition = self.data[:,feature]==0
            new_features = np.ones(self.data.shape[1], dtype=bool)
            new_features[feature] = False
            left_data = self.data[condition,:][:, new_features]
            right_data = self.data[~condition,:][:, new_features]
            left_weight = (left_data.shape[0] ) / (self.data.shape[0] )
            right_weight = (right_data.shape[0] ) / (self.data.shape[0] )        

            if self.sample_weight is not None:
                left_sample_weight = self.sample_weight[condition]
                right_sample_weight = self.sample_weight[~condition]
            else:
                left_sample_weight = None
                right_sample_weight = None
           
            if left_data.shape[0] > self.min_instances and right_data.shape[0] > self.min_instances:
                left_scope = np.concatenate((self.node.cltree.scope[0:feature],self.node.cltree.scope[feature+1:]))
                right_scope = np.concatenate((self.node.cltree.scope[0:feature],self.node.cltree.scope[feature+1:]))
                CL_l = Cltree()
                CL_r = Cltree()

                CL_l.fit(left_data,self.m_priors,self.j_priors,scope=left_scope,alpha=self.alpha*left_weight, beta=self.beta,
                              and_leaves=self.and_leaves, sample_weight = left_sample_weight)
                CL_r.fit(right_data,self.m_priors,self.j_priors,scope=right_scope,alpha=self.alpha*right_weight, beta=self.beta,
                              and_leaves=self.and_leaves, sample_weight = right_sample_weight)

                l_ll = CL_l.score_samples_log_proba(left_data, left_sample_weight)
                r_ll = CL_r.score_samples_log_proba(right_data, right_sample_weight)


                ll = ((l_ll+logr(left_weight))*left_data.shape[0] + (r_ll+logr(right_weight))*right_data.shape[0])/self.data.shape[0]
            else:
                ll = -np.inf

            if ll>bestlik:

                bestlik = ll
                best_clt_l = CL_l
                best_clt_r = CL_r
                best_feature_cut = feature
                best_left_weight = left_weight
                best_right_weight = right_weight
                best_right_data = right_data
                best_left_data = left_data
                best_l_ll = l_ll
                best_r_ll = r_ll

                best_left_sample_weight = left_sample_weight
                best_right_sample_weight = right_sample_weight
                
                found = True

        gain = (bestlik - self.orig_ll)
        print (" gain:", gain, end = "")
            
        if found==True and gain > self.min_gain:

            self.node = OrNode()
            Csn._or_nodes = Csn._or_nodes + 1
            Csn._or_edges = Csn._or_edges + 2
         
            self.node.or_feature = best_feature_cut
            print(" cutting on feature ", self.node.or_feature)

            instances = self.data.shape[0]
            
            self.node.left_weight = best_left_weight
            self.node.right_weight = best_right_weight

            # free memory before to recurse
            self.free_memory()

            self.node.left_child = Csn(data=best_left_data, 
                                       clt=best_clt_l, ll=best_l_ll, 
                                       min_instances=self.min_instances, 
                                       min_features=self.min_features, alpha=self.alpha*best_left_weight, 
                                       d=self.d, random_forest=self.random_forest,
                                       m_priors = self.m_priors, j_priors = self.j_priors,
                                       n_original_samples = self.n_original_samples,
                                       and_leaves=self.and_leaves, and_inners=self.and_inners,
                                       min_gain = self.min_gain, beta=self.beta, depth=self.depth+1,
                                       sample_weight = best_left_sample_weight)
            self.node.right_child = Csn(data=best_right_data, 
                                        clt=best_clt_r, ll=best_r_ll, 
                                        min_instances=self.min_instances, 
                                        min_features=self.min_features, alpha=self.alpha*best_right_weight, d=self.d, 
                                        random_forest=self.random_forest,
                                        m_priors = self.m_priors, j_priors = self.j_priors,
                                        n_original_samples = self.n_original_samples,
                                        and_leaves=self.and_leaves, and_inners=self.and_inners,
                                        min_gain = self.min_gain, beta=self.beta, depth=self.depth+1,
                                        sample_weight = best_right_sample_weight)
        else:
            print(" no cutting")
            if self.node.cltree.is_forest():
                print("   -> Forest with",self.node.cltree.num_trees, "trees")
            else:
                print("   -> Tree")


    def free_memory(self):
        self.data = None
        self.validation = None
