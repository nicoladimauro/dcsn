#!/usr/bin/python

import numpy as np
from scipy import sparse
import math 
import logging
import sys
import itertools 
import random

from logr import logr
from cltree import Cltree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    def __init__(self, 
                 data, 
                 clt = None, 
                 ll = 0.0,  
                 min_instances = 5, 
                 min_features = 3, 
                 alpha = 1.0, 
                 beta = 1.0, 
                 d = None, 
                 random_forest = False, 
                 m_priors = None, 
                 j_priors = None, 
                 dataset_instances = None,
                 and_leaves=False, 
                 and_inners=False,
                 min_gain = None,
                 depth = 1):

        self.min_instances = min_instances
        self.min_features = min_features
        self._or = False
        self._and = False
        self._leaf = True

        self.and_leaves = and_leaves
        self.and_inners = and_inners
        self.alpha = alpha
        self.beta = beta
        self.depth = depth

        self.data = data
        if dataset_instances is None:
            self.dataset_instances = self.data.shape[0]
        else:
            self.dataset_instances = dataset_instances

        if min_gain is None:
            self.min_gain = np.log(self.data.shape[0])/(2*self.data.shape[0])
        else:
            self.min_gain = min_gain

        self.random_forest = random_forest

        self.lprior = 1

        if m_priors is None:
            self.m_priors = np.zeros((self.data.shape[1],2))
            for i in range(self.data.shape[1]):
                self.m_priors[i,1] = (self.data[:,i].sum() + self.lprior/2)/ (self.data.shape[0] + self.lprior)
                self.m_priors[i,0] = 1 - self.m_priors[i][1]

            self.j_priors = np.zeros((self.data.shape[1],self.data.shape[1],2,2))

            sparse_cond = sparse.csr_matrix(self.data)
            cond0 = sparse_cond.T.dot(sparse_cond)
            cond = np.array(cond0.todense())
            for i in range(self.data.shape[1]):
                for j in range(self.data.shape[1]):
                    if i != j:
                        self.j_priors[i,j,1,1] = cond[i,j] 
                        self.j_priors[i,j,0,1] = cond[j,j] - cond[i,j] 
                        self.j_priors[i,j,1,0] = cond[i,i] - cond[i,j] 
                        self.j_priors[i,j,0,0] = self.data.shape[0] - self.j_priors[i,j,1,1] - self.j_priors[i,j,0,1] - self.j_priors[i,j,1,0]
                        self.j_priors[i,j,1,1] = (self.j_priors[i,j,1,1] + self.lprior/4)/ (self.data.shape[0]+ self.lprior)
                        self.j_priors[i,j,0,1] = (self.j_priors[i,j,0,1] + self.lprior/4)/ (self.data.shape[0]+ self.lprior)
                        self.j_priors[i,j,1,0] = (self.j_priors[i,j,1,0] + self.lprior/4)/ (self.data.shape[0]+ self.lprior)
                        self.j_priors[i,j,0,0] = (self.j_priors[i,j,0,0] + self.lprior/4)/ (self.data.shape[0]+ self.lprior)

            sparse_cond = None
            cond0 = None
            cond = None
        else:
            self.m_priors = m_priors
            self.j_priors = j_priors


        self.cltree = None
        if clt is None:
            self.cltree = Cltree(data, self.m_priors, self.j_priors, alpha=self.alpha, beta=self.beta, is_root=True,
                                 and_leaves=self.and_leaves)
            self.orig_ll = self.cltree.ll(self.data)
            self.d = int(math.sqrt(self.data.shape[1]))

            sparsity = 0.0
            sparsity = len(self.data.nonzero()[0])
            sparsity /= (self.data.shape[1] * self.data.shape[0])
            logger.info("Dataset sparsity: %f", sparsity)
        else:
            self.cltree = clt
            self.orig_ll = ll
            self.d = d

        self.features_id = self.cltree.features_id

        self.root_feature = None
        self.left_child = None
        self.right_child = None

        self.left_weight = 0.0
        self.right_weight = 0.0

        self.id = Csn._id_node_counter
        Csn._id_node_counter = Csn._id_node_counter + 1
        print("Block", self.id, "on", len(self.cltree.features_id), "features",  "local ll:", self.orig_ll)



        if self.data.shape[0] > self.min_instances:
            if self.data.shape[1] >= self.min_features:
                if not self.cltree._forest:
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




        if self._leaf:
            if self.depth > Csn._depth:
                Csn._depth = self.depth
            Csn._mean_depth = Csn._mean_depth + self.depth
            Csn._leaf_nodes = Csn._leaf_nodes + 1
            if self.cltree._forest:
                Csn._clforests = Csn._clforests + 1
            else:
                Csn._cltrees = Csn._cltrees + 1
            Csn._clt_edges = Csn._clt_edges + self.cltree.num_edges


        self.data = None
        self.m_priors = None
        self.j_priors = None
        if self.root_feature is not None:
            self.cltree = None

    def check_correctness(self,k):
        mean = 0.0
        for world in itertools.product([0,1], repeat=k):
            prob = np.exp(self.instance_ll(world))
            mean = mean + prob
        return mean 


    def show(self):

        print ("Learned Cut Set Network")
        self.showl(0)
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

    def showl(self,level):
        if self._or:
            print(self.id,"OR", self.left_weight,self.left_child.id,self.right_child.id,"on",self.features_id[self.root_feature])
            self.left_child.showl(level+1)
            self.right_child.showl(level+1)
        elif self._and:
            print(self.id, "AND", end="")
            for i in range(len(self.tree_forest)):
                if self.cut_features[i] == None:
                    print("()", end="")
                else:
                    print("(",self.children_left[i].id,self.children_right[i].id,"on",self.cltree.features_id[self.tree_forest[i][self.cut_features[i]]],")", end="")
            print("")
            for i in range(len(self.tree_forest)):
                if self.cut_features[i] is not None:
                    self.children_left[i].showl(level+1)
                    self.children_right[i].showl(level+1)
        else:
            print(self.id, "LEAF", end=" ")
            if self.cltree._forest:
                print("Forest")
            else:
                print("Tree")
                print(self.cltree.tree)
                print(self.cltree.features_id)

            

    def instance_ll(self,x):
        prob = 0.0
        if self._leaf:
            prob = prob + self.cltree.ll_instance_from_csn(x)
        elif self._and:
            for i in range(len(self.tree_forest)):
                if self.cut_features[i] == None:
                    prob = prob + self.cltree.sub_ll_instance_from_csn(x,self.tree_forest[i])
                else:
                    x0 = x[self.tree_forest[i]]
                    x1 = np.concatenate((x0[0:self.cut_features[i]],x0[self.cut_features[i]+1:]))
                    if x0[self.cut_features[i]] == 0:
                        prob = prob + logr(self.left_weights[i]) + self.children_left[i].instance_ll(x1)
                    else:
                        prob = prob + logr(self.right_weights[i]) + self.children_right[i].instance_ll(x1)
        else:
            x1 = np.concatenate((x[0:self.root_feature],x[self.root_feature+1:]))
            if x[self.root_feature] == 0:
                prob = prob + logr(self.left_weight) + self.left_child.instance_ll(x1)
            else:
                prob = prob + logr(self.right_weight) + self.right_child.instance_ll(x1)
        return prob
        
    def ll(self, data):
        mean = 0.0
        for x in data:
            prob = self.instance_ll(x)
            mean = mean + prob
        return mean / data.shape[0]
        

    def and_cut(self):

        n_features = self.data.shape[1]
        self.forest = np.zeros(n_features, dtype=np.int)
        self.roots = []

        # naive approach to build the tree_forest
        for i in range(n_features):        
            if self.cltree.tree[i] == -1:
                self.roots.append(i)
        for i in range(n_features):        
            if self.cltree.tree[i] != -1:
                parent = self.cltree.tree[i]
                while self.cltree.tree[parent] != -1:
                    parent = self.cltree.tree[parent]
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

                    
        self.children_left = [None] * self.cltree.num_trees
        self.children_right = [None] * self.cltree.num_trees        
        self.cut_features = [None] * self.cltree.num_trees
        self.left_weights = [None] * self.cltree.num_trees
        self.right_weights = [None] * self.cltree.num_trees        

        for i in range(self.cltree.num_trees):

            print(" tree", self.tree_forest[i])
            sys.stdout.flush()

            tree_n_features = len(self.tree_forest[i])

            if self.data.shape[0] > self.min_instances:
                if tree_n_features >= self.min_features:

                    tree_data = self.data[:,self.tree_forest[i]]

                    found = False

                    orig_ll = self.cltree.sub_ll(self.data, self.tree_forest[i])
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



                        if left_data.shape[0]>self.min_instances and right_data.shape[0]>self.min_instances:
                            # compute the tree features id
                            tree_features_id = np.zeros(tree_n_features, dtype=np.int)
                            for f in range(tree_n_features):
                                tree_features_id[f] = self.cltree.features_id[self.tree_forest[i][f]]

                            


                            left_features_id = np.concatenate((tree_features_id[0:feature],tree_features_id[feature+1:]))
                            right_features_id = np.concatenate((tree_features_id[0:feature],tree_features_id[feature+1:]))


                            CL_l = Cltree(left_data,self.m_priors,self.j_priors,features_id=left_features_id,alpha=self.alpha*left_weight, beta=self.beta,
                                          and_leaves=self.and_leaves)
                            CL_r = Cltree(right_data,self.m_priors,self.j_priors,features_id=right_features_id,alpha=self.alpha*right_weight, beta=self.beta,
                                          and_leaves=self.and_leaves)

                            l_ll = CL_l.ll(left_data)
                            r_ll = CL_r.ll(right_data)


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

                            found = True

                    gain = (bestlik - orig_ll)*self.data.shape[0]/self.dataset_instances
                    print (" gain:", gain, end = " ")

                    if gain <= self.min_gain:
                        print("no improvement")

                    if found==True and gain > self.min_gain:


                        self._leaf = False
                        self._and = True
                        Csn._or_nodes = Csn._or_nodes + 1
                        Csn._or_edges = Csn._or_edges + 2
                        
                        
                        self.cut_features[i] = best_feature_cut
                        print(" cutting on feature ", self.cut_features[i])

                        instances = self.data.shape[0]

                        self.left_weights[i] = best_left_weight
                        self.right_weights[i] = best_right_weight

                        self.children_left[i] = Csn(data=best_left_data, 
                                                    clt=best_clt_l, ll=best_l_ll, 
                                                    min_instances=self.min_instances, 
                                                    min_features=self.min_features, alpha=self.alpha*best_left_weight, 
                                                    d=self.d, random_forest=self.random_forest,
                                                    m_priors = self.m_priors, j_priors = self.j_priors,
                                                    dataset_instances = self.dataset_instances,
                                                    and_leaves=self.and_leaves, and_inners=self.and_inners,
                                                    min_gain = self.min_gain, beta=self.beta, depth=self.depth+1)
                        self.children_right[i] = Csn(data=best_right_data, 
                                                     clt=best_clt_r, ll=best_r_ll, 
                                                     min_instances=self.min_instances, 
                                                     min_features=self.min_features, alpha=self.alpha*best_right_weight, d=self.d, 
                                                     random_forest=self.random_forest,
                                                     m_priors = self.m_priors, j_priors = self.j_priors,
                                                     dataset_instances = self.dataset_instances,
                                                     and_leaves=self.and_leaves, and_inners=self.and_inners,
                                                     min_gain = self.min_gain, beta=self.beta, depth=self.depth+1)


                else:
                    print( " > no cutting due to few features")
            else:
                print(" > no cutting due to few instances")
        if self._and:
            Csn._and_nodes = Csn._and_nodes + 1




    def or_cut(self):
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

        if self.random_forest:
            if self.d > self.cltree.n_features:
                selected = range(self.cltree.n_features)
            else:
                selected = sorted(random.sample(range(self.cltree.n_features), self.d))

        else:
            selected = range(self.cltree.n_features)


        for feature in selected:

            condition = self.data[:,feature]==0
            new_features = np.ones(self.data.shape[1], dtype=bool)
            new_features[feature] = False
            left_data = self.data[condition,:][:, new_features]
            right_data = self.data[~condition,:][:, new_features]

            left_weight = (left_data.shape[0] ) / (self.data.shape[0] )
            right_weight = (right_data.shape[0] ) / (self.data.shape[0] )        

           
            if left_data.shape[0]>self.min_instances and right_data.shape[0]>self.min_instances:
                left_features_id = np.concatenate((self.cltree.features_id[0:feature],self.cltree.features_id[feature+1:]))
                right_features_id = np.concatenate((self.cltree.features_id[0:feature],self.cltree.features_id[feature+1:]))
                CL_l = Cltree(left_data,self.m_priors,self.j_priors,features_id=left_features_id,alpha=self.alpha*left_weight, beta=self.beta,
                              and_leaves=self.and_leaves)
                CL_r = Cltree(right_data,self.m_priors,self.j_priors,features_id=right_features_id,alpha=self.alpha*right_weight, beta=self.beta,
                              and_leaves=self.and_leaves)

                l_ll = CL_l.ll(left_data)
                r_ll = CL_r.ll(right_data)


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

                found = True


        gain = (bestlik - self.orig_ll)*self.data.shape[0]/self.dataset_instances
        print (" gain:", gain, end = "")

        if found==True and gain > self.min_gain:

            self._leaf = False
            self._or = True
            Csn._or_nodes = Csn._or_nodes + 1
            Csn._or_edges = Csn._or_edges + 2
         
            self.root_feature = best_feature_cut
            print(" cutting on feature ", self.root_feature)

            instances = self.data.shape[0]
            self.data = None
            self.validation = None
            self.cltree = None
            
            self.left_weight = best_left_weight
            self.right_weight = best_right_weight


            self.left_child = Csn(data=best_left_data, 
                                  clt=best_clt_l, ll=best_l_ll, 
                                  min_instances=self.min_instances, 
                                  min_features=self.min_features, alpha=self.alpha*best_left_weight, 
                                  d=self.d, random_forest=self.random_forest,
                                  m_priors = self.m_priors, j_priors = self.j_priors,
                                  dataset_instances = self.dataset_instances,
                                  and_leaves=self.and_leaves, and_inners=self.and_inners,
                                  min_gain = self.min_gain, beta=self.beta, depth=self.depth+1)
            self.right_child = Csn(data=best_right_data, 
                                   clt=best_clt_r, ll=best_r_ll, 
                                   min_instances=self.min_instances, 
                                   min_features=self.min_features, alpha=self.alpha*best_right_weight, d=self.d, 
                                   random_forest=self.random_forest,
                                   m_priors = self.m_priors, j_priors = self.j_priors,
                                   dataset_instances = self.dataset_instances,
                                   and_leaves=self.and_leaves, and_inners=self.and_inners,
                                   min_gain = self.min_gain, beta=self.beta, depth=self.depth+1)

        else:
            print(" no cutting")
            if self.cltree.num_trees>1:
                print("   -> Forest with",self.cltree.num_trees, "trees")
            else:
                print("   -> Tree")
