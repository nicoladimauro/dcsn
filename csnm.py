import numpy as np
import random
import time
import logging
import scipy
import csv

from logr import logr
import csn as CSN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
DATA_PATH = 'data/'

def csv_2_numpy(file, path=DATA_PATH, sep=',', type='int'):
    """
    WRITEME
    """
    file_path = path + file
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    dataset = np.array(x).astype(type)
    return dataset


class Csnm:
    
    def __init__(self, training_data, sample_weight = None, max_components=1,
                 p=1.0, min_instances=5, min_features=3, alpha=1.0, random_forest=False, leaf_vars = [],
                 and_leaves=False, and_inners=False, sum_nodes=False):

        self.max_components = max_components
        self.training_data = training_data
        self.min_instances = min_instances
        self.min_features = min_features
        self.and_leaves = and_leaves
        self.and_inners = and_inners
        if self.and_leaves:
            self.and_nodes = True

        self.leaf_vars = leaf_vars

        self.sum_nodes = sum_nodes
        self.alpha = int(self.training_data.shape[0]*alpha/100)
        logger.info("Setting alpha to %d",self.alpha)

        self.p = p
        self.random_forest = random_forest
        
        self.sample_weight = sample_weight


        self.bags = [None] * self.max_components
        self.bags_weight = [None] * self.max_components
        self.csns = [None] * self.max_components
        self.weights = [1/self.max_components] * self.max_components
        self.lls = [0.0] * self.max_components


        self.or_nodes = [0.0] * self.max_components
        self.n_sum_nodes = [0.0] * self.max_components
        self.leaf_nodes = [0.0] * self.max_components
        self.or_edges = [0.0] * self.max_components
        self.clt_edges = [0.0] * self.max_components
        self.and_nodes = [0.0] * self.max_components
        self.cltrees = [0.0] * self.max_components
        self.clforests = [0.0] * self.max_components
        self.depth = [0.0] * self.max_components
        self.mdepth = [0.0] * self.max_components

        self.create_bags()
#        print("Correctness:",self.check_correctness())

    def create_bags(self):
        if self.max_components == 1:
            self.bags[0] = self.training_data
            self.bags_weight[0] = self.sample_weight
        else:
            for i in range(self.max_components):
                self.bags[i], self.bags_weight[i] = self._create_bag()

    def _create_bag(self):
        n_instances = int(self.p * self.training_data.shape[0])
        bag = np.zeros((n_instances, self.training_data.shape[1]), dtype='int')
        if self.sample_weight is None:
            bag_weight = None
        else:
            bag_weight = np.zeros(n_instances)
        for i in range(n_instances):
            choice = random.randint(0, self.training_data.shape[0]-1)
            bag[i] = self.training_data[choice]
            if self.sample_weight is not None:
                bag_weight[i] = self.sample_weight[choice]
        return bag, bag_weight

    
    def fit(self):
        for i in range(self.max_components):

            CSN.Csn.init_stats()

            self.csns[i] = CSN.Csn(data=self.bags[i],  
                                   sample_weight = self.bags_weight[i],
                                   n_original_samples = self.bags[i].shape[0],
                                   min_instances=self.min_instances, min_features=self.min_features, alpha=self.alpha, 
                                   random_forest=self.random_forest,
                                   leaf_vars = self.leaf_vars,
                                   and_leaves=self.and_leaves, and_inners=self.and_inners,
                                   depth = 1, sum_nodes=self.sum_nodes)

            self.csns[i].show()
            self.lls[i] = self.csns[i].score_samples_log_proba(self.training_data)
            self.or_nodes[i] = CSN.Csn._or_nodes 
            self.n_sum_nodes[i] = CSN.Csn._sum_nodes 
            self.leaf_nodes[i] = CSN.Csn._leaf_nodes
            self.or_edges[i] = CSN.Csn._or_edges
            self.clt_edges[i] = CSN.Csn._clt_edges
            self.and_nodes[i] = CSN.Csn._and_nodes
            self.cltrees[i] = CSN.Csn._cltrees
            self.clforests[i] = CSN.Csn._clforests
            self.depth[i] = CSN.Csn._depth
            self.mdepth[i] = CSN.Csn._mean_depth / CSN.Csn._leaf_nodes


#            print("Correct:", self.csns[i].check_correctness(self.bags[i].shape[1]))



    def compute_weights(self, n_c):
        sum_ll = 0.0
        for i in range(n_c):
            sum_ll += self.lls[i]
        for i in range(n_c):
            self.weights[i] = self.lls[i] / sum_ll

    def score_samples(self, data, n_c, out_filename):
        with open(out_filename, 'w') as out_log:
            self.compute_weights(n_c)
            mean = 0.0
            for x in data:
                prob = 0.0
                for k in range(n_c):
                    prob = prob + np.exp(self.csns[k].score_sample_log_proba(x))*self.weights[k]
                mean = mean + logr(prob)
                out_log.write('%.10f\n'%logr(prob))
        out_log.close()
        return mean / data.shape[0]

    def mpe(self, evidence = {}):
        if self.max_components > 1:
            raise mpeError("mpe can be executed with 1 components only.")
        return self.csns[0].mpe(evidence)

    def naiveMPE(self, evidence = {}):
        if self.max_components > 1:
            raise mpeError("mpe can be executed with 1 components only.")
        return self.csns[0].naiveMPE(evidence)


    def check_correctness(self):
        mean = 0.0
        for world in itertools.product([0,1], repeat=self.training_data.cols):
            prob = 0.0
            for k in range(self.max_components):
                prob = prob + np.exp(self.csns[k]._score_sample_log_proba(world))
            prob = prob / self.max_components
            mean = mean + prob
        return mean 

