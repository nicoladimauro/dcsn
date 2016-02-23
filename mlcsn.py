import numpy as np
import csn as CSN


###############################################################################
class mlcsn:
    
    def __init__(self, data, sample_weight = None, 
                 p=1.0, min_instances=5, min_features=3, alpha=1.0, random_forest=False, leaf_vars = [],
                 and_leaves=False, and_inners=False, sum_nodes=False):

        self.data = data
        self.min_instances = min_instances
        self.min_features = min_features
        self.and_leaves = and_leaves
        self.and_inners = and_inners
        if self.and_leaves:
            self.and_nodes = True

        self.leaf_vars = leaf_vars
        self.sum_nodes = sum_nodes
        self.alpha = int(self.data.shape[0]*alpha/100)
        self.p = p
        self.random_forest = random_forest
        self.sample_weight = sample_weight

        self.or_nodes = 0.0
        self.n_sum_nodes = 0.0
        self.leaf_nodes = 0.0
        self.or_edges = 0.0
        self.clt_edges = 0.0
        self.and_nodes = 0.0
        self.cltrees = 0.0
        self.clforests = 0.0
        self.depth = 0.0
        self.mdepth = 0.0

        self.csn = None
   
    def fit(self):
        
        CSN.Csn.init_stats()

        self.csn = CSN.Csn(data=self.data,
                           sample_weight = self.sample_weight,
                           n_original_samples = self.data.shape[0],
                           min_instances=self.min_instances, min_features=self.min_features, alpha=self.alpha, 
                           random_forest=self.random_forest,
                           leaf_vars = self.leaf_vars,
                           and_leaves=self.and_leaves, and_inners=self.and_inners,
                           depth = 1, sum_nodes=self.sum_nodes)

        self.ll = self.csn.score_samples_log_proba(self.data)
        self.or_nodes = CSN.Csn._or_nodes 
        self.n_sum_nodes = CSN.Csn._sum_nodes 
        self.leaf_nodes = CSN.Csn._leaf_nodes
        self.or_edges = CSN.Csn._or_edges
        self.clt_edges = CSN.Csn._clt_edges
        self.and_nodes = CSN.Csn._and_nodes
        self.cltrees = CSN.Csn._cltrees
        self.clforests = CSN.Csn._clforests
        self.depth = CSN.Csn._depth
        self.mdepth = CSN.Csn._mean_depth / CSN.Csn._leaf_nodes

    def score_samples(self, data, out_filename):
        with open(out_filename, 'w') as out_log:
            mean = 0.0
            for x in data:
                prob = self.csn.score_sample_log_proba(x)
                mean = mean + prob
                out_log.write('%.10f\n'%prob)
        out_log.close()
        return mean / data.shape[0]

    def mpe(self, evidence = {}):
        return self.csn.mpe(evidence)

    def naiveMPE(self, evidence = {}):
        return self.csn.naiveMPE(evidence)

    def compute_predictions(self, X, n_labels):
        predictions = np.zeros((X.shape[0],n_labels),dtype=np.int)
        n_attributes = X.shape[1]
        k = 0
        for x in X:
            evidence = {}
            for i in range(n_attributes):
                evidence[i]=x[i]
            (state, prob) = self.mpe(evidence = evidence)
            sum = 0
            for i in range(n_attributes, n_attributes + n_labels):
                sum += state[i]
            if sum == 0:
                # avoiding empty predictions
                max_state = None
                max_prob = -np.inf
                for i in range(n_attributes, n_attributes + n_labels):
                    evidence[i] = 1
                    if i > (n_attributes):
                        del evidence[i-1]
                    (state1, prob1) = self.mpe(evidence = evidence)
                    if (prob1 > max_prob):
                        max_prob = prob1
                        max_state = state1
                state = max_state
                prob = max_prob
            y = 0
            for i in range(n_attributes, n_attributes + n_labels):
                predictions[k,y]=state[i]
                y += 1
            k += 1
        return predictions