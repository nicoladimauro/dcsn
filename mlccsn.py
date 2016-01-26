#!/usr/bin/python3

from csnm import csv_2_numpy, Csnm
import numpy as np
import argparse
import shutil

import arff

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import numpy
import datetime
import os
import logging
import random

DATA_PATH = 'data/'



def load_train_valid_test_arff(dataset,
                             path=DATA_PATH,
                             sep=',',
                             type='int',
                             suffixes=['.train.arff',
                                       '.valid.arff',
                                       '.test.arff']):


    datasets = []
    files = [path + dataset + ext for ext in suffixes]
    for file in files:
        data = arff.load(open(file, 'r'),encode_nominal=True)
        datasets.append(np.array(data['data']).astype(type))
    return datasets


def stats_format(stats_list, separator, digits=5):
    formatted = []
    float_format = '{0:.' + str(digits) + 'f}'
    for stat in stats_list:
        if isinstance(stat, int):
            formatted.append(str(stat))
        elif isinstance(stat, float):
            formatted.append(float_format.format(stat))
        else:
            formatted.append(stat)
    # concatenation
    return separator.join(formatted)


def exact_match(C, X, n_labels):
    c = 0
    for x in X:
        evidence = {}
        for i in range(len(x)-n_labels):
            evidence[i]=x[i]
        (state, prob) = C.mpe(evidence = evidence)
        Z = []
        for i in range(len(x)-n_labels,len(x)):
            Z.append(state[i])
        exact = True
        Y = x[-n_labels:]
        for i in range(n_labels):
            if Z[i]!=Y[i]:
                exact = False
                break
        if exact == True:
            c += 1
    return (c / X.shape[0])

def subset_accuracy(C, X, n_labels):
    c = 0
    for x in X:
        evidence = {}
        for i in range(len(x)-n_labels):
            evidence[i]=x[i]
        (state, prob) = C.mpe(evidence = evidence)
        Z = []
        for i in range(len(x)-n_labels,len(x)):
            Z.append(state[i])
        Y = x[-n_labels:]
        for i in range(n_labels):
            if Z[i]==Y[i]:
                c +=1
    return (c / X.shape[0])

# Jaccard Index -- often simply called multi-label 'accuracy'. Multi-label only. 
def accuracy(C, X, n_labels):
    c = 0.0
    for x in X:
        evidence = {}
        for i in range(len(x)-n_labels):
            evidence[i]=x[i]
        (state, prob) = C.mpe(evidence = evidence)
        Z = []
        for i in range(len(x)-n_labels,len(x)):
            Z.append(state[i])
        Y = x[-n_labels:]
        union = 0
        inter = 0
        for i in range(n_labels):
            if (Z[i]==1 and Y[i]==1):
                inter += 1
            if (Z[i]==1 or Y[i]==1):
                union +=1
        # = intersection / union; (or, if both sets are empty, then = 1.)
        if (union > 0):
            c = c + (inter / union)
        else:
            c = c + 1.0
    return (c / X.shape[0])


def hamming_loss(C, X, n_labels):
    c = 0.0
    for x in X:
        evidence = {}
        for i in range(len(x)-n_labels):
            evidence[i]=x[i]
        (state, prob) = C.mpe(evidence = evidence)
        Z = []
        for i in range(len(x)-n_labels,len(x)):
            Z.append(state[i])
        Y = x[-n_labels:]
        loss = 0
        for i in range(n_labels):
            if (Z[i]!=Y[i]):
                loss += 1
        c = c + (loss / n_labels)
    return (c / X.shape[0])



#########################################
# creating the opt parser
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, nargs=1,
                    help='Specify a dataset name from data/ (es. nltcs)')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/csn/',
                    help='Output dir path')

parser.add_argument('-r', '--random', action='store_true', default=False,
                    help='Random Forest. If set a Random Forest approach is used.')

parser.add_argument('--sum', action='store_true', default=False,
                    help='Use sum nodes.')

parser.add_argument('-k', type=int, nargs='+',
                    default=[1],
                    help='Number of components to use. If greater than 1, then a bagging approach is used.')

parser.add_argument('-d', type=int, nargs='+',
                    default=[10],
                    help='Min number of instances in a slice to split.')

parser.add_argument('-s', type=int, nargs='+',
                    default=[4],
                    help='Min number of features in a slice to split.')

parser.add_argument('-a', '--alpha', type=float, nargs='+',
                    default=[1.0],
                    help='Smoothing factor for leaf probability estimation')

parser.add_argument('--al', action='store_true', default=False,
                    help='Use and nodes as leaves (i.e., CL forests).')

parser.add_argument('--an', action='store_true', default=False,
                    help='Use and nodes as inner nodes and leaves (i.e., CL forests).')

parser.add_argument('-c', type=int, nargs=1,
                    help='Number of class labels.')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')


#
# parsing the args
args = parser.parse_args()

#
# setting verbosity level
if args.verbose == 1:
    logging.basicConfig(level=logging.INFO)
elif args.verbose == 2:
    logging.basicConfig(level=logging.DEBUG)

logging.info("Starting with arguments:\n%s", args)
# I shall print here all the stats

#
# gathering parameters
alphas = args.alpha
n_components = args.k
rf = args.random
m_instances = args.d
m_features = args.s

and_leaf = args.al
and_node = args.an

n_labels = args.c[0]


sum_nodes = args.sum

#
# elaborating the dataset
#
logging.info('Loading datasets: %s', args.dataset)
(dataset_name,) = args.dataset
train, valid, test = load_train_valid_test_arff(dataset_name)
n_instances = train.shape[0]
n_test_instances = test.shape[0]

#
# Opening the file for test prediction
#
logging.info('Opening log file...')
date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_path = args.output + dataset_name + '_' + date_string
out_log_path = out_path + '/exp.log'

#
# creating dir if non-existant
if not os.path.exists(os.path.dirname(out_log_path)):
    os.makedirs(os.path.dirname(out_log_path))

best_valid_avg_ll = -np.inf
best_state = {}

preamble = ("""components,alpha,minst,mfeat,or_nodes,sum_nodes,and_nodes,leaf_nodes,or_edges,clt_edges,cltrees,clforests,depth,mdepth,time,""" +
            """train_ll,valid_ll,test_ll\n""")

max_components = max(n_components)

np.random.seed(1)

with open(out_log_path, 'w') as out_log:

    out_log.write("parameters:\n{0}\n\n".format(args))
    out_log.write(preamble)
    out_log.flush()
    #
    # looping over all parameters combinations
    for alpha in alphas:
        for min_instances in m_instances:
            for min_features in m_features:

                C = None

                # initing the random generators
                seed = args.seed
                numpy_rand_gen = numpy.random.RandomState(seed)
                random.seed(seed)

                ######################################################################
                #                    _sample_weight = np.ones(train.shape[0])
                #                    mean = 1
                #                    variance = 0.1
                #                    g_alpha = mean * mean / variance
                #                    g_beta = mean / variance
                #                    for i in range(train.shape[0]):
                #                        _sample_weight[i] = random.gammavariate(g_alpha, 1/g_beta)
                ######################################################################
                _sample_weight = None

                learn_start_t = perf_counter()
                C = Csnm(max_components=max_components, 
                         training_data=train, 
                         sample_weight = _sample_weight,
                         min_instances=min_instances, 
                         min_features=min_features, 
                         alpha=alpha, random_forest=rf,
                         leaf_vars = [],
                         and_leaves = and_leaf,
                         and_inners = and_node,sum_nodes = sum_nodes)

                C.fit()

                learn_end_t = perf_counter()

                learning_time = (learn_end_t - learn_start_t)


                #
                # gathering statistics
    #            n_nodes = csn.n_nodes()
    #            n_levels = csn.n_levels()
    #            n_leaves = csn.n_leaves()

                for c in n_components:
                    #
                    # Compute LL on training set

                    print ("Exact match: ",exact_match(C, train, n_labels))
                    print ("Subset accuracy: ",subset_accuracy(C, train, n_labels))
                    print ("Accuracy: ",accuracy(C, train, n_labels))
                    print ("Hamming loss: ",hamming_loss(C, train, n_labels))

                    out_filename = out_path + '/c' + str(c) +'train.lls'
                    logging.info('Evaluating on training set')
                    train_avg_ll = C.score_samples(train, c, out_filename)

                    #
                    # Compute LL on validation set
                    out_filename = out_path + '/c' + str(c) +'valid.lls'
                    logging.info('Evaluating on validation set')
                    valid_avg_ll = C.score_samples(valid, c, out_filename)

                    #
                    # Compute LL on test set
                    out_filename = out_path + '/c' + str(c) +'test.lls'
                    logging.info('Evaluating on test set')
                    test_avg_ll = C.score_samples(test, c, out_filename)

                    #
                    # updating best stats according to valid ll
                    if valid_avg_ll > best_valid_avg_ll:
                        best_valid_avg_ll = valid_avg_ll
                        best_state['alpha'] = alpha
                        best_state['m_inst'] = min_instances
                        best_state['m_feat'] = min_features
                        best_state['time'] = learning_time
                        best_state['train_ll'] = train_avg_ll
                        best_state['valid_ll'] = valid_avg_ll
                        best_state['test_ll'] = test_avg_ll
                        shutil.copy2(out_path + '/c' + str(c) +'train.lls',out_path+'/besttrain.lls')
                        shutil.copy2(out_path + '/c' + str(c) +'test.lls',out_path+'/besttest.lls')
                        shutil.copy2(out_path + '/c' + str(c) +'valid.lls',out_path+'/bestvalid.lls')
                    os.remove(out_path + '/c' + str(c) +'train.lls')
                    os.remove(out_path + '/c' + str(c) +'test.lls')
                    os.remove(out_path + '/c' + str(c) +'valid.lls')

                    or_nodes = sum(C.or_nodes[:c])/c
                    n_sum_nodes = sum(C.n_sum_nodes[:c])/c
                    and_nodes = sum(C.and_nodes[:c])/c
                    leaf_nodes = sum(C.leaf_nodes[:c])/c
                    or_edges = sum(C.or_edges[:c])/c
                    clt_edges = sum(C.clt_edges[:c])/c
                    cltrees = sum(C.cltrees[:c])/c
                    clforests = sum(C.clforests[:c])/c
                    depth = sum(C.depth[:c])/c
                    mdepth = sum(C.mdepth[:c])/c



                    #
                    # writing to file a line for the grid
                    stats = stats_format([c,
                                          alpha,
                                          min_instances,
                                          min_features,
                                          or_nodes,
                                          n_sum_nodes,
                                          and_nodes,
                                          leaf_nodes,
                                          or_edges,
                                          clt_edges,
                                          cltrees,
                                          clforests,
                                          depth,
                                          mdepth,
                                          learning_time,
                                          train_avg_ll,
                                          valid_avg_ll,
                                          test_avg_ll],
                                         ',',
                                         digits=5)
                    out_log.write(stats + '\n')
                    out_log.flush()

    #
    # writing as last line the best params
    out_log.write("{0}".format(best_state))
    out_log.flush()

logging.info('Grid search ended.')
logging.info('Best params:\n\t%s', best_state)
