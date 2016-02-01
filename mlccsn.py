#!/usr/bin/python3

from csnm import Csnm
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
import mlcmetrics

DATA_PATH = 'data/'

def load_train_valid_test_arff(dataset,
                             path=DATA_PATH,
                             sep=',',
                             type='int',
                             suffixes=['.train.arff',
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

parser.add_argument('-l',  action='store_true', default=False,
                    help='Labels as leafs.')


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
#train, valid, test = load_train_valid_test_arff(dataset_name)
train, test = load_train_valid_test_arff(dataset_name)
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

                if args.l:
                    l_vars = [i for i in range(train.shape[1]-n_labels,train.shape[1])]
                else:
                    l_vars = []

                learn_start_t = perf_counter()
                C = Csnm(max_components=max_components, 
                         training_data=train, 
                         sample_weight = _sample_weight,
                         min_instances=min_instances, 
                         min_features=min_features, 
                         alpha=alpha, random_forest=rf,
                         leaf_vars = l_vars,
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
                    
                    Y_pred = mlcmetrics.compute_predictions(C, train, n_labels)
                    Y = mlcmetrics.extract_true_labels(train, n_labels)
                    print("p_exact_match", mlcmetrics.p_exact_match(Y, Y_pred))
                    print("l_hamming_loss", mlcmetrics.l_hamming_loss(Y, Y_pred))
                    print("p_accuracy", mlcmetrics.p_accuracy(Y, Y_pred))
                    print("p_precision_instances",mlcmetrics.p_precision_instances(Y, Y_pred))
                    print("p_precision_macro",mlcmetrics.p_precision_macro(Y, Y_pred))
                    print("p_precision_micro",mlcmetrics.p_precision_micro(Y, Y_pred))
                    print("p_recall_instances",mlcmetrics.p_recall_instances(Y, Y_pred))
                    print("p_recall_macro",mlcmetrics.p_recall_macro(Y, Y_pred))
                    print("p_recall_micro",mlcmetrics.p_recall_micro(Y, Y_pred))
                    print("p_F1_instances",mlcmetrics.p_F1_instances(Y, Y_pred))
                    print("p_F1_micro",mlcmetrics.p_F1_micro(Y, Y_pred))

                    Y_pred = mlcmetrics.compute_predictions(C, test, n_labels)
                    Y = mlcmetrics.extract_true_labels(test, n_labels)
                    print("p_exact_match", mlcmetrics.p_exact_match(Y, Y_pred))
                    print("l_hamming_loss", mlcmetrics.l_hamming_loss(Y, Y_pred))
                    print("p_accuracy", mlcmetrics.p_accuracy(Y, Y_pred))
                    print("p_precision_instances",mlcmetrics.p_precision_instances(Y, Y_pred))
                    print("p_precision_macro",mlcmetrics.p_precision_macro(Y, Y_pred))
                    print("p_precision_micro",mlcmetrics.p_precision_micro(Y, Y_pred))
                    print("p_recall_instances",mlcmetrics.p_recall_instances(Y, Y_pred))
                    print("p_recall_macro",mlcmetrics.p_recall_macro(Y, Y_pred))
                    print("p_recall_micro",mlcmetrics.p_recall_micro(Y, Y_pred))
                    print("p_F1_instances",mlcmetrics.p_F1_instances(Y, Y_pred))
                    print("p_F1_micro",mlcmetrics.p_F1_micro(Y, Y_pred))



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
