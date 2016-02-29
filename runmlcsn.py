#!/usr/bin/python3

from mlcsn import mlcsn
import numpy as np
import argparse
import shutil

import sklearn.metrics

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
import sklearn.metrics
from dataset import Dataset
import arff
from tabulate import tabulate

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
                    default='./exp/mlcsn/',
                    help='Output dir path')

parser.add_argument('-r', '--random', action='store_true', default=False,
                    help='Random Forest. If set a Random Forest approach is used.')

parser.add_argument('--sum', action='store_true', default=False,
                    help='Use sum nodes.')

parser.add_argument('-k', type=int, nargs='+',
                    default=[1],
                    help='Number of components to use. If greater than 1, then a bagging approach is used.')

parser.add_argument('-d', type=float, nargs='+',
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

parser.add_argument('-f', type=int, nargs='?',
                    default=5,
                    help='Number of folds for the dataset')


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

np.random.seed(1)



#train = Dataset.load_arff("./data/"+dataset_name, 6, endian = "big", input_feature_type = 'int', encode_nominal = True)
#n_instances = train.shape[0]
#n_test_instances = test.shape[0]

with open(out_log_path, 'w') as out_log:



    out_log.write("parameters:\n{0}\n\n".format(args))
    out_log.flush()
    #
    # looping over all parameters combinations
    for alpha in alphas:
        for min_instances in m_instances:
            for min_features in m_features:

                Accuracy = ['Accuracy']
                Hamming_score = ['Hamming Score']
                Exact_match = ['Exact match']
                Learning_time = ['Learning time']
                Testing_time = ['Testing time']
                Headers = ['Metric']

                for f in range(args.f):
                
                    C = None

                    # initing the random generators
                    seed = args.seed
                    numpy_rand_gen = numpy.random.RandomState(seed)
                    random.seed(seed)

                    _sample_weight = None

                    train = Dataset.load_arff("./data/"+dataset_name+".f"+str(f)+".train.arff", n_labels, endian = "big", input_feature_type = 'int', encode_nominal = True)
                    train_data = np.concatenate((train['X'],train['Y']), axis = 1)

                    if args.l:
                        l_vars = [i+train['X'].shape[1] for i in range(train['Y'].shape[1])]
                    else:
                        l_vars = []


                    if min_instances <= 1:
                        min_instances_ = int(train['X'].shape[0] * min_instances)+1
                        print("Setting min_instances to ", min_instances)
                    else:
                        min_instances_ = min_instances

                    learn_start_t = perf_counter()
                    C = mlcsn(train_data, 
                              sample_weight = _sample_weight,
                              min_instances=min_instances_, 
                              min_features=min_features, 
                              alpha=alpha, random_forest=rf,
                              leaf_vars = l_vars,
                              and_leaves = and_leaf,
                              and_inners = and_node,
                              sum_nodes = sum_nodes)

                    C.fit()
                    learn_end_t = perf_counter()

                    learning_time = (learn_end_t - learn_start_t)


                    test_data = Dataset.load_arff("./data/"+dataset_name+".f"+str(f)+".test.arff", n_labels, endian = "big", input_feature_type = 'int', encode_nominal = True)
                    test_start_t = perf_counter()
                    Y_pred = C.compute_predictions(test_data['X'], n_labels)
                    test_end_t = perf_counter()
                    testing_time = (test_end_t - test_start_t)

#                    Y1_pred = C.compute_predictions1(test_data['X'], n_labels)

                    Accuracy.append(sklearn.metrics.jaccard_similarity_score(test_data['Y'], Y_pred))
                    Hamming_score.append(1-sklearn.metrics.hamming_loss(test_data['Y'], Y_pred))
                    Exact_match.append(1-sklearn.metrics.zero_one_loss(test_data['Y'], Y_pred))
                    Learning_time.append(learning_time)
                    Testing_time.append(testing_time)
                    Headers.append("Fold "+ str(f))

#                    print("Accuracy ", sklearn.metrics.jaccard_similarity_score(test_data['Y'], Y1_pred))
#                    print("Hamming ",1-sklearn.metrics.hamming_loss(test_data['Y'], Y1_pred))
#                    print("Exact ", 1-sklearn.metrics.zero_one_loss(test_data['Y'], Y1_pred))
    
                    
                    or_nodes = C.or_nodes
                    n_sum_nodes = C.n_sum_nodes
                    and_nodes = C.and_nodes
                    leaf_nodes = C.leaf_nodes
                    or_edges = C.or_edges
                    clt_edges = C.clt_edges
                    cltrees = C.cltrees
                    clforests = C.clforests
                    depth = C.depth
                    mdepth = C.mdepth


                    """
                    #
                    # writing to file a line for the grid
                    stats = stats_format([alpha,
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
                    """
                    print(tabulate([Accuracy, Hamming_score, Exact_match, Learning_time, Testing_time], 
                                   headers=Headers, tablefmt='orgtbl'))

                print('\nAccuracy (mean/std)      :', np.mean(np.array(Accuracy[1:])),"/",np.std(np.array(Accuracy[1:])))
                print('Hamming score (mean/std) :', np.mean(np.array(Hamming_score[1:])), "/", np.std(np.array(Hamming_score[1:])))
                print('Exact match (mean/std)   :', np.mean(np.array(Exact_match[1:])), "/", np.std(np.array(Exact_match[1:])))
                print('Learning Time (mean/std)          :', np.mean(np.array(Learning_time[1:])), "/", np.std(np.array(Learning_time[1:])))
                print('Testing Time (mean/std)          :', np.mean(np.array(Testing_time[1:])), "/", np.std(np.array(Testing_time[1:])))


                out_log.write(tabulate([Accuracy, Hamming_score, Exact_match, Learning_time, Testing_time], 
                                       headers=Headers, tablefmt='orgtbl'))
                out_log.write('\n\nAccuracy (mean/std)      : %f / %f' % (np.mean(np.array(Accuracy[1:])),np.std(np.array(Accuracy[1:]))))
                out_log.write('\nHamming score (mean/std) : %f / %f' % (np.mean(np.array(Hamming_score[1:])), np.std(np.array(Hamming_score[1:]))))
                out_log.write('\nExact match (mean/std)   : %f / %f' % (np.mean(np.array(Exact_match[1:])), np.std(np.array(Exact_match[1:]))))
                out_log.write('\nLearning Time (mean/std) : %f / %f' % (np.mean(np.array(Learning_time[1:])), np.std(np.array(Learning_time[1:]))))
                out_log.write('\nTesting Time (mean/std)  : %f / %f' % (np.mean(np.array(Testing_time[1:])), np.std(np.array(Testing_time[1:]))))
                out_log.flush()
