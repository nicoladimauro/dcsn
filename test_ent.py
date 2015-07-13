from csn_ent import csv_2_numpy, csn_entropy, csnm_entropy
import numpy as np
import argparse

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


def load_train_val_test_csvs(dataset,
                             path=DATA_PATH,
                             sep=',',
                             type='int',
                             suffixes=['.ts.data',
                                       '.valid.data',
                                       '.test.data']):
    """
    WRITEME
    """
    csv_files = [dataset + ext for ext in suffixes]
    return [csv_2_numpy(file, path, sep, type) for file in csv_files]


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
                    default='./exp/csn-ent/',
                    help='Output dir path')

parser.add_argument('-r', '--random', action='store_true', default=False,
                    help='Random Forest')


parser.add_argument('-p', '--perc', type=float, nargs='+',
                    default=[1.0],
                    help='Percentage for the bootstrap sample')

parser.add_argument('-n', '--n-components', type=int, nargs='+',
                    default=[1],
                    help='Min number of instances in a slice to split by cols')

parser.add_argument('-i', '--min-instances', type=int, nargs='+',
                    default=[10],
                    help='Min number of instances in a slice to split by cols')

parser.add_argument('--prune', action='store_true',
                    help='Post pruning on the validation set')

parser.add_argument('-a', '--alpha', type=float, nargs='+',
                    default=[1.0],
                    help='Smoothing factor for leaf probability estimation')

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
percs = args.perc
n_components = args.n_components
rf = args.random
m_instances = args.min_instances


# initing the random generators
seed = args.seed
numpy_rand_gen = numpy.random.RandomState(seed)
random.seed(seed)

#
# elaborating the dataset
#
logging.info('Loading datasets: %s', args.dataset)
(dataset_name,) = args.dataset
train, valid, test = load_train_val_test_csvs(dataset_name)
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

preamble = ("""comp:\talpha:\tminst:\tc_nodes:""" +
            """\ttime:""" +
            """\ttrain_ll\tvalid_ll:\ttest_ll\n""")

max_components = max(n_components)

with open(out_log_path, 'w') as out_log:

    out_log.write("parameters:\n{0}\n\n".format(args))
    out_log.write(preamble)
    out_log.flush()
    #
    # looping over all parameters combinations
    for alpha in alphas:
        for perc in percs:
            for min_instances in m_instances:

                learn_start_t = perf_counter()
                C = csnm_entropy(max_components=max_components,
                                 training_data=train,
                                 max_depth=9999,
                                 min_instances=min_instances,
                                 min_features=3,
                                 alpha=alpha,
                                 random_forest=rf,
                                 prune=args.prune,
                                 valid_data=valid)

                learn_end_t = perf_counter()

                learning_time = (learn_end_t - learn_start_t)

                # #
                # # pruning
                # # if args.prune:
                # #     C.prune(valid)

                # #
                # # gathering statistics
                # n_nodes = C.cut_nodes
                # #
                # # Compute LL on training set
                # logging.info('Evaluating on training set')
                # train_avg_ll = C.ll(train)

                # #
                # # Compute LL on validation set
                # logging.info('Evaluating on validation set')
                # valid_avg_ll = C.ll(valid)

                # #
                # # Compute LL on test set
                # logging.info('Evaluating on test set')
                # test_avg_ll = C.ll(test)

                # #
                # # updating best stats according to valid ll
                # if valid_avg_ll > best_valid_avg_ll:
                #     best_valid_avg_ll = valid_avg_ll
                #     best_state['alpha'] = alpha
                #     best_state['time'] = learning_time
                #     best_state['train_ll'] = train_avg_ll
                #     best_state['valid_ll'] = valid_avg_ll
                #     best_state['test_ll'] = test_avg_ll

                # #
                # # writing to file a line for the grid
                # stats = stats_format([alpha,
                #                       n_nodes,
                #                       learning_time,
                #                       train_avg_ll,
                #                       valid_avg_ll,
                #                       test_avg_ll],
                #                      '\t',
                #                      digits=5)
                # out_log.write(stats + '\n')
                # out_log.flush()
                for c in n_components:
                    #
                    # Compute LL on training set

                    out_filename = out_path + '/c' + str(c) + 'train.lls'
                    logging.info('Evaluating on training set')
                    train_avg_ll = C.ll(train, c, out_filename)

                    #
                    # Compute LL on validation set
                    out_filename = out_path + '/c' + str(c) + 'valid.lls'
                    logging.info('Evaluating on validation set')
                    valid_avg_ll = C.ll(valid, c, out_filename)

                    #
                    # Compute LL on test set
                    out_filename = out_path + '/c' + str(c) + 'test.lls'
                    logging.info('Evaluating on test set')
                    test_avg_ll = C.ll(test, c, out_filename)

                    #
                    # updating best stats according to valid ll
                    if valid_avg_ll > best_valid_avg_ll:
                        best_valid_avg_ll = valid_avg_ll
                        best_state['alpha'] = alpha
                        best_state['perc'] = perc
                        best_state['m_inst'] = min_instances
                        best_state['time'] = learning_time
                        best_state['train_ll'] = train_avg_ll
                        best_state['valid_ll'] = valid_avg_ll
                        best_state['test_ll'] = test_avg_ll

                    nodes = C.nodes(c)
                    #
                    # writing to file a line for the grid
                    stats = stats_format([c,
                                          alpha,
                                          min_instances,
                                          nodes,
                                          learning_time,
                                          train_avg_ll,
                                          valid_avg_ll,
                                          test_avg_ll],
                                         '\t',
                                         digits=5)
                    out_log.write(stats + '\n')
                    out_log.flush()

    #
    # writing as last line the best params
    out_log.write("{0}".format(best_state))
    out_log.flush()

logging.info('Grid search ended.')
logging.info('Best params:\n\t%s', best_state)
