#!/usr/bin/python3

from meka import Meka
import numpy as np
import argparse
import sklearn.metrics
from tabulate import tabulate

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import datetime
from dataset import Dataset


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, nargs=1,
                    help='Specify a dataset name from data/ (es. nltcs)')

parser.add_argument('-f', type=int, nargs='?',
                    default=5,
                    help='Number of folds for the dataset')


parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/meka/',
                    help='Output dir path')

parser.add_argument('-mc', type=str, nargs='?',
                    default="meka.classifiers.multilabel.LC",
                    help='Meka classifier')

parser.add_argument('-wc', type=str, nargs='?',
                    default="weka.classifiers.bayes.NaiveBayes",
#                    default="weka.classifiers.bayes.BayesNet -- -D -Q weka.classifiers.bayes.net.search.local.TAN -- -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5",
                    help='Weka classifier')

parser.add_argument('-mp', type=str, nargs='?',
                    default='./meka/lib/',
                    help='Meka classpath')

parser.add_argument('-c', type=int, nargs=1,
                    help='Number of class labels.')


args = parser.parse_args()
(dataset_name,) = args.dataset

date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_path = args.output + dataset_name + '_' + date_string

Accuracy = ['Accuracy']
Hamming_score = ['Hamming Score']
Exact_match = ['Exact match']
Time = ['Time']
Headers = ['Metric']

for f in range(args.f):
    train_file_name = dataset_name + ".f" + str(f) + ".train.arff"
    test_file_name = dataset_name + ".f" + str(f) + ".test.arff"

    data = Dataset.load_arff("./data/"+test_file_name, args.c[0], endian = "big", input_feature_type = 'int', encode_nominal = True)

    meka = Meka(args.mc, args.wc, meka_classpath=args.mp)
    learn_start_t = perf_counter()
    predictions, statistics = meka.run("./data/"+train_file_name, "./data/" + test_file_name)
    learn_end_t = perf_counter()
    learning_time = (learn_end_t - learn_start_t)


    print("Accuracy :     :", statistics['Accuracy'])
    print('Hammingloss    :', statistics['Hammingloss'])
    print('Exactmatch', statistics['Exactmatch'])
    print('BuildTime', statistics['BuildTime'])
    print('TestTime', statistics['TestTime'])
    """
    print("Accuracy score ", sklearn.metrics.jaccard_similarity_score(data['Y'], predictions))
    print("Hamming loss ", sklearn.metrics.hamming_loss(data['Y'], predictions))
    print("zeroOneLoss ", sklearn.metrics.zero_one_loss(data['Y'], predictions))
    """

    Accuracy.append(sklearn.metrics.jaccard_similarity_score(data['Y'], predictions))
    Hamming_score.append(1-sklearn.metrics.hamming_loss(data['Y'], predictions))
    Exact_match.append(1-sklearn.metrics.zero_one_loss(data['Y'], predictions))
    Time.append(learning_time)
    Headers.append("Fold "+ str(f))
    

print(tabulate([Accuracy, Hamming_score, Exact_match, Time], 
               headers=Headers, tablefmt='orgtbl'))

print('\nAccuracy (mean/std)      :', np.mean(np.array(Accuracy[1:])),"/",np.std(np.array(Accuracy[1:])))
print('Hamming score (mean/std) :', np.mean(np.array(Hamming_score[1:])), "/", np.std(np.array(Hamming_score[1:])))
print('Exact match (mean/std)   :', np.mean(np.array(Exact_match[1:])), "/", np.std(np.array(Exact_match[1:])))
print('Time (mean/std)          :', np.mean(np.array(Time[1:])), "/", np.std(np.array(Time[1:])))

