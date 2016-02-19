import numpy as np
import argparse
import arff
from discretize import LAIMdiscretize
from cross_validation import StratifiedKFold
from dataset import Dataset
import shutil


def unique_rows(data,c):
    dict = {}
    for row in data:
        dict[tuple(row[c:])]=row[0:c]
    A = []
    for k in dict:
        A.append(list(dict[k])+list(k))
    return np.array(A)
        
    
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, nargs=1,
                    help='Specify a dataset name from data/ (es. nltcs)')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('-b', action='store_true', default=False,
                    help='Whether the ARFF file contains labels at the beginning of the attributes list.')

parser.add_argument('-c', type=int, nargs=1, default=1,
                    help='Number of class labels.')

parser.add_argument('-k', type=int, nargs=1, default=1,
                    help='Number of folds.')

parser.add_argument('-s', action='store_true', default=False,
                    help='Shuffle for cross validation.')




args = parser.parse_args()
n_labels = args.c[0]
endian_big = args.b
n_folds = args.k[0]
shuffle = args.s
(dataset_name_,) = args.dataset

dataset_name = "data/" + dataset_name_

shutil.copy(dataset_name + ".orig.arff", dataset_name + ".arff")

# first in big endian
if endian_big == False:
    Dataset.arff_to_big_endian(dataset_name + ".arff", dataset_name_, n_labels)

# Discretize the dataset

data = Dataset.load_arff(dataset_name + ".arff", n_labels, endian = "big", input_feature_type = 'float', encode_nominal = True)
D = LAIMdiscretize(data)
D.discretize()

discretized_data_matrix = np.concatenate((data['Y'],D.X_discretized), axis=1)

Uniques = unique_rows(discretized_data_matrix,data['Y'].shape[1])

print("Unique ", discretized_data_matrix.shape[0], Uniques.shape[0])

data_frame = arff.load(open(dataset_name + ".arff", 'r'), encode_nominal = True, return_type=arff.DENSE)
data_frame['data'] = discretized_data_matrix.astype(int).tolist()
# make the attributes nominal
for i in range(len(data_frame['attributes'])):
    (attr_name, attr_value) = data_frame['attributes'][i]
    data_frame['attributes'][i] = (attr_name, ['0', '1'])

discretized_dataset = dataset_name + ".discr.arff"
f = open(discretized_dataset, "w")
arff.dump(data_frame, f)
f.close()

discretized_data = {}
discretized_data['X'] = D.X_discretized
discretized_data['Y'] = data['Y']


SKF = StratifiedKFold(discretized_data,  n_folds, shuffle, args.seed)

(train_f, test_f) = SKF.run()


for k in range(n_folds):
    X_train = discretized_data['X'][train_f[k],:]
    Y_train = discretized_data['Y'][train_f[k],:]
    X_test = discretized_data['X'][test_f[k],:]
    Y_test = discretized_data['Y'][test_f[k],:]

    Dataset.dump_data_arff(discretized_dataset, dataset_name + ".f" + str(k) + ".train.arff", X_train, Y_train)
    Dataset.dump_data_arff(discretized_dataset, dataset_name + ".f" + str(k) + ".test.arff", X_test, Y_test)

