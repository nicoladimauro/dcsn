import numpy as np
import argparse
import arff
from discretize import LAIMdiscretize
from cross_validation import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, nargs=1,
                    help='Specify a dataset name from data/ (es. nltcs)')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('-b', action='store_true', default=False,
                    help='Indexes labels as the beginning attributes.')

parser.add_argument('-c', type=int, nargs=1, default=1,
                    help='Number of class labels.')

parser.add_argument('-k', type=int, nargs=1, default=1,
                    help='Number of folds.')

parser.add_argument('-s', action='store_true', default=False,
                    help='Shuffle for cross validation.')


args = parser.parse_args()

# Discretize the dataset

(dataset_name,) = args.dataset



D = LAIMdiscretize(dataset_name, args.c[0], args.b)
D.run()

discretized_dataset = dataset_name + ".discr"

data = arff.load(open(discretized_dataset+".arff", 'r'), encode_nominal=True)

XY = np.array(data['data'])

SKF = StratifiedKFold(XY, args.c[0], args.k[0], args.s, args.seed)

(train_f, test_f) = SKF.run()

for k in range(args.k[0]):
    TRAIN = XY[train_f[k],:]
    TEST = XY[test_f[k],:]

    data['data'] = TRAIN
    f = open(dataset_name + ".f" + str(k) + ".train.arff","w")
    arff.dump(data,f)
    f.close()

    data['data'] = TEST
    f = open(dataset_name + ".f" + str(k) + ".test.arff","w")
    arff.dump(data,f)
    f.close()
