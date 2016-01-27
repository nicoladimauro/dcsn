import arff
import argparse
import numpy as np

class DiscretizeException(Exception):
    message = None
    
    def __str__(self):
        return self.message

class CategoricalAttribute(DiscretizeException):
    '''Error raised when some attribute has categorical values.'''
    message = 'Categorical @ATTRIBUTE.'

class BadLabelAttribute(DiscretizeException):
    message = 'Label attribute with more than two values @ATTRIBUTE.'


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, nargs=1,
                    help='Specify a dataset name from data/ (es. nltcs)')
parser.add_argument('-c', type=int, nargs=1,
                    help='Number of class labels.')
parser.add_argument('-b', action='store_true', default=False,
                    help='Indexes labels as the beginning attributes.')


args = parser.parse_args()

(dataset_name,) = args.dataset
(n_labels,) = args.c
beginning_labels = args.b

data = arff.load(open(dataset_name, 'r'),encode_nominal=True)

print(dataset_name, n_labels, beginning_labels)

XY = np.array(data['data'])
n_attributes = XY.shape[1]


def compute_LAIM(l, r, midpoint, XY, n_labels,i):
    quanta_matrix = np.zeros((n_labels,2))
    k = 0
    for l in range(XY.shape[1]-n_labels,XY.shape[1]):
        quanta_matrix[k][0]=np.sum(np.logical_and(XY[:,i]<=midpoint,XY[:,l]==1))
        quanta_matrix[k][1]=np.sum(np.logical_and(XY[:,i]>midpoint,XY[:,l]==1))
        k += 1

    m = np.sum(quanta_matrix[:,0])
    if m == 0:
        return -1
    sum = pow(np.max(quanta_matrix[:,0]),2) / m
    m = np.sum(quanta_matrix[:,1])
    if m == 0:
        return -1
    sum += pow(np.max(quanta_matrix[:,1]),2) / m
    return sum / (n_labels * np.sum(quanta_matrix))
        
    

f = open("prova.arff","w")
arff.dumps(data)


unique_dict = {}
for attr in range(XY.shape[1]):
    unique_dict[attr] = np.unique(XY[:,attr])
# check whether the labels have 0 1 value
for attr in range(n_attributes-n_labels,n_attributes):
    if (len(unique_dict[attr])>2):
        print("Attribute", attr)
        raise BadLabelAttribute()
#check for categorical attributes
for (attr, domain) in data['attributes']:
    if isinstance(domain, list) and len(domain)>2:
        print(attr, domain)
#        raise CategoricalAttribute()
# discretize
discr_intervals = {}
#for i in range(n_attributes-n_labels):
for i in range(10):
    # check for numeric attribute
    if data['attributes'][i][1] == 'NUMERIC' or data['attributes'][i][1] == 'REAL':
        print("attribute", i, end="")
        max_LAIM = 0.0
        best_cut = 0.0
        for j in range(len(unique_dict[i])-1):
            midpoint = (unique_dict[i][j+1] + unique_dict[i][j])/2
            LAIM_value = compute_LAIM(unique_dict[i][0], unique_dict[i][-1], midpoint, XY, n_labels,i)
            if LAIM_value > max_LAIM:
                best_cut = midpoint
                max_LAIM = LAIM_value
        (attr_name, val) = data['attributes'][i]
        data['attributes'][i] = (attr_name, ['0', '1'])
        for r in range(XY.shape[0]):
            if data['data'][r][i] <= best_cut:
                data['data'][r][i] = 0
            else:
                data['data'][r][i] = 1



