# =============================================================================
# Univerity of Bari "Aldo Moro"
# Department of Computer Science
# Nicola Di Mauro 
# =============================================================================
# Copyright (c) 2016 Nicola Di Mauro, 
#   n1col4.d1m4uro un1b4.1t 
#   (replace 1 with i, 4 with a, and insert @ between o and u)

"""
 The discretize module implements discretization as reported in [1]. 
 It produces 2 intervals only.

 [1] LAIM discretization for multi-label data, 
     Alberto Canoa, José María Lunab, Eva L. Gibajab, Sebastián Ventura
"""

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

class LAIMdiscretize(object):

    def __init__(self, dataset, c, b=False):
        # Specify a dataset name from data/ (es. nltcs)
        self.dataset_name = dataset
        # Number of class labels
        self.n_labels = c
        # Indexes labels as the beginning attributes
        self.beginning_labels = b

    def run(self):
        data = arff.load(open(self.dataset_name+".arff", 'r'), encode_nominal=True)

        XY = np.array(data['data'])
        n_attributes = XY.shape[1]


        # put the labels at the end
        if self.beginning_labels == True:
            indexing = np.array([i for i in range(n_attributes - self.n_labels, n_attributes)] + [i for i in range(n_attributes - self.n_labels)])
            XY = XY[:,indexing]


        unique_dict = {}
        for attr in range(XY.shape[1]):
            unique_dict[attr] = np.unique(XY[:,attr])
        # check whether the labels have 0 1 value
        for attr in range(n_attributes - self.n_labels, n_attributes):
            if (len(unique_dict[attr])>2):
                print("Attribute", attr)
                raise BadLabelAttribute()

        #check for categorical attributes
        for (attr, domain) in data['attributes']:
            if isinstance(domain, list) and len(domain)>2:
                print("")
#                raise CategoricalAttribute()

        print("Discretizing", XY.shape[0], "instances, ", n_attributes - self.n_labels, "attributes, ", self.n_labels, "labels")

        # discretize
        discr_intervals = {}
        for i in range(n_attributes - self.n_labels):
            # check for numeric attribute
            if data['attributes'][i][1] == 'NUMERIC' or data['attributes'][i][1] == 'REAL':
                print("attribute", i, len(unique_dict[i]))
                max_LAIM = 0.0
                best_cut = 0.0
                for j in range(len(unique_dict[i])-1):
                    midpoint = (unique_dict[i][j+1] + unique_dict[i][j]) / 2
                    LAIM_value = self._compute_LAIM(unique_dict[i][0], unique_dict[i][-1], midpoint, XY, self.n_labels, i)
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

        f = open(dataset_name + ".discr.arff","w")
        arff.dump(data,f)
        f.close()


    def _compute_LAIM(self, l, r, midpoint, XY, n_labels,i):
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
        
