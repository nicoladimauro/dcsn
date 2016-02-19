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

    def __init__(self, data):
        # Specify a dataset name from data/ (es. nltcs)
        self.data = data
        # Number of class labels
        self.n_labels = self.data['Y'].shape[1]
        self.X_discretized = np.zeros((self.data['X'].shape[0],self.data['X'].shape[1]))

    def discretize(self):

        unique_dict = {}
        for attr in range(self.data['X'].shape[1]):
            unique_dict[attr] = np.unique(self.data['X'][:,attr])

        # NOTE: all the attributes are supposed to be numeric, real or categorical with numeric values

        print("Discretizing", self.data['X'].shape[0], "instances, ", self.data['X'].shape[1], "attributes, ", self.n_labels, "labels")

        # discretize
        discr_intervals = {}
        for i in range(self.data['X'].shape[1]):

            print("attribute", i, len(unique_dict[i]), end=" ")
            max_LAIM = 0.0
            best_cut = 0.0
            for j in range(len(unique_dict[i])-1):
                midpoint = (unique_dict[i][j+1] + unique_dict[i][j]) / 2
                LAIM_value = self._compute_LAIM(unique_dict[i][0], unique_dict[i][-1], \
                                                    midpoint, i)
                if LAIM_value > max_LAIM:
                    best_cut = midpoint
                    max_LAIM = LAIM_value
            print ("[",unique_dict[i][0],",",best_cut,"] [",best_cut,",",unique_dict[i][-1],"]")

            for r in range(self.data['X'].shape[0]):
                if self.data['X'][r,i] > best_cut:
                    self.X_discretized[r,i] = 1


    def _compute_LAIM(self, l, r, midpoint, i):
        quanta_matrix = np.zeros((self.n_labels,2))
        for k in range(self.n_labels):
            quanta_matrix[k][0]=np.sum(np.logical_and(self.data['X'][:,i]<=midpoint,self.data['Y'][:,k]==1))
            quanta_matrix[k][1]=np.sum(np.logical_and(self.data['X'][:,i]>midpoint,self.data['Y'][:,k]==1))

        m = np.sum(quanta_matrix[:,0])
        if m == 0:
            return -1
        sum = pow(np.max(quanta_matrix[:,0]),2) / m
        m = np.sum(quanta_matrix[:,1])
        if m == 0:
            return -1
        sum += pow(np.max(quanta_matrix[:,1]),2) / m
        return sum / (self.n_labels * np.sum(quanta_matrix))
        
