import numpy as np
import random

class StratifiedKFold():
    
    def __init__(self, X, n_labels, n_folds=5, shuffle=False, seed=0):
        numpy_rand_gen = np.random.RandomState(seed)
        random.seed(seed)

        if shuffle == True:
            np.random.shuffle(X)

        n_attributes = X.shape[1]-n_labels
        # find the less representative label
        l_min = np.sum(X[:,n_attributes])
        l_repr = n_attributes
        for i in range(n_attributes+1, X.shape[1]):
            l_sum = np.sum(X[:,i])
            if l_sum < l_min:
                l_min = l_sum
                l_repr = i
                
        (A,) = np.where(X[:, l_repr] == 1)
        (B,) = np.where(X[:, l_repr] == 0)

        folds_train = [ None for i in range(n_folds)] 
        folds_test = [ None for i in range(n_folds)] 

        for k in range(n_folds):
            folds_train[k] = [x for i,x in enumerate(A) if i % n_folds != k] + \
                             [x for i,x in enumerate(B) if i % n_folds != k]
            folds_test[k] = [x for i,x in enumerate(A) if i % n_folds == k] + \
                            [x for i,x in enumerate(B) if i % n_folds == k]

        self.training_folds = np.array(folds_train)
        self.testing_folds = np.array(folds_test)

       
