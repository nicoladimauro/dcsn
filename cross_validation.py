import numpy as np
import random

class StratifiedKFold():
    
    def __init__(self, X, n_labels, n_folds=5, shuffle=False, seed=0):
        self.X = X
        self.n_labels = n_labels
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.seed = seed

    def run(self):
        numpy_rand_gen = np.random.RandomState(self.seed)
        random.seed(self.seed)

        if self.shuffle == True:
            np.random.shuffle(self.X)

        n_attributes = self.X.shape[1] - self.n_labels
        # find the less representative label
        l_min = np.sum(self.X[:,n_attributes])
        l_repr = n_attributes
        for i in range(n_attributes + 1, self.X.shape[1]):
            l_sum = np.sum(self.X[:,i])
            if l_sum < l_min:
                l_min = l_sum
                l_repr = i
                
        (A,) = np.where(self.X[:, l_repr] == 1)
        (B,) = np.where(self.X[:, l_repr] == 0)

        folds_train = [ None for i in range(self.n_folds)] 
        folds_test = [ None for i in range(self.n_folds)] 

        for k in range(self.n_folds):
            folds_train[k] = [x for i,x in enumerate(A) if i % self.n_folds != k] + \
                             [x for i,x in enumerate(B) if i % self.n_folds != k]
            folds_test[k] = [x for i,x in enumerate(A) if i % self.n_folds == k] + \
                            [x for i,x in enumerate(B) if i % self.n_folds == k]

        return( np.array(folds_train), np.array(folds_test))

       
