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

        print(l_repr,A,B)

        folds = [ [] for i in range(n_folds)] 

        n_A = int(len(A) / n_folds)
        if n_A == 0:
            n_A = 1
        

        n_B = int(len(B) / n_folds)
        if n_B == 0:
            n_B = 1

        f = 0
        for i in range(len(A)):
            folds[f].append(A[i])
            if i % n_A == 0:
                f += 1
            print (folds, i , f)
                
        f = 0
        for i in range(len(B)):
            folds[f].append(B[i])
            if i % n_B == 0:
                f += 1
            print (folds, i , f)

        print (folds)
        

        
A = np.random.randint(2,size=(10,10))
print(A)
C = StratifiedKFold(A, 3)
