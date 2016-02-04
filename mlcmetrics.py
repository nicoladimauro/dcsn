"""
Evaluation Metrics

l_ are loss/error measures (less is better)
p_ are payoff/accuracy measures (higher is better)

"""
import numpy as np


def extract_true_labels(X, n_labels):
    return X[:,-n_labels:]

def compute_predictions(C, X, n_labels):
    predictions = np.zeros((X.shape[0],n_labels),dtype=np.int)
    k = 0
    for x in X:
        evidence = {}
        for i in range(len(x)-n_labels):
            evidence[i]=x[i]
        (state, prob) = C.mpe(evidence = evidence)
        sum = 0
        for i in range(len(x)-n_labels,len(x)):
            sum += state[i]
        if sum == 0:
            # avoiding empty predictions
            max_state = None
            max_prob = -np.inf
            for i in range(len(x)-n_labels, len(x)):
                evidence[i] = 1
                if i > (len(x) - n_labels):
                    del evidence[i-1]
                (state1, prob1) = C.mpe(evidence = evidence)
                if (prob1 > max_prob):
                    max_prob = prob1
                    max_state = state1
            state = max_state
            prob = max_prob
        y = 0
        for i in range(len(x)-n_labels,len(x)):
            predictions[k,y]=state[i]
            y += 1
        k += 1
    return predictions
    
"""
 Exact Match, i.e., 1 - [0/1 Loss]
"""
def p_exact_match(Y, Y_pred):
    return 1.0 - l_zero_one(Y,Y_pred)

"""
0/1 Loss
"""
def l_zero_one(Y, Y_pred):
    loss = 0.0
    m = Y.shape[1]
    for i in range(Y.shape[0]):
        l = 0.0
        for j in range(m):
            if Y[i,j]!=Y_pred[i,j]:
                l = 1.0
                break
        loss += l
    return loss/Y.shape[0]

""" 
Hamming loss
"""

def l_hamming_loss(Y, Y_pred):
    loss = 0.0
    m = Y.shape[1]
    for i in range(Y.shape[0]):
        l = 0.0
        for j in range(m):
            if Y[i,j]!=Y_pred[i,j]:
                l += 1.0
        loss += l/m
    return loss/Y.shape[0]
    
""" 
Hamming score aka label accuracy
"""
def p_hamming(Y, Y_pred):
    return 1.0 - l_hamming_loss(Y, Y_pred)


"""
Jaccard Index -- often simply called multi-label 'accuracy'
"""

def p_accuracy(Y, Y_pred):
    c = 0.0
    m = Y.shape[1]
    for i in range(Y.shape[0]):
        union = 0
        inter = 0
        for j in range(m):
            if (Y[i,j]==1 and Y_pred[i,j]==1):
                inter += 1
            if (Y[i,j]==1 or Y_pred[i,j]==1):
                union +=1
        # = intersection / union; (or, if both sets are empty, then = 1.)
        if (union > 0):
            c += inter / union
        else:
            c += 1.0
    return c / Y.shape[0]


"""
True Positives 
"""
def p_true_positive(y, y_pred):
    return np.sum(np.logical_and(y_pred==1,y==1))

"""
False Positives 
"""
def p_false_positive(y, y_pred):
    return np.sum(np.logical_and(y_pred==1,y==0))

"""
True Negatives
"""
def p_true_negative(y, y_pred):
    return np.sum(np.logical_and(y_pred==0,y==0))

"""
False Negatives
"""
def p_false_negative(y, y_pred):
    return np.sum(np.logical_and(y_pred==0,y==1))

"""
Precision
"""
def p_precision(y, y_pred):
    tp = p_true_positive(y, y_pred)
    fp = p_false_positive(y, y_pred)
    if (tp == 0 and fp == 0):
        return 0.0
    return tp / (tp + fp)

"""
Precision
"""
def p_recall(y, y_pred):
    tp = p_true_positive(y, y_pred)
    fn = p_false_negative(y, y_pred)
    if (tp == 0 and fn == 0):
        return 0.0
    return tp / (tp + fn)

"""
F1
"""
def p_F1(y, y_pred):
    p = p_precision(y, y_pred)
    r = p_recall(y, y_pred)
    if (p == 0.0 and r == 0.0):
        return 0.0
    return 2 * p * r / (p + r)


"""
Precision Instances
"""
def p_precision_instances(Y, Y_pred):
    m = 0.0
    for i in range(Y.shape[0]):
        m += p_precision(Y[i],Y_pred[i]) 
    return m / Y.shape[0]

"""
Recall Instances
"""
def p_recall_instances(Y, Y_pred):
    m = 0.0
    for i in range(Y.shape[0]):
        m += p_recall(Y[i],Y_pred[i]) 
    return m / Y.shape[0]

"""
F1 Instances
"""
def p_F1_instances(Y, Y_pred):
    m = 0.0
    for i in range(Y.shape[0]):
        m += p_F1(Y[i],Y_pred[i]) 
    return m / Y.shape[0]

"""
Recall Macro
"""
def p_recall_macro(Y, Y_pred):
    m = 0.0
    for i in range(Y.shape[1]):
        m += p_recall(Y[:,i],Y_pred[:,i]) / Y.shape[0]
    return m


"""
Precision Macro
"""
def p_precision_macro(Y, Y_pred):
    m = 0.0
    for i in range(Y.shape[1]):
        m += p_precision(Y[:,i],Y_pred[:,i]) / Y.shape[0]
    return m

"""
Recall Macro
"""
def p_recall_macro(Y, Y_pred):
    m = 0.0
    for i in range(Y.shape[1]):
        m += p_recall(Y[:,i],Y_pred[:,i]) / Y.shape[0]
    return m



"""
Precision Micro
"""
def p_precision_micro(Y, Y_pred):
    return p_precision(Y.flatten(), Y_pred.flatten())

"""
Recall Micro
"""
def p_recall_micro(Y, Y_pred):
    return p_recall(Y.flatten(), Y_pred.flatten())

"""
F1 Micro
"""
def p_F1_micro(Y, Y_pred):
    return p_F1(Y.flatten(), Y_pred.flatten())









def subset_accuracy(C, X, n_labels):
    c = 0
    for x in X:
        evidence = {}
        for i in range(len(x)-n_labels):
            evidence[i]=x[i]
        (state, prob) = C.mpe(evidence = evidence)
        Z = []
        for i in range(len(x)-n_labels,len(x)):
            Z.append(state[i])
        Y = x[-n_labels:]
        for i in range(n_labels):
            if Z[i]==Y[i]:
                c +=1
    return (c / X.shape[0])

