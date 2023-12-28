import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from numba import jit, njit
from sklearn.model_selection import KFold
import multiprocessing as mp


def poly_kernel(data, vec, d):
    '''
    INPUTS:
    data:       (#points,#pixels) matrix of data points (labels taken off)
    vec:        (#pixels,) vector to calculate kernel of
    d:          polynomial exponent
    OUTPUT:
    val:        (#points,) vector of kernel values with each training point
    '''

    val = (data @ vec.reshape(-1,1))**d
    return val.ravel()


def class_pred(data, vec, alphas, d):
    '''
    INPUTS:
    data:       (#points,#pixels) matrix of data points (labels taken off)
    vec:        (#pixels,) vector to calculate kernel of
    d:          polynomial exponent
    alphas:     (#labels,#points) matrix of values of alpha in the dual problem
    OUTPUT:
    preds:      (#labels,) vector of predictions for each label 
    '''
    kervals = poly_kernel(data,vec,d)
    preds = alphas @ kervals.reshape(-1,1)
    return preds.ravel()


def init_alphas(data):
    alphas = np.zeros((10,int(len(data))))
    return alphas


def cross_val(data, d, k=5):
    kf = KFold(n_splits=k)
    score = np.zeros(k)
    
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        train_cv = data[train_index,:]
        test_cv = data[test_index,:]

        alpha_cv = init_alphas(train_cv)
            
        _, alpha_cv = train_perceptron(train_cv, alpha_cv, d+1)
        test_error = test_perceptron(train_cv, test_cv, alpha_cv, d+1)

        score[i] = test_error
    
    return score.mean()
     
     
#training algorithm:
#have num_label individual classifiers which each classify whether it is or isnt that label
#on each point, we see how the classifier predicts, and update each classifier's coefficients which made a mistake
#the overall classifier's prediction is the classifier with the largest prediction 

def train_perceptron(data, alphas, d):
    '''
    INPUTS:
    data:        (#points,#pixels+1) matrix of data points
    alphas:      (#labels,#points) matrix of values of alpha in the dual problem
    d:           polynomial exponent
    OUTPUT:
    error_rate, alphas
    '''
    num_labels = len(set(data[:,0]))
    mistakes = 0
    #for each training point
    for i in range(len(data)):
        label = data[i,0]

        #obtain prediction made by each classifier
        preds = class_pred(data[:,1:],data[i,1:],alphas,d)
        preds_binary = np.where(preds <= 0, -1, 1)

        #check which classifier made a mistake
        truth_col = -np.ones(num_labels)
        truth_col[int(label)] = 1
        is_pred_wrong = (preds_binary != truth_col).astype(np.int32)          #a vector which has a 1 if the kth classifier was wrong

        #update alpha
        alphas[:,i] -= is_pred_wrong*preds_binary

        #add mistake
        if np.argmax(preds) != label:
            mistakes += 1
    
    error_rate = mistakes/len(data)
    return error_rate, alphas


#testing algorithm
def test_perceptron(train, test, alphas, d, calc_conf=False):
    '''

    '''
    mistakes = 0
    conf_mat = np.zeros((10,10))
    
    for i in range(len(test)):
        label = test[i,0]
        preds = class_pred(train[:,1:],test[i,1:],alphas,d)
        if int(np.argmax(preds)) != int(label):
            mistakes += 1
            if calc_conf:
                conf_mat[int(label),int(np.argmax(preds))] += 1

    #normalise conf_mat
    if calc_conf:
        label_counts = np.bincount(test[:,0].astype(int),minlength=10)
        label_counts[label_counts==0] = 1
        label_counts = label_counts.reshape(-1,1)

        #turn it into a rate
        conf_mat = conf_mat / label_counts

    error_rate = mistakes/len(test)

    if calc_conf:
        return error_rate, conf_mat  
    else:
        return error_rate


def train_with_d_star(data):
    #split training and test
    shuffled_data = np.random.permutation(data)

    split_idx = int(len(data) * 0.8)

    train = shuffled_data[:split_idx, :]
    test = shuffled_data[split_idx:, :]

    # #cross validate to find best d
    test_score = np.array([cross_val(train, d) for d in range(1,8)])
    
    d_star = np.argmin(test_score) + 1

    #train perceptron with d_star
    alpha_list = init_alphas(train)

    _, alpha_list = train_perceptron(train, alpha_list, d_star)
    test_error, conf_mat = test_perceptron(train, test, alpha_list, d_star, calc_conf=True)

    return d_star, test_error, conf_mat