from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random

import scipy
import sys

dataDir = '/u/cs401/A3/data/'
# dataDir = "/mnt/c/Users/j9108c/BitTorrent Sync/school/UofT/CSC401/a3/test/cs401/data/"


class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta. See equation 1 of the handout.

    As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function. If you do this, you pass that precomputed component in preComputedForM.
    '''
    d = myTheta.mu.shape[1] # number of dimensions (d = 13)
    mu_m = myTheta.mu[m] # slides 2 pg.24 mu_m
    sig_sq_m = myTheta.Sigma[m] # slides 2 pg.24 sigma_squared_m
    x_t = x # slides 2 pg.24 x_t

    term_1 = np.sum((x_t - mu_m)**2 / (2*sig_sq_m)) if len(x.shape)==1 else np.sum((x_t - mu_m)**2 / (2*sig_sq_m), axis=1)
    term_2 = (d/2) * np.log(2 * np.pi)
    term_3 = (1/2) * np.log(np.prod(sig_sq_m))

    return -term_1-term_2-term_3

    
def log_p_m_x(m, x, myTheta, log_Bs=None):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta. See equation 2 of handout.
    '''
    M = myTheta.mu.shape[0]
    omegas = myTheta.omega
    omega_m = omegas[m]

    if (log_Bs is None):
        log_Bs = np.array([log_b_m_x(i, x, myTheta) for i in range(M)])

    log_omegas = np.log(omegas)
    term_1 = np.log(omega_m)
    term_2 = log_b_m_x(m, x, myTheta)
    term_3 = scipy.special.logsumexp(log_omegas+log_Bs, axis=0)

    return term_1+term_2-term_3

    
def logLik(log_Bs, myTheta):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x.

    X can be training data, when used in train( ... ), and
    X can be testing data, when used in test( ... ).

    We don't actually pass X directly to the function because we instead pass:

    log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

    See equation 3 of the handout.
    '''
    omegas = myTheta.omega
    log_omegas = np.log(omegas)

    return np.sum(scipy.special.logsumexp(log_omegas+log_Bs, axis=0))

    
def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma).
    '''
    # init theta (slides 2 pg.19)
    T, d = X.shape
    myTheta = theta(speaker, M, d)
    myTheta.mu = X[np.random.choice(T, M, replace=False)]
    myTheta.Sigma[:] = 1
    myTheta.omega[:] = 1/M

    i = 0
    prev_L = -np.inf
    improvement = np.inf
    while (i <= maxIter and improvement >= epsilon):
        # compute intermediate results
        log_Bs = np.array([log_b_m_x(i, X, myTheta) for i in range(M)])
        log_Ps = np.array([log_p_m_x(i, X, myTheta, log_Bs) for i in range(M)])

        L = logLik(log_Bs, myTheta) # compute likelihood

        # update parameters (slides 2 pg.17)
        for j in range(M):
            myTheta.omega[j] = np.sum(np.exp(log_Ps[j])) / T
            myTheta.mu[j] = (np.exp(log_Ps[j]) @ X) / np.sum(np.exp(log_Ps[j]))
            myTheta.Sigma[j] = (np.exp(log_Ps[j]) @ X**2) / np.sum(np.exp(log_Ps[j])) - myTheta.mu[j]**2

        improvement = L - prev_L
        prev_L = L
        i += 1

    return myTheta


def test(mfcc, correctID, models, k=5):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'.

    If k>0, print to stdout the actual (correct) speaker and the k best likelihoods in this format:
            [ACTUAL_ID]
            [SNAME1] [LOGLIK1]
            [SNAME2] [LOGLIK2]
            ...
            [SNAMEK] [LOGLIKK] 

    e.g.,
            S-5A -9.21034037197

    the format of the log likelihood (number of decimal places, or exponent) does not matter.
    '''
    bestModel = -1
    
    names_lls = {}
    M = 0
    for model in models:
        M = len(model.omega)
        log_Bs = np.array([log_b_m_x(i, mfcc, model) for i in range(M)])
        names_lls[model.name] = logLik(log_Bs, model)

    if (k > 0):
        actual_speaker = models[correctID].name
        print(actual_speaker)
        descending = sorted(names_lls.items(), key=lambda x: x[1], reverse=True) # sort dict by values, descending

        if (descending[0][0] == actual_speaker):
            bestModel = correctID

        for i in range(k):
            print(f"{descending[i][0]} {descending[i][1]}")

    # print("\t\t\t\t\tTRUE") if bestModel==correctID else print("\t\t\t\t\t\tFALSE")
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":
    trainThetas = []
    testMFCCs = []

    # print('you will need to modify this main block for Sec 2.4')
    d = 13
    S = 32 # number of possible speakers. default: 32
    k = 5 # number of top speakers to display, <= 0 if none. default: 5
    M = 8 # number of components. default: 8
    epsilon = 0.0 # convergence threshold. default: 0.0
    maxIter = 20 # max training iterations. default: 20

    # train a model for each speaker, and reserve data for testing
    i = 1
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            if (i <= S):
                # print(i)
                print(speaker)

                files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
                random.shuffle(files)
                
                testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
                testMFCCs.append(testMFCC)

                X = np.empty((0, d))
                for file in files:
                    myMFCC = np.load(os.path.join(dataDir, speaker, file))
                    X = np.append(X, myMFCC, axis=0)

                trainThetas.append(train(speaker, X, M, epsilon, maxIter))
            i += 1

    # evaluate 
    numCorrect = 0
    for i in range(0,len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k) 
    accuracy = 1.0*numCorrect/len(testMFCCs)
