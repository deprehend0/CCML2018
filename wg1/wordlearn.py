import numpy as np
import sklearn
import pylab as plt

# each row is a trial, comprised of a subset of word (object) indices
# sampled from the entire vocabulary
hiCDord = np.genfromtxt('freq369-3x3hiCD.txt', delimiter='\t')
loCDord = np.genfromtxt('freq369-3x3loCD.txt', delimiter='\t')


# define a function that accepts an array of trials (e.g., hiCDord)
# and returns a word x object co-occurrence matrix
def coocMatrix(ord):
    nw = len(np.unique(ord))
    no = len(np.unique(ord))
    M = np.zeros(shape=(nw, no))

    for trial in range(0, len(ord)):
        for i in range(0, len(ord[trial])):
            for j in range(0, len(ord[trial])):
                row = int(ord[trial][i]) - 1
                col = int(ord[trial][j]) - 1
                M[row][col] += 1

    return M


def plotCoocFrequencies(M, title):
    plt.pcolormesh(M)
    plt.title(title)
    plt.show()


hiCooc = coocMatrix(hiCDord)
loCooc = coocMatrix(loCDord)


# define a test function that accepts a 'memory' matrix
# --containing word-object hypotheses or associations--
# and a decision parameter, and returns choice probabilities
# of each object, given each word, according to softmax:
# https://en.wikipedia.org/wiki/Softmax_function
# (decision parameter = RL 'temperature')
def softmax(M, temp):
    x_exp = np.exp((M - np.max(M)) / temp)
    return x_exp / x_exp.sum(axis=0)


# define a function that accepts an array of trials and parameter
# values, and returns a memory matrix with the learned representation
def model(ord, par):
    M = np.zeros(shape=(len(np.unique(ord)), len(np.unique(ord)))) + 0.00000000001
    # learning: i.e., not just co-occurrence counting,
    # but a process that corresponds to what you think
    # people might be doing as they go through the trials
    # (guess-and-test hypothesis generation? biased association?)

    for trial in ord:
        for i in range(0, len(trial)):
            for j in range(0, len(trial)):
                col = int(trial[j] - 1)
                row = int(trial[i] - 1)
                u, c = np.unique(trial, return_counts=True)
                other_words = M[col, :]
                other_words[row] = 0

                M[row][col] += (c[i] / len(trial)) / sum(other_words)

    return M  # this matrix will then be passed through softmax to extract pr(correct)
# define a function that accepts a vocabulary (1:M), a
# distribution over the likelihood of sampling each word (object)
# and a desired number and size of trials, and returns a trial order
# (e.g., to accomplish simulations like those in Blythe et al., (2016))


# graph the mean performance for different softmax parameter values (e.g., .1 to 10)
# http://matplotlib.org/users/pyplot_tutorial.html
# you can first try feeding a co-occurrence matrix through softmax,
# and then try your cognitive model's output
def plot_performance_by_temperature(ord):
    temp = np.arange(.1, 10.2, .5)
    meanPerf = np.zeros(len(temp))
    # for each temp, call softMax(coocMatrix(ord)) and save the mean perforfmance
    for i in range(0, len(temp)):
        meanPerf[i] = np.mean(softmax(coocMatrix(ord), temp[i]))
        print(model(ord, 0))
    plt.plot(temp, meanPerf)
    plt.show()


plotCoocFrequencies(hiCooc, 'hiCooc')
hiModel = model(hiCDord, 0)
plotCoocFrequencies(hiModel, 'hiModel heatmap')
plot_performance_by_temperature(hiCDord)
plotCoocFrequencies(loCooc, 'loCooc')
loModel = model(loCDord, 0)
plotCoocFrequencies(loModel, 'loModel heatmap')
plot_performance_by_temperature(loCDord)


### Evaluating model fit ###

# try implementing each of the following three methods (SSE, crossEntropy,
# and negative log likelihood) and get a sense of their values for varying discrepancies of p and q
# human response probabilities for each correct of the 18 correct pairs:
# human_accuracy_variedCD.csv has columns hiCDacc and loCDacc

# implement sum of squared error measure of model fit (p) to observed data (q)
def SSE(p, q):
    mse = sklearn.metrics.mean_squared_error(q, p)
    return mse * len(q)


# implement cross entropy measure
def crossEntropy(observed_probs, model_probs):
    return -1 * sklearn.metrics.log_loss(observed_probs, model_probs)


# implement negative log-likelihood measure, assuming each test
# problem is binomial (since I didn't give you the full response matrix)
def negloglik(obs, mod):
    return sklearn.metrics.log_loss(obs, mod)


# implement a function (BIC) calculating the Bayesian Information Criterion
# https://en.wikipedia.org/wiki/Bayesian_information_criterion


### Fitting a model ###

# given a trial-ordering, model parameters, and a set of human Pr(correct),
# return your favorite goodness-of-fit value
def evaluateModel(parms, order, hum):
    return fitval

# write a function for optimizing parameter values (for a given trial-ordering
# and human Pr(correct)
# https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/optimize.html
