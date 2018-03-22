

# each row is a trial, comprised of a subset of word (object) indices 
# sampled from the entire vocabulary
hiCDord = read.csv('freq369-3x3hiCD.txt')
loCDord = read.csv('freq369-3x3loCD.txt')


# define a function that accepts an array of trials (e.g., hiCDord)
# and returns a word x object co-occurrence matrix
coocMatrix <- function(ord) {
  # nw = number of words
  # no = number of objects
  M = matrix(0, nrow=nw, ncol=nw)
  # counting...
  return(M)
}

# define a test function that accepts a 'memory' matrix
# --containing word-object hypotheses or associations--
# and a decision parameter, and returns choice probabilities
# of each object, given each word, according to softmax:
# https://en.wikipedia.org/wiki/Softmax_function


# define a function that accepts an array of trials and parameter
# values, and returns a memory matrix with the learned representation
model <- function(ord) {
  # nw = number of words
  # no = number of objects
  M = matrix(0, nrow=nw, ncol=nw)
  # learning...
  return(M)
}

# define a function that accepts a vocabulary (1:M), a 
# distribution over the likelihood of sampling each word (object)
# and a desired number and size of trials, and returns a trial order
# (e.g., to accomplish simulations like those in Blythe et al., (2016))


# ToDo: how do we evaluate a model's fit to the data?