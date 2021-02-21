# Author: Nick Sebasco 
# Date: 1/31/2021
# Version: python version 3.8
# Updates: added validation scheme: kfold cross validation 
import numpy as np
from sys import argv
from copy import deepcopy
from random import shuffle
from THB_data import gen_dataset, plot_data

def sign(x: float) -> int:
    """sign: R -> {-1,+1} | specifically 1 if x > 0 and -1 otherwise."""
    return 1 if x > 0 else -1

def perceptron_classifier_compact(weights: np.array, test: np.array):
    """
    Using a more compact notation and linear algebra.  Take the vector dot product 
    of the transpose of the weights and the test vector.
    .item() will return the first element from a vector/ matrix and sice we have
    a 1x1 or a scalar, we are simply pulling out our scalar.
    x0 = 1 -> I need to manually stitch a 1 to the front of these vectors.
    w0 = bias
    (1xd)T * 1xd -> dx1 * 1xd thus the product dimension = 1*1 or a scalar
    """
    return sign(weights.T.dot(test).item())

# helper function cycle
def cycle(lst: list, i: int = 0) -> iter:
    """Return: An iterator that will infinitely cycle through the values from an
    input list.
    """
    j = i
    while True:
        yield (j, lst[i])
        i = i + 1 if i + 1 < len(lst) else 0
        j += 1

def perceptron_learning_algorithm(
    train: list,
    weights: np.array = None,
    permute: bool = True,
    max_iterations: float = float("inf")
    ) -> (np.array, int, int):
    """
    parameters:
        train: [
            (xi: np.array(float, ...), yi: {+1, -1})
        ]
        weights: np.array(float, ...) | default None -> np.array[0,...]
        permute:
        max_iterations: # of maximum allowed iterations.  default set at infinity.
    Assumptions:
        + Data set is linearly separable.  If it is not we risk infinite iteration.
        To mitigate this we can set max_iterations or throw an error. This can 
        be restated as there exists a vector w such that all transpose(w)x_n = y_n
        on all training examples.
    Return:
        (np.array, int): 
            np.array: represents our weights which constitutes the hyperplane that separates
            our data. 
            int: represents the number of points that remain misclassified.
            int: # of iterations to reach solution.
    """
    # 0) Optionally randomly shuffle the training set
    if permute:
        train = deepcopy(train)
        shuffle(train)

    # 1) initialize weights (if needed)
    weights = weights or np.zeros(len(train[0][0]))
    
    # 2) this variable will track how many correct classifications we have.
    correctly_classified = 0

    # 3) Iterate through training set as long as we have misclassified data. 
    # i: int = current iteration, t: tuple = (xi, yi)
    for i, t in cycle(train):
        # 3a) unpack training example & test for misclassification
        x, y = t        
        if y * perceptron_classifier_compact(weights, x) > 0:
            correctly_classified += 1
        else:
            # update the weights
            weights = weights + y * x
            correctly_classified = 0
        # 3b) 3 possible ways this program terminates:
        # 1. correct classification for all traing examples
        # 2. exceeded max iterations
        # 3. infinite loop (never happens if max_iterations is set)
        if correctly_classified == len(train) or i >= max_iterations:
            # Case 1) correct classification for all training examples. 
            # Case 2) we failed to find a solution within max_iterations.
            return (weights, len(train) - correctly_classified, i)
        # Case 3) infinite loop because we failed to meet the criterion 
        # of linear separability
        # print("iteration: ", i)

# Methods for evaluating classification algorithms.
def evaluate_model(classifier, weights, test):
    """Evaluate accuracy of the model by calculating the out of sample
    error.  The fraction of correct classifications in the test set.
    """
    correct = 0
    for x, y in test:
        if classifier(weights, x) == y:
            correct += 1
    return correct/ len(test)

def ttsplit(learning_algorithm, classifier, dataset: list, k: float = 0.7):
    """A procedure used to evaluate a machine learning model by splitting the
    dataset into test/ train sets
    """
    partition = int(len(data) * k)
    train = dataset[:partition]
    test = dataset[partition:]
    weights, Ein, it = learning_algorithm(train)
    Eout = evaluate_model(classifier, weights, test)

    return (weights, Ein, Eout, it)

def kfold(learning_algorithm, classifier, dataset: list, k: int = 10):
    """Implementation of k fold cross validation.  
    A resampling procedure used to evaluate machine learning on limited data sample.  Estimate 
    skill of model on unseen data.  Generally results in less biased/ optimistic model estimate
    than traditional test/ train split.

    Algorithm overview:
    1. Randomly shuffle the dataset
    2. Split the dataset into k groups
    3. For each group:
        a.  Take this group to be a the test dataset
        b.  Take the remaining groups as the training dataset
        c.  Fit model on training set and evaluate on test set.
        d.  Retain evaluation score and discard the model.
    4.  The result is generally the mean of the model scores.  Also good practice to report the
        variance.

    Choice of k:
    1.  Representative - Choose k such that each sample will be representative of the larger data set.
    2.  k = 10 (default)
    3.  The choice of k is usually 5 - 10, but this is not a formal rule.
    4.  Leave one out cross validation
    """
    # train = [other[j:j+k] for j in range(0, len(other), k)]
    data = []
    for i in range(0, len(dataset), k):
        test = dataset[i:i+k]
        train = dataset[:i] + dataset[i+k:]
        weights, Ein, it = learning_algorithm(train) # Ein = in sample missclassifications.
        Eout = evaluate_model(classifier, weights, test)
        data.append(((Ein, Eout), weights, it))
    return data
        
if __name__ == "__main__":
    # 0) Generate thai coin dataset.
    error = 0.25
    N = 500
    data, labels = gen_dataset(N = N, error = error)
    validation = "ttsplit" if len(argv) == 1 else argv[1] # "ttsplit", "kfold"
    k = 10 if len(argv) < 2 else int(argv[2])
    learning_algorithm = lambda train: perceptron_learning_algorithm(
        train,
        weights = None,
        permute = False,
        max_iterations = 1e4
    )

    # 1) Choose validation scheme
    #   a) Partition dataset into test/ train sets.
    #   b) Kfold cross validation
    #   c) Leave one out cross validation
    #   d) Stratified
    #   e) Repeated
    #   f) Nested

    if validation == "ttsplit":
        weights, Ein, Eout, it = ttsplit(learning_algorithm, perceptron_classifier_compact, data, k = 0.7)
    elif validation == "kfold":
        dataset = kfold(learning_algorithm, perceptron_classifier_compact, data, k = 10)
        min_error, weights, it = float("inf"), None, "NA"
        Ein_total, Eout_total = 0, 0
        for d in dataset:
            print(d)
            error = d[0]
            w = d[1]
            Ein, Eout = error
            Ein_total += Ein
            Eout_total += Eout
            sample_error = (Ein**2 + Eout**2)**0.5 # choose weights with lowest Ein + Eout
            if sample_error < min_error:
                min_error = sample_error
                weights = w

        Ein = Ein_total/ len(dataset)
        Eout = Eout_total/ len(dataset)

    # 2) Logging.
    print(f"Using perceptron to classify {labels}")
    print(f"Validation: {validation}")
    print("Weights: ", weights, type(weights))
    print("Misclassifications: ", Ein)
    print(f"Found a separating hyperplace in {it} iterations.")
    print(f"Accuracy:  {100 * Eout:.2f}%")
    plot_data(data, weights)

