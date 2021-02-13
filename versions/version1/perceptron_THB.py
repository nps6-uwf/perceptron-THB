# Author: Nick Sebasco 
# Date: 1/31/2021
# python version 3.8
import numpy as np
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
    print("weights:", weights)
    
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

if __name__ == "__main__":
    # 0) Generate thai coin dataset.
    error = 0.35
    data, labels = gen_dataset(N = 1000, error = error)

    # 1) Partition dataset into test/ train sets.
    partition = int(len(data) * 0.7)
    train = data[:partition]
    test = data[partition:]

    # 2) Training: Use the pla to find the separating hyperplane.
    weights, misclassifications, it = perceptron_learning_algorithm(train, permute=False, max_iterations=1e4)

    # 3) Testing: Use new weights to classify the test points.
    correct = 0
    for x, y in test:
        if perceptron_classifier_compact(weights, x) == y:
            correct += 1

    # 4) Logging.
    print(f"Using perceptron to classify {labels}")
    print("Weights: ", weights)
    print("Misclassifications: ", misclassifications)
    print(f"Found a separating hyperplace in {it} iterations.")
    print(f"Accuracy:  {100 * correct/len(test):.2f}%")
    plot_data(train, weights)
    plot_data(data, weights)
