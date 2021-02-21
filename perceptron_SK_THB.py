# Author: Nick Sebasco
# Date: 2/ 6 / 2021

"""
Notes from sklearn on the Perceptron:
1. Does not require a learning rate
2. Not regularized (penalized)
3. Updates model on mistakes only
"""

from THB_data import gen_dataset, plot_data, plot_hyperplane
from sklearn.linear_model import Perceptron

def main():
    error = 0.25
    N = 500
    data, labels = gen_dataset(N = 500, error = error)
    TRAIN_X, TRAIN_Y = [], []
    TEST_X, TEST_Y = [], []
    i = 0
    threshold = 0.7
    for x, y in data:
        if i/len(data) <= threshold:
            TRAIN_X.append(x); TRAIN_Y.append(y)
        else:
            TEST_X.append(x); TEST_Y.append(y)
        i+= 1
    
    print(labels)
    classifier = Perceptron(tol=1e-6)
    classifier.fit(TRAIN_X, TRAIN_Y)
    weights = classifier.coef_[0]
    print("Training score:",classifier.score(TRAIN_X, TRAIN_Y))
    print("Testing score:",classifier.score(TEST_X, TEST_Y))
    print("Weights: ", weights)

    plot_data(data, weights)
    plot_hyperplane(data, classifier)

    # print("Prediction: ", classifier.predict(TEST_X[0].reshape(-1,1)))

if __name__ == "__main__": main()