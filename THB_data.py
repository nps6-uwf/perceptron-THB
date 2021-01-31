from random import sample, choice, random
import numpy as np
import matplotlib.pyplot as plt

specs = {
    "1 satang": {
        "diameter": 15, # mm
        "mass": 0.5 # g
    },
    "5 satang": {
        "diameter": 16, # mm
        "mass": 0.6 # g
    },
    "10 satang": {
        "diameter": 17.5, # mm
        "mass": 0.8 # g
    },
    "25 satang": {
        "diameter": 16, # mm
        "mass": 1.9 # g
    },
    "50 satang": {
        "diameter": 18, # mm
        "mass": 2.4 # g
    },
    "1 baht": {
        "diameter": 20, # mm
        "mass": 3 # g
    },
    "2 baht": {
        "diameter": 21.75, # mm
        "mass": 4 # g
    },
    "5 baht": {
        "diameter": 24, # mm
        "mass": 6 # g
    },
    "10 baht": {
        "diameter": 26, # mm
        "mass": 8.5 # g
    }
}

def gen_dataset(N = 100, error = 0.1):
    data = []
    labels = sample(specs.keys(), k=2)
    for _ in range(N):
        key = choice(labels)
        yi = 1 if labels.index(key) == 0 else -1
        x1 = specs[key]["diameter"]
        x2 = specs[key]["mass"]
        # the first value is a 1 beacuse of the bias weight.
        xi = np.array([1, x1 + choice([1,-1])*random()*(error*x1), x2 + choice([1,-1])*random()*(x2*error)])
        data.append((xi, yi))
    return (data, labels)

def plot_data(dataset, w = None):
    x1 = [d[0][1] for d in dataset]
    x2 = [d[0][2] for d in dataset]
    if  type(w) == type(np.array([])):
        # [bias x1, x2]
        # [w0, w1, w2]  w2y + w1x + w0 = 0 -> transform into a line.
        f = lambda x: x * (-(w[0]/w[2])/(w[0]/w[1])) + (-w[0]/w[2])
        x = np.linspace(min(x1), max(x1),100)
        plt.plot(x, [f(i) for i in x], "r")
    plt.scatter(x1, x2)
    plt.xlabel("diameter (mm)")
    plt.ylabel("weight (g)")
    plt.show()
