# perceptron-THB

## Overview
Using the perceptron learning algorithm to differentiate between Thai currency, specifically Thai coins.

This repository contains two files:
<ol>
  <li><strong>perceptron_THB.py</strong>- Implentation of the perceptron classifcation algorithm and perceptron learning algorithm from scratch.  This file does the testing and training.</li> 
  <li><strong>THB_data.py</strong> - Creates an artificial data set using the specifications of Thai coins.</li> 
</ol>

## Data

<code>
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
</code>

Artificial data is generated by the THB_data.py file.  This file contains two importable methods:
<ol>
    <li><strong>gen_dataset</strong>- Genrates an artificial dataset by randomly selecting two coins from the <i>specs</i> dictionary shown above, introduces random error to the diameter and mass of each coin, generates an <i>N</i> length sample.</li> 
  <li><strong>plot_data</strong> - Use matplotlib library to create simple data visualization.</li> 
</ol>

## Learning Algorithm

## Results

The more error I introduce into the dataset, the more interesting the results become.  The separating hyperplane is rather obvious without error.

<img src="https://github.com/nps6-uwf/perceptron-THB/blob/main/assets/Figure_1.png?raw=true"></img>

Sample output after running "perceptron-THB.py":
Using perceptron to classify ['1 satang', '5 baht']
Weights:  [  3.77165317 -41.54356934]
Misclassifications:  0
Found a separating hyperplace in 968 iterations.
Accuracy:  100.00%

## Conclusions/ Improvements
<ul>
  <li>The algorithm is only capable of binary classifcation which is kind of boring in this case because there are much more than 2 THai coins.</li>
  <li>No consideration was made on finding the optimal hyperplane.</li>
  <li>It might be interesting to add a parameter to the perceptron learning algorithm "allowed_misclassifcations", this will allow more leniency.</li>
</ul>

