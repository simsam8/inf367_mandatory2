---
title: INF367 Mandatory 2
author: 
    - "Christoffer Slettebø"
    - "Simon Vedaa"
    - "Khalil Ibrahim"
---

# Data exploration and pre-processing:

* We found out that we have the following dimension: 

* We have splitted the data into 60% training data, 20% validation data and 20% test data

* We chose to use angle encoding, and therefore we 
    scaled the data so every data point is larger or equal to zero and smaller or equal to pi.

![Class distribution of training data](images/class_distribution.png)

# Circuits and models

Choice of QNN

* We tried 3 curcuits:
    - Real amplitudes
    - Convolution Neural Network
    - A variation of Real Amplitudes

![Circut 1](images/circuit1.png)

![Circut 2](images/circuit2.png)

![Circut 3](images/circuit3.png)

# Training


## Loss function

We are using cross-entropy loss(log_loss) as we are solving a classification problem.


## Gradient Descent

Our gradient descent function is just a standard stochastic gradient descent using
the finite difference method.

## Early stopping


## Model selection

For model selection we implemented random sampling of our hyperparameters.
Each is trained 4 times on randomly selected values and ran for maximum 20 epochs.
We select the model with the highest validation accuracy.

| Learning Rate | Epsilon | Layers(Model 2 and 3) |
| --------------- | --------------- | --------------- |
| 0.1 to 1 | 0.01 to 1 | 2 to 6 |






# Results

Chosen parameters:

| Model | Learning Rate | $\epsilon$ |
| --------------- | --------------- | --------------- |
| Model1 | 1.397 | 0.473 |

Performance(accuracy):

| Training | Validation | Test |
| --------------- | --------------- | --------------- |
| 33.33% | 90.00% | 96.67% |




![Loss](images/loss.png)

![Confusion matrix](images/confusion_matrix.png)


What we learned:



Hardships:

* Long computation time 
* Gradient decent is slow

