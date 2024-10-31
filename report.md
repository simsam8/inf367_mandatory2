- Authors: Christoffer Slettebø, Simon Vedaa, Khalil Ibrahim

Data processing:

* We found out that we have the following dimension: 

* We have splitted the data into 70% training data, 15% validation data and 15% test data

* Scaled the data so every data point is larger or equal to zero and smaller or equal to pi


Loss function

* We use log_loss as out loss function since we have a classifier




Choice of QNN

* We tried 3 curcuits:
    - Real amplitudes
    - Convolution Neural Network


Underway results:
- Convolution Neural Network
    - good inital result, 100% accuracy on type 0, 1 but only 40% on 2
    - tried to change some gets from rx to rz or ry, reduced the accuracy heavly, but seemed that the learning ended to early


What we learned:



Hardships:

* Long computation time 
* Gradient decent is slow

