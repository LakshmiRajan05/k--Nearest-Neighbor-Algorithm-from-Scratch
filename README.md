# k--Nearest-Neighbor-Algorithm-from-Scratch

This is an implementation of k- nearest neighbor algorithm using L2 norm (Euclidean distance). The major steps onvolved in the implementation are

1. Calculate distances: Compute the distance (usually Euclidean distance) between the test point and all points in the training dataset.
2. Identify K nearest neighbors: Select the K points with the smallest distances — which are the "K nearest neighbors".
3. If classifier - Vote for the majority class: Count the occurrences of each class label among the neighbors and select the class with the most votes.
4. For Regression - Calculate the mean of target values of k neighbors.

The model is trained and tested on CIFAR-10 image data set