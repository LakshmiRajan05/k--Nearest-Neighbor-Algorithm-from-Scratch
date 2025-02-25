import scipy
import numpy as np
from collections import Counter

class KNN():

    def __init__(self, X, y, K=5, distance_metric="L2_norm"):
        
        self.X_train = X
        self.y_train = y
        self.K = K

class KNN_Classifier(KNN):
    
    def __init__(self,X, y, K=5, distance_metric="L2_norm"):

        super().__init__(X, y, K, distance_metric)

    def predict(self,X):

        y_pred = []
        for i in range(len(X)):
            distance = np.sqrt(np.sum(np.square(self.X_train-X[i,:]),axis=1))
            k_neighbours_ind = distance.argsort()[:self.K]
            k_labels = self.y_train[k_neighbours_ind] 
             # predict label based on majority vote
            y_pred.append(Counter(k_labels).most_common(1)[0][0])
        
        return np.array(y_pred)

class KNN_Regression(KNN):

    def __init__(self,X, y, K=5, distance_metric="L2_norm"):

        super().__init__(X, y, K, distance_metric)

    def predict(self,X):

        y_pred = []
        for i in range(len(X)):
            distance = np.sqrt(np.sum(np.square(self.X_train-X[i,:]),axis=1))
            k_neighbours_ind = distance.argsort()[:self.K]
            k_targets = self.y_train[k_neighbours_ind] 
            # compute y by averaging targtes of k-nearest neighbors
            y_pred.append(mean(k_targets))
        
        return np.array(y_pred)