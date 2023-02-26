import os
import math
import numpy as np


def concave_function(x):
    """
    \phi(x)
    """
    # options: modA**0.5, np.log(1+modA), (1-np.exp(-modA)), modA/(1+modA)
    return x**0.5


class submodular_oracle:
    
    def __init__(self, concave_function, weights, X):
        """
        :param concave_function: function \phi
        :param weights: for linear combination of \phi(\sum_{i \in A} X_{ij})) (length m)
        :param X: feature matrix (n x m)
        """
        self.concave_function = concave_function
        self.weights = weights
        self.X = X
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.set = []
        self.modular_values = np.zeros(self.m)
        self.current_objective = 0
    
    def add_element(self, idx):
        self.modular_values += self.X[idx,:]
        self.current_objective = np.dot(
            self.weights, self.concave_function(self.modular_values)
        )
        self.set.append(idx)
    
    def compute_gain(self, idx):
        new_modular_values = self.modular_values + self.X[idx,:]
        return np.dot(
            self.weights, self.concave_function(new_modular_values)
        ) - self.current_objective

    def compute_set_value(self, indices):
        modular_values = np.sum(self.X[indices, :], axis=0)
        return np.dot(
            self.weights, self.concave_function(modular_values)
        )
