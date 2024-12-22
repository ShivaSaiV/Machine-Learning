import numpy as np

class Perceptron:
    '''
    Parameters: 
    eta: learning rate
    n_iter: # of passes over train dataset
    random_state: random number generator seed for random weight
    
    Attributes:
    w: weights after fitting
    b: bias unit after fitting
    
    errors: number of misclassifications in each epoch
    '''

    def __init__(self, eta = 0.1, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        mu, sigma = 0, 0.01
        self.w = rgen.normal(loc=mu, scale=sigma, size=X.shape[1])

        self.b = np.float_(0.)
        self.errors = []

        for i in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w += update * xi
                self.b += update
                if (update != 0.0):
                    error += 1
            self.errors.append(error)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)








'''
z = w_i*x_i + ... + w_m*x_m + b
z = w^T*x + b
sigmoid(z) = 1 if z >= 0

Updates:
w_j = n(y_i - ^y_i) * x_j
'''