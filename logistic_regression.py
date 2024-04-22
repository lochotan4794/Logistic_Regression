import numpy as np
from regression import Regression
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
class LogisticRegression(Regression):
    model_name = "Logistic Regression"
    """
    Inittialize parameter
        _fit_intercept: allow to train intercept
    """
    def __init__(self, use_intercept=True):
        self._fit_intercept = use_intercept
        np.seterr(divide = 'ignore') 
    """
    Training method 
    Parameter
    --------
        X: sample features
        y: label
        ilters: Number of ilterations
        learning_rate: learning rate in case using Gradient Descent
        method: algorithm to training model IRLS or Gradient Descent
    return
    --------
        beta: parameters
    """
    def fit(self, X, y, ilters=10, learning_rate=0.0001, method="GD"):
        self._random_init_parameter(X.shape[1])
        self.logs = []
        self.acc = []
        self.X = X
        self.y = y
        # Replace None dimension in training label
        if y.ndim == 1:
            self.y = np.expand_dims(y, axis=1)
        # Update paprameter by Gradient Descent method
        self.num_of_samples = X.shape[0]
        for _ in range(ilters):
            dB, db = self._get_gradient()
            self.beta = self.beta - learning_rate * dB * (1/self.num_of_samples)
            if self._fit_intercept:
                self.bias = self.bias - learning_rate * db * (1/self.num_of_samples)
            self.logs.append(self.cost_function())
            self.acc.append(accuracy_score(self.predict(self.X), y))
        return self.beta
    """
    Method caculate gradient
    """
    def _get_gradient(self):
        if  self._fit_intercept:
            prop = sigmoid(np.dot(self.X, self.beta.T) + self.bias)
        else:
            prop = sigmoid(np.dot(self.X, self.beta.T))
        error = prop - self.y
        dB = np.dot(error.T, self.X)
        db = np.sum(error)
        return dB, db
    """
    Method using to predict probabilty.
    In case of two class, method return probabilty of samples goes into True class
    """
    def predict_proba(self, X): 
        if self._fit_intercept:
            prob = sigmoid(np.dot(X, self.beta.T) + self.bias)
        else:
            prob = sigmoid(np.dot(X, self.beta.T))
        return prob
    """
    Method using to predict class
    In case multiclass, the One-Vs-Rest was used (not yet implemented)
    """
    def predict(self, X): 
        if self._fit_intercept:
            prob = sigmoid(np.dot(X, self.beta.T) + self.bias)
        else:
            prob = sigmoid(np.dot(X, self.beta.T))
        return np.round(prob).astype(np.int64)
    """
    Method caculate loss function after each iteration
    return:
        Log-loss value 
    """
    def cost_function(self):
        if self._fit_intercept:
            prob = sigmoid(np.dot(self.X, self.beta.T) + self.bias)
        else:
            prob = sigmoid(np.dot(self.X, self.beta.T))
        inv_prop =  1 - prob
        log_prop = np.log(prob)
        log_inv_prop = np.log(inv_prop)
        total_cost = -(1 / self.num_of_samples) * np.sum(self.y*log_prop + (1 - self.y)*log_inv_prop)
        return total_cost
    """
    Method inittilize paramters
    Input:
        feature_shape: number of feature used
    return:
        random normal paramters array with or without intercept
    """
    def _random_init_parameter(self, feature_shape):
        np.random.seed(1)
        if self._fit_intercept:
            self.beta = np.random.randn(1, feature_shape)
            self.bias = np.random.rand(1, 1) 
        else:
            self.beta = np.random.randn(1, feature_shape) 
        return