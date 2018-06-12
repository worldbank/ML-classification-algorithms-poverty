from sklearn.base import BaseEstimator
class Sampler(BaseEstimator):
    def __init__(self, technique=None):        
        self.technique = technique

    def fit(self, X, y):
        self.technique.fit(X, y)
        return self
        
    def sample(self, X, y):
        return self.technique.sample(X, y)
    
    def fit_sample(self, X, y):
        return self.technique.fit(X, y).sample(X, y)