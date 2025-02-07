
import numpy as np
from scipy import stats

class LinearRegression:
    def __init__(self, X, y):
        self.X = np.column_stack([np.ones(y.shape[0]), X])
        self.y = np.array(y)
        self.some_formulas()
    
    @property
    def d(self):
        return self.X.shape[1] - 1 
    
    @property
    def n(self):
        return self.y.shape[0]
    
    
    def some_formulas(self):
        self.b = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y
        self.SSE = np.sum(np.square(self.y - self.X @ self.b))
        self.Syy = (self.n * np.sum(np.square(self.y)) - np.square(np.sum(self.y))) / self.n
        self.SSR = self.Syy - self.SSE
        self.variance_v = self.SSE / (self.n - self.d - 1)

    def variance(self):
        return self.variance_v
    
    def standard_deviation(self):
        return np.sqrt(self.variance_v)
    
    def r_squared(self):
        return self.SSR / self.Syy
    
    def significance(self):
        f_statistic = (self.SSR / self.d) / self.variance_v
        p_value = stats.f.sf(f_statistic, self.d, self.n - self.d - 1)
        return f_statistic, p_value
    






