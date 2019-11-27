from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import cm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class LinearRegressionOneVar:

    def __init__(self, max_steps=1000, linear_rate=0.02):
        self.theta_0 = None
        self.theta_1 = None
        self.linear_rate = linear_rate
        self.max_steps = max_steps
        self.__logs = None

    def fit(self, X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
    
        self.theta_0 = -1
        self.theta_1 = 1

        cur_step = 0
        cur_loss = self.loss_func(X, y)
        self.__logs = [[cur_step, self.theta_0, self.theta_1, cur_loss]]

        for cur_step in range(self.max_steps):
            self.gradient_descent(X, y)
            new_loss = self.loss_func(X, y)
            
            if cur_step % 10 == 0:
                self.__logs.append([cur_step, self.theta_0, self.theta_1, new_loss])

            cur_loss = new_loss

        if cur_step > self.max_steps:
            raise Exception("Model reached maximum steps number.")

    def gradient_descent(self, X, y):
        hypotesis = self._calculate_hypotises(X, y)
        y_size = y.size
        theta_0 = self.theta_0 - self.linear_rate * np.sum(hypotesis - y) / y_size
        theta_1 = self.theta_1 - self.linear_rate * np.sum(np.multiply(hypotesis - y, X)) / y_size
        self.theta_0, self.theta_1 = theta_0, theta_1
    
    def loss_func(self, X, y, theta_0=None, theta_1=None):
        hypotesis = self._calculate_hypotises(X, y, theta_0, theta_1)
        coef = 1 / (2 * y.size)
        return coef * np.sum((hypotesis - y)**2)

    def predict(self, x):
        if self.theta_0 is None or self.theta_1 is None:
            raise Exception("Model is not trained. Call `fit` method.")

        return self.theta_0 + self.theta_1 * x

    @property
    def logs(self):
        return pd.DataFrame(self.__logs, columns=['iter', 'theta_0', 'theta_1', 'loss'])

    def _calculate_hypotises(self, X, y, theta_0=None, theta_1=None):
        t0 = theta_0 if theta_0 is not None else self.theta_0
        t1 = theta_1 if theta_1 is not None else self.theta_1
        A = np.column_stack((np.ones(X.size), X))
        b = np.array([t0, t1])
        return A.dot(b)
    

class MultivariateLinearRegression:
    THRESHOLD = 1

    def __init__(self, max_steps=500000, linear_rate=0.01, normalize=True, vectorize=True, method=None):
        self.weights = None
        self.costs = []
        self.normalize = normalize
        self.vectorize = vectorize
        self.max_steps = max_steps
        self.linear_rate = linear_rate
        self.method = method

    def predict(self, X):
        if self.weights is None:
            raise Exception("Model is not trained. Call `fit` method.")

        A = np.insert(X, 0, 1)        
        return self._calculate_hypotesis(A)

    def fit(self, X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        X = X.astype('float64') 
        y = y.astype('float64')

        if self.normalize:
            X = self.normalize_features(X)
        
        X = np.column_stack((np.ones(X.shape[0]), X))
        
        if self.method == 'normal_equation':
            self.weights = self.normal_equation(X, y)
            return
        
        self.weights = np.zeros(X.shape[1])
        cur_loss = self.cost_func(X, y)

        cur_step = 0
        while cur_step < self.max_steps:
            cur_step += 1
            self.gradient_descent(X,y)
            new_loss = self.cost_func(X, y)
            self.costs.append(new_loss)
            if abs(new_loss - cur_loss) < self.THRESHOLD:
                break

            cur_loss = new_loss

        self.steps = cur_step
            
    def normalize_features(self, X):
        N = X.shape[1]
        copy_X = X.copy()
        for i in range(N):
            feature = X[:, i]
            mean = np.mean(feature)
            delta = np.max(feature) - np.min(feature)            
            copy_X[:, i] -= mean
            copy_X[:, i] /= delta
        return copy_X
        
    def cost_func(self, X, y, weights=None):
        if weights is None:
            weights = self.weights

        predictions = self._calculate_hypotesis(X, weights)
        squared_error = (predictions - y) ** 2
        return np.mean(squared_error) / 2

    def normal_equation(self, X, y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def gradient_descent(self, X, y):
        predictions = self._calculate_hypotesis(X)
        diff = predictions - y
        if self.vectorize:
            self._gradient_descent_vectorized(X, diff)
        else:
            self._gradient_secent_simple(X, diff)
    
    def _calculate_hypotesis(self, X, weights=None):
        if weights is None:
            weights = self.weights
        
        return X.dot(weights)
    
    def _gradient_descent_vectorized(self, X, diff):
        gradient = np.dot(X.T, diff)
        gradient /= X.shape[0]
        gradient *= self.linear_rate
        self.weights -= gradient
    
    def _gradient_secent_simple(self, X, diff):
        feature_size = X.shape[1]
        for i in range(feature_size):
            self.weights[i] -= self.linear_rate * np.mean(X[:, i] * diff)


def main():
    FILENAME = 'ex1data1.txt'
    df = pd.read_csv(FILENAME, header=None, names=['population', 'profit'])
    X_train, y_train = df['population'], df['profit']


    plt.plot(X_train, y_train, 'o', markersize=4)
    plt.xlabel('population')
    plt.ylabel('profit')
    plt.show()


    lin = LinearRegressionOneVar(max_steps=1000, linear_rate=0.02)
    lin.fit(X_train, y_train)
    min_x, max_x = int(min(X_train)), int(max(X_train))
    xi = list(range(min_x, max_x + 1))
    line = [lin.predict(i) for i in xi]
    print(f"Function: y = {round(lin.theta_0, 3)} + {round(lin.theta_1, 3)} * x")
    plt.plot(X_train, y_train, 'o', xi, line, markersize=4)
    plt.xlabel('population')
    plt.ylabel('profit')
    plt.show()


    data = lin.logs
    data = data[data['theta_1'] > 1]
    X, Y = np.meshgrid(data['theta_0'], data['theta_1'])
    Z = np.zeros((data['theta_0'].size, data['theta_1'].size))

    for i, x in enumerate(data['theta_0']):
        for j, y in enumerate(data['theta_1']):
            Z[i, j] = lin.loss_func(X_train, y_train, x, y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
    ax.set_title('Зависимость функции потерь от параметров (3D поверхность)')
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Loss function')
    plt.show()

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, cmap='viridis')
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Countour plot for loss function')
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    plt.show()


    FILENAME2 = 'ex1data2.txt'
    df = pd.read_csv(FILENAME2, header=None, names=['area', 'rooms', 'price'])
    X_train, y_train = df.filter(['area', 'rooms']), df['price']


    mult = MultivariateLinearRegression(max_steps=100, linear_rate=0.005, normalize=False)
    mult.fit(X_train, y_train)
    xi_nn, costs_non_normalized = list(range(mult.steps)), mult.costs

    mult2 = MultivariateLinearRegression(max_steps=100, linear_rate=0.005, normalize=True)
    mult2.fit(X_train, y_train)
    xi_norm, costs_normalized = list(range(mult2.steps)), mult2.costs

    fig, ax1 = plt.subplots()
    ax1.plot(xi_nn, costs_non_normalized)
    ax1.set_xlabel('Number of steps')
    ax1.set_ylabel('Cost function')
    ax1.set_title('Not normalized')

    fig2, ax2 = plt.subplots()         
    ax2.plot(xi_norm, costs_normalized)
    ax2.set_xlabel('Number of steps')
    ax2.set_ylabel('Cost function')
    ax2.set_title('Normalized')
    plt.show()


    TIME_FORMAT = "%H:%M:%S"

    mult = MultivariateLinearRegression(linear_rate=0.001, vectorize=False)
    mult2 = MultivariateLinearRegression(linear_rate=0.001, vectorize=True)

    start_time = datetime.now()
    print(f"Start time without vectorization: {start_time.strftime(TIME_FORMAT)}")
    mult.fit(X_train, y_train)
    end_time = datetime.now()
    print(f"End time without vectorization: {end_time.strftime(TIME_FORMAT)}\n")
    spent_non_vec = end_time - start_time

    start_time = datetime.now()
    print(f"Start time with vectorization: {start_time.strftime(TIME_FORMAT)}")
    mult2.fit(X_train, y_train)
    end_time = datetime.now()
    print(f"End time with vectorization: {end_time.strftime(TIME_FORMAT)}\n")
    spent_vec = end_time - start_time

    print(f"Spent time without vectorization: {spent_non_vec}")
    print(f"Spent time with vectorization: {spent_vec}")


    mult = MultivariateLinearRegression(linear_rate=0.1)
    mult2 = MultivariateLinearRegression(linear_rate=0.01)
    mult.fit(X_train, y_train)
    mult2.fit(X_train, y_train)

    xi1, costs1 = list(range(mult.steps)), mult.costs
    xi2, costs2 = list(range(mult2.steps)), mult2.costs

    fig, ax = plt.subplots()
    ax.plot(list(range(mult.steps)), mult.costs)
    ax.set_xlabel('Number of steps')
    ax.set_ylabel('Cost function')
    ax.set_title('Linear rate = 0.01')

    fig, ax2 = plt.subplots()
    ax2.plot(list(range(mult2.steps)), mult2.costs)
    ax2.set_xlabel('Number of steps')
    ax2.set_ylabel('Cost function')
    ax2.set_title('Linear rate = 0.1')
    plt.show()


    mult = MultivariateLinearRegression(method='normal_equation', normalize=False)
    mult.fit(X_train, y_train)
    A = np.column_stack((np.ones(X_train.shape[0]), X_train))
    cost_normal = mult.cost_func(A, y_train)

    m = MultivariateLinearRegression(linear_rate=0.001)
    m.fit(X_train, y_train)
    cost_gradient = m.costs[-1]

    print(f"Cost function with analytical approach: {cost_normal}")
    print(f"Cost function with gradient descent: {cost_gradient}")
    

if __name__ == '__main__':
    main()
