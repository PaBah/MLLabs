import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_file(filename, names):
    return pd.read_csv(filename, header=None, names=names)


df = load_file('ex2data1.txt', ['first_exam', 'second_exam', 'accepted'])
X_train, y_train = df.filter(['first_exam', 'second_exam']), df['accepted']


z_true = df[df['accepted'] == 1]
z_false = df[df['accepted'] == 0]
fig, ax = plt.subplots()
ax.scatter(z_true['first_exam'], z_true['second_exam'], marker='o', c='g', label='Accepted', s=20)
ax.scatter(z_false['first_exam'], z_false['second_exam'], marker='x', c='r', label='Not accepted', s=20)
ax.legend(loc='upper right');
ax.set_xlabel('First exam')
ax.set_ylabel('Second exam')
plt.show()


from utils import sigmoid

class LogisticRegression:
    THRESHOLD = 1e-6

    def __init__(self, fit_method='gradient_descent', max_steps=100000,
                 learning_rate=0.01, regularized=False, reg_L=0.5, log=False):
        self.weights = []
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.regularized = regularized
        self.reg_L = reg_L
        self.cost_func = self.cost_func_regularized if regularized else self.cost_func_non_regularized
        self.cost_der = self.cost_der_regularized if regularized else self.cost_der_non_regularized
        self.fit_method = getattr(self, fit_method)
        self.log = log
        
    def fit(self, X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        X = X.astype('float64') 
        y = y.astype('float64')
        
        if not self.regularized:
            X = np.column_stack((np.ones(X.shape[0]), X))
        
        self.fit_method(X, y)
    
    def predict(self, X):
        if self.weights is None:
            raise Exception("Model is not trained. Call `fit` method.")

        X = np.array(X)
        if not self.regularized:
            X = np.insert(X, 0, 1)
        h = self.calculate_hypotesis(X)
        return 1 if h >= 0.5 else 0
    
    def gradient_descent(self, X, y):
        self.cost_history = []
        self.weights = np.zeros(X.shape[1])
        cur_loss = self.cost_func(X, y)

        cur_step = 0
        while cur_step < self.max_steps:
            cur_step += 1
            self.gradient_descent_step(X, y)
            new_loss = self.cost_func(X, y)
            self.cost_history.append(new_loss)
            if abs(new_loss - cur_loss) < self.THRESHOLD:
                break

            cur_loss = new_loss
    
    def gradient_descent_step(self, X, y):
        gradient = self.cost_der(X, y, self.weights)
        gradient *= self.learning_rate
        self.weights -= gradient
    
    def cost_func_non_regularized(self, X, y, weights=None):
        if weights is None:
            weights = self.weights
        
        predictions = self.calculate_hypotesis(X, weights)
        cost_trues = y * np.log(predictions)
        cost_falses = (1 - y) * np.log(1 - predictions)
        total_cost = -np.mean(cost_trues + cost_falses)
        return total_cost
    
    def cost_func_regularized(self, X, y, weights=None):
        if weights is None:
            weights = self.weights
        
        cost = self.cost_func_non_regularized(X, y, weights)
        weights_R = weights[1:]
        total_cost = cost + (self.reg_L / 2 / X.shape[0]) * np.dot(weights_R.T, weights_R)
        return total_cost
    
    def calculate_hypotesis(self, X, weights=None):
        if weights is None:
            weights = self.weights

        return sigmoid(X.dot(weights))
    
    def cost_der_non_regularized(self, X, y, theta):
        predictions = self.calculate_hypotesis(X, weights=theta)
        sq_error = predictions - y
        gradient = np.dot(X.T, sq_error)
        gradient /= X.shape[0]
        return gradient

    def cost_der_regularized(self, X, y, theta):
        predictions = self.calculate_hypotesis(X, weights=theta)
        sq_error = predictions - y
        gradient_first = np.dot(X.T[:1], sq_error)
        gradient_full = np.dot(X.T[1:], sq_error) + self.reg_L * theta[1:]
        gradient = np.insert(gradient_full, 0, gradient_first)
        gradient /= X.shape[0]
        return gradient
    
    def nelder_mead_algo(self, X, y):
        from scipy.optimize import fmin

        N = X.shape[0]

        def func(theta):
            return self.cost_func(X, y, theta)
        
        init_theta = np.zeros(X.shape[1])
        self.weights = fmin(func, init_theta, xtol=self.THRESHOLD, maxfun=100000)
    
    def bfgs_algo(self, X, y):
        from scipy.optimize import fmin_bfgs

        N = X.shape[0]

        def func(theta):
            return self.cost_func(X, y, theta)
        
        def func_der(theta):
            return self.cost_der(X, y, theta)

        init_theta = np.zeros(X.shape[1])
        self.weights = fmin_bfgs(func, init_theta, fprime=func_der, gtol=self.THRESHOLD, disp=self.log)
        

cls_grad = LogisticRegression(fit_method='gradient_descent', max_steps=300000, learning_rate=0.004)
cls_grad.fit(X_train, y_train)
print(f'Minimum cost function value: {cls_grad.cost_history[-1]}')
print(f'Iterations: {len(cls_grad.cost_history)}')
print(f'Weights: {cls_grad.weights}')

cls_nm = LogisticRegression(fit_method='nelder_mead_algo')
cls_nm.fit(X_train, y_train)
print(f'Weights: {cls_nm.weights}')

cls_bfgs = LogisticRegression(fit_method='bfgs_algo', log=True)
cls_bfgs.fit(X_train, y_train)
print(f'Weights: {cls_bfgs.weights}')


z_true = df[df['accepted'] == 1]
z_false = df[df['accepted'] == 0]

def decision_boundary(x, weights):
    return -(weights[0] + weights[1] * x) / weights[2]

fig, ax = plt.subplots()
ax.scatter(z_true['first_exam'], z_true['second_exam'], marker='o', c='g', label='Accepted', s=20)
ax.scatter(z_false['first_exam'], z_false['second_exam'], marker='x', c='r', label='Not accepted', s=20)
ax.plot(z_false['first_exam'],
        [decision_boundary(i, cls_grad.weights) for i in z_false['first_exam']],
        c='b', label='Decision boundary')
ax.legend(loc='upper right');
ax.set_xlabel('First exam')
ax.set_ylabel('Second exam')
plt.show()


df = load_file('ex2data2.txt', names=['first_test', 'second_test', 'passed'])
X_train, y_train = df.filter(['first_test', 'second_test']), df['passed']


z_true = df[df['passed'] == 1]
z_false = df[df['passed'] == 0]
fig, ax_reg = plt.subplots()
ax_reg.scatter(z_true['first_test'], z_true['second_test'], marker='o', c='g', label='Passed', s=20)
ax_reg.scatter(z_false['first_test'], z_false['second_test'], marker='x', c='r', label='Not passed', s=20)
ax_reg.legend(loc='upper right');
ax_reg.set_xlabel('First test')
ax_reg.set_ylabel('Second test')
plt.show()


def build_poly_features(x1, x2, log=False):
    degree = 6
    res = []
    str_res = []

    for i in range(degree + 1):
        for j in range(i, degree + 1):
            res.append(x1**(j - i) * x2**i)
            first = '' if j - i == 0 else 'x1' if j - i == 1 else f'x1^{j - i}'
            second = '' if i == 0 else 'x2' if i == 1 else f'x2^{i}'
            if not first and not second:
                str_append = '1'
            elif first and not second:
                str_append = first
            elif second and not first:
                str_append = second
            else:
                str_append = f"{first}*{second}"
            str_res.append(str_append)

    str_res = ' + '.join(str_res)
    if log:
        print(str_res)
    assert len(res) == 28
    return np.array(res).T


X_poly = build_poly_features(X_train['first_test'], X_train['second_test'], log=True)

cls_grad_reg = LogisticRegression(fit_method='gradient_descent', regularized=True,
                                  max_steps=300000, learning_rate=0.5, reg_L=0.5)
cls_grad_reg.fit(X_poly, y_train)
print(f'Minimum cost function value: {cls_grad_reg.cost_history[-1]}')
print(f'Iterations: {len(cls_grad_reg.cost_history)}')
print(f'Weights: {cls_grad_reg.weights}')

cls_nm_reg = LogisticRegression(fit_method='nelder_mead_algo', regularized=True)
cls_nm_reg.fit(X_poly, y_train)
print(f'Weights: {cls_nm_reg.weights}')

cls_bfgs_reg = LogisticRegression(fit_method='bfgs_algo', regularized=True, log=True)
cls_bfgs_reg.fit(X_poly, y_train)
print(f'Weights: {cls_bfgs_reg.weights}')


print(f"Predicted class: {cls_grad_reg.predict(X_poly[0])}, actual class: {y_train[0]}")
print(f"Predicted class: {cls_nm_reg.predict(X_poly[0])}, actual class: {y_train[0]}")
print(f"Predicted class: {cls_bfgs_reg.predict(X_poly[0])}, actual class: {y_train[0]}")


def decision_boundary_contour(theta1, theta2, theta3):
    u = np.linspace(-1, 1.2, 50)
    v = np.linspace(-1, 1.3, 50)
    z1 = np.zeros(shape=(len(u), len(v)))
    z2 = np.zeros(shape=(len(u), len(v)))
    z3 = np.zeros(shape=(len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z1[i, j] = build_poly_features(np.array(u[i]), np.array(v[j])).dot(theta1)
            z2[i, j] = build_poly_features(np.array(u[i]), np.array(v[j])).dot(theta2)
            z3[i, j] = build_poly_features(np.array(u[i]), np.array(v[j])).dot(theta3)

    z1 = z1.T
    z2 = z2.T
    z3 = z3.T
    fig, ax_reg = plt.subplots()
    ax_reg.contour(u, v, z1, levels=0, colors='b')
    ax_reg.contour(u, v, z2, levels=0, colors='g')
    ax_reg.contour(u, v, z3, levels=0, colors='y')
    z_true = df[df['passed'] == 1]
    z_false = df[df['passed'] == 0]
    ax_reg.scatter(z_true['first_test'], z_true['second_test'], marker='o', c='g', label='Passed', s=20)
    ax_reg.scatter(z_false['first_test'], z_false['second_test'], marker='x', c='r', label='Not passed', s=20)
    ax_reg.legend(loc='upper right');
    ax_reg.set_xlabel('First test')
    ax_reg.set_ylabel('Second test')
    ax_reg.set_title('Decision boundary, lambda = %f' % cls_grad_reg.reg_L)
    plt.show()
    
decision_boundary_contour(cls_grad_reg.weights, cls_nm_reg.weights, cls_bfgs_reg.weights)


cls1 = LogisticRegression(fit_method='gradient_descent', max_steps=300000, learning_rate=0.5,
                          regularized=True, reg_L=0.5)
cls1.fit(X_poly, y_train)

cls2 = LogisticRegression(fit_method='gradient_descent', max_steps=300000, learning_rate=0.5,
                          regularized=True, reg_L=0.05)
cls2.fit(X_poly, y_train)

cls3 = LogisticRegression(fit_method='gradient_descent', max_steps=300000, learning_rate=0.5,
                          regularized=True, reg_L=0.005)
cls3.fit(X_poly, y_train)

decision_boundary_contour(cls1.weights, cls2.weights, cls3.weights)


from scipy.io import loadmat

mat = loadmat('ex2data3.mat')
X_train, y_train = mat['X'], mat['y']
y_train = y_train.reshape(y_train.shape[0])
y_train = np.where(y_train != 10, y_train, 0)


def vector_to_matrix(x):
    len_vec = len(x)
    step = int(np.sqrt(len_vec))
    assert step ** 2 == len_vec, 'Matrix should be squared' 
    matrix = [x[left:left+step] for left in range(0, len_vec, step)]
    np_matrix = np.array(matrix).T
    reversed_matrix = np.flip(np_matrix, axis=0)
    return reversed_matrix

nums = list(range(150, 5000, 500))
pictures = [vector_to_matrix(X_train[i]) for i in nums]

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i, ax in enumerate(axs.flatten()):
    ax.pcolor(pictures[i], cmap=cm.gray)
    res = y_train[nums[i]]
    if res == 10:
        res = 0
    ax.set_title(f'Number {res}')

plt.show()


class MulticlassLogisticRegression:
    classifier = LogisticRegression

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.classifiers = [
            self.classifier(fit_method='gradient_descent', learning_rate=0.5, regularized=True, reg_L=0.1)
            for i in range(num_classes)
        ]
    
    def fit(self, X, y):
        for i in range(self.num_classes):
            y_train = (y == i).astype(int)
            self.classifiers[i].fit(X, y_train)
    
    def predict(self, X):
        h = []
        for cls in self.classifiers:
            h.append(cls.calculate_hypotesis(X))
            
        return np.argmax(np.array(h), axis=0)
    
    
cls_mult = MulticlassLogisticRegression()
cls_mult.fit(X_train, y_train)
pred_value = cls_mult.predict(X_train[-1])
print(f"Predicted class: {pred_value}, actual class: {y_train[-1]}")


def accuracy(cls, X, y):
    error = cls.predict(X) - y
    return 1.0 - (float(np.count_nonzero(error)) / len(error))

acc = accuracy(cls_mult, X_train, y_train)
print(f"Accuracy: {acc}")
