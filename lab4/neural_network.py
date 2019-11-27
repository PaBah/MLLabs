import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline


mat = loadmat('ex4data1.mat')
X_train, y_train = mat['X'], mat['y']
y_train = y_train.reshape(y_train.shape[0])
y_train = np.where(y_train != 10, y_train, 0)


mat_weights = loadmat('ex4weights.mat')
theta1 = mat_weights['Theta1']
theta2 = mat_weights['Theta2']
s_L = [400, 25, 10]


def sigmoid(z):
    return 1 / (1 + np.e ** (-z))


def insert_ones(x):
    if len(x.shape) == 1:
        return np.insert(x, 0, 1)
    return np.column_stack((np.ones(x.shape[0]), x))


def unroll(weights):
    result = np.array([])

    for theta in weights:
        result = np.concatenate((result, theta.flatten()))

    return result


def roll(weights):
    weights = np.array(weights)
    thetas = []
    left = 0

    for i in range(len(s_L) - 1):
        x, y = s_L[i + 1], s_L[i] + 1
        right = x*y
        thetas.append(weights[left:left + right].reshape(x, y))
        left = right

    return thetas


def forward_prop(x, thetas, cache=False):
    cur_activation = x.copy()
    activations = [cur_activation]

    for theta_i in thetas:
        temp_a = insert_ones(cur_activation)
        z_i = theta_i.dot(temp_a.T).T
        cur_activation = sigmoid(z_i)
        if cache:
            activations.append(cur_activation)

    return activations if cache else cur_activation


def accuracy(hyp, y):
    return 1 - ((np.count_nonzero(hyp.argmax(axis=1) - y) / y.shape[0]))


weights = [theta1, theta2]
hypotesis = forward_prop(X_train, weights)
acc = accuracy(hypotesis, y_train)
print(f"Accuracy on training set: {acc}")


def to_one_spot(y, num_classes=10):
    y_one_spot = np.zeros((y.shape[0], num_classes))

    for i, y_i in enumerate(y):
        y_one_spot[i][y_i] = 1

    return y_one_spot


y_one_spot = to_one_spot(y_train)
print(y_train.shape, y_one_spot.shape)


ONE = 1.0 + 1e-15

def cost_func(X, y, weights):
    total_cost = 0
    K = y.shape[1]
    hyp = forward_prop(X, weights)
    for k in range(K):
        y_k, hyp_k = y[:, k], hyp[:, k]
        cost_trues =  y_k * np.log(hyp_k)
        cost_falses = (1 - y_k) * np.log(ONE - hyp_k)
        cost = cost_trues + cost_falses
        total_cost += cost
    return -total_cost.sum() / y.shape[0]


def cost_func_regularized(X, y, weights, reg_L=1):
    weights = roll(weights)
    reg = 0
    cost = cost_func(X, y, weights)

    for theta in weights:
        theta_R = theta[:, 1:]
        reg += (theta_R ** 2).sum()

    return cost + (reg_L / 2 / y.shape[0]) * reg


def activation_der(act):
    return act * (1 - act)


INIT_EPS = 1e-2

def initialize_weights():
    weights = []

    for i in range(len(s_L) - 1):
        theta = np.random.random((s_L[i + 1], s_L[i] + 1)) * 2 * INIT_EPS - INIT_EPS
        weights.append(theta)

    return unroll(weights)


init_weights = initialize_weights()
print(init_weights)


def back_prop(X, y, weights, reg_L=0):
    M = y.shape[0]
    L = len(weights)
    act = forward_prop(X, weights, cache=True)
    Deltas = [np.zeros(theta.shape) for theta in weights]
    
    for i in range(M):
        delta_L = y[i] - act[-1][i]
        deltas = [delta_L]

        for l in reversed(range(1, L)):
            d = np.dot(weights[l].T, deltas[-1]) * activation_der(insert_ones(act[l][i]))
            deltas.append(d[1:])

        deltas = list(reversed(deltas))
        for l in range(L):
            Deltas[l] = Deltas[l] + np.dot(deltas[l].reshape((-1,1)), insert_ones(act[l][i]).reshape((1, -1)))
    
    D = []
    for l, Delta_l in enumerate(Deltas):
        D_l = Delta_l / M
        D_l[:, 1:] += reg_L * weights[l][:, 1:]
        D.append(D_l)
    
    return D


Deltas = back_prop(X_train, y_one_spot, weights)


GRAD_EPS = 1e-4

def check_gradient(X, y, thetas, D_vec, edge=500):
    def J(theta):
        return cost_func_regularized(X, y, theta)

    N = min(len(thetas), edge)
    grad_approx = np.zeros(N)

    for i in range(N):
        theta_plus, theta_minus = thetas.copy(), thetas.copy()
        theta_plus[i] += GRAD_EPS
        theta_minus[i] -= GRAD_EPS
        grad_approx[i] = (J(theta_plus) - J(theta_minus)) / (2 * GRAD_EPS)

    return np.allclose(grad_approx, D_vec[:N], atol=1)


check_gradient(X_train, y_one_spot, unroll(weights), unroll(Deltas))


def train(X, y, reg_L, l_rate=0.5, max_steps=1e+3, with_history=False):
    history = []
    cur_weights = initialize_weights()
    cur_loss = cost_func_regularized(X, y, cur_weights, reg_L)

    cur_step = 0
    while cur_step < max_steps:
        cur_step += 1
        new_weights = update_weights(X, y, cur_weights, l_rate, reg_L)
        new_loss = cost_func_regularized(X, y, new_weights, reg_L)

        if np.isnan(new_loss):
            break

        history.append((new_weights, new_loss))
        cur_weights = new_weights
        cur_loss = new_loss

    if with_history:
        return history
    
    return cur_weights


def update_weights(X, y, weights, l_rate, reg_L):
    gradient = unroll(back_prop(X, y, roll(weights), reg_L))
    gradient *= l_rate
    return weights + gradient


grad_weights = train(X_train, y_one_spot, reg_L=0.003, l_rate=0.5)


hypotesis = forward_prop(X_train, roll(grad_weights))
acc = accuracy(hypotesis, y_train)
print(f"Accuracy on training set: {acc}")


def plot_hidden_layer(X, w):
    hyp = forward_prop(X, roll(w), cache=True)
    print(f"Accuracy on training set: {accuracy(hyp[-1], y_train)}")
    hidden_layer = hyp[1]
    
    nums = list(range(150, 5000, 250))
    size = int(np.sqrt(hidden_layer.shape[1]))
    pictures = [hidden_layer[i].reshape((size, size)) for i in nums]

    fig, axs = plt.subplots(1, 20, figsize=(20, 0.85))
    for i, ax in enumerate(axs.flatten()):
        ax.pcolor(pictures[i], cmap=cm.gray)
        ax.axis('off')

    plt.show()
    
    
plot_hidden_layer(X_train, grad_weights)


reg_L_list = [1, 0.3, 0.1, 0.03, 0.01, 0.003]
steps = [20, 50, 75, 400, 600, 1000]

for i, reg_l in enumerate(reg_L_list):
    weights_l = train(X_train, y_one_spot, reg_L=reg_l, l_rate=0.5, max_steps=steps[i])
    plot_hidden_layer(X_train, weights_l)
