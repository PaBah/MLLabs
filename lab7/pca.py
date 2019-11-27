import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D


def load_file(filename, keys=None):
    if keys is None:
        keys = ['X', 'y']
    mat = loadmat(filename)
    ret = tuple([mat[k].reshape(mat[k].shape[0]) if k.startswith('y') else mat[k] for k in keys])
    return ret


X, = load_file('ex7data1.mat', keys=['X'])
X = X - X.mean(axis=0)  # mean normalization
print(f'X shape: {X.shape}')


ax = plt.subplot()
ax.plot(X[:, 0], X[:, 1], marker='o', linestyle="None", markersize=3)
ax.set_aspect('equal')
ax.set_title("Principle Component Analysis data")
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()


def cov(X):
    return np.dot(X.T, X) / X.shape[0]


def get_eigenvectors(X):
    Sigma = cov(X)
    return np.linalg.svd(Sigma, full_matrices=False)

U, S, V = get_eigenvectors(X)
print(U)


mu = X.mean(axis=0)
projected_data = np.dot(X, U)
sigma = projected_data.std(axis=0).mean()
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], marker='o', linestyle="None", markersize=3)
for ind, axis in enumerate(U):
    start, end = mu, mu + (S[ind] + sigma) * axis
    ax.annotate(
        '', xy=end, xycoords='data',
        xytext=start, textcoords='data',
        arrowprops=dict(facecolor='red', width=2.0))
ax.set_aspect('equal')
ax.set_title("Principle Component Analysis data with eigenvectors")
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()


def transform(X, k=None):
    if k is None:
        k = X.shape[1] - 1
    U, *_ = get_eigenvectors(X)
    U_reduce = U[:, :k]
    return np.dot(X, U_reduce)


X_reduced = transform(X)
print(f'Reduced X shape: {X_reduced.shape}')


def inverse_transform(X, Z, k=None):
    if k is None:
        k = X.shape[1] - 1
    U, *_ = get_eigenvectors(X)
    U_reduce = U[:, :k]
    return np.dot(Z, U_reduce.T)


X_approx = inverse_transform(X, X_reduced)
print(f'Inversed X shape: {X_approx.shape}')


mu = X.mean(axis=0)
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], marker='o', linestyle="None", markersize=3)

U, *_ = get_eigenvectors(X)
m = U[0][1] / U[0][0]
ax.plot(np.linspace(-3, 2.5, 10), m * np.linspace(-3, 2.5, 10), 
        color="black", linestyle="--")

ax.scatter(X_approx[:, 0], X_approx[:, 1], c="r", marker="x", s=32)
ax.set_aspect('equal')

ax.set_title("Principle Component Analysis data with eigenvectors")
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()


X_faces, = load_file('ex7faces.mat', keys=['X'])
print(f'X faces shape: {X_faces.shape}')


rand_inds = np.random.choice(np.arange(0, 5000), 100)
fig, axs = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(20,20)) 
axs = axs.flatten()
graymap = plt.get_cmap("gray")

for i, indx in enumerate(rand_inds):
    im_mat = np.reshape(X_faces[indx, :], (32, 32), order="F")
    axs[i].imshow(im_mat, cmap=graymap, interpolation="None")
    axs[i].xaxis.set_visible(False)
    axs[i].yaxis.set_visible(False) 
    

X_faces = X_faces - X_faces.mean(axis=0)
Uf, Sf, Vf = get_eigenvectors(X_faces)


def plot_components(n_components):
    size = int(np.sqrt(n_components))
    fig, axs = plt.subplots(size, size, sharex=True, sharey=True, figsize=(10,10)) 
    fig.suptitle(f"{n_components} PCA Eigenfaces", fontsize=18)
    axs = axs.flatten() 
    graymap = plt.get_cmap("gray")

    for i, indx in enumerate(range(n_components)):
        im_mat = np.reshape(Vf[indx, :], (32, 32), order="F")
        axs[i].imshow(im_mat, cmap=graymap, interpolation="None")
        axs[i].xaxis.set_visible(False)
        axs[i].yaxis.set_visible(False) 
        
plot_components(36)
plot_components(100)

A, = load_file('bird_small.mat', keys=['A'])
print(f'Shape: {A.shape}')
fig, axs = plt.subplots(ncols=1, figsize=[12, 5])
axs.imshow(A)
plt.show()


Ax = np.reshape(A, [A.shape[0] * A.shape[1], A.shape[2]])
Ax = Ax - Ax.mean(axis=0)
Ax_reduced = transform(Ax)
Ax_approx = inverse_transform(Ax, Ax_reduced)


fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(xs=Ax_approx[:, 0], ys=Ax_approx[:, 1], zs=Ax_approx[:, 2], cmap=cm.coolwarm, s=1)

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(Ax_reduced[:, 0], Ax_reduced[:, 1], cmap=cm.coolwarm, s=2)
plt.show()
