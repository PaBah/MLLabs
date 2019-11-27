import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

from scipy.io import loadmat

def load_file(filename, keys=None):
    if keys is None:
        keys = ['X', 'y']
    mat = loadmat(filename)
    ret = tuple([mat[k].reshape(mat[k].shape[0]) if k.startswith('y') else mat[k] for k in keys])
    return ret


X, = load_file('ex6data1.mat', keys=['X'])
print(f'X shape: {X.shape}')


def initialize_cetroids(X, K):
    idx = np.random.randint(len(X), size=K)
    return X[idx, :]


def get_clusters(X, centroids):
    c = np.zeros(len(X), dtype=int)

    for i, x in enumerate(X):
        c[i] = ((x - centroids)**2).sum(axis=1).argmin()

    return c


def update_centroids(X, clusters, K):
    new_centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        if len(X[clusters == k]) == 0:
            continue
        new_centroids[k] = X[clusters == k].mean(axis=0)

    return new_centroids


def cost_func(X, c, centroids):
    M = X.shape[0]
    cost = 0
    for i in range(M):
        cost += ((X[i] - centroids[int(c[i])]) ** 2).sum()
    return cost / M


def k_means_algo(X, K):
    Result = namedtuple('Result', ['clusters', 'centroids_history'])
    centroids = initialize_cetroids(X, K)
    clusters = get_clusters(X, centroids)
    centroids_history = [centroids]

    while True:
        cur_centroids = update_centroids(X, clusters, K)
        cur_clusters = get_clusters(X, cur_centroids)
        centroids_history.append(cur_centroids)

        if (cur_clusters == clusters).all():
            break
        
        clusters = cur_clusters

    return Result(clusters=clusters, centroids_history=centroids_history)


from collections import namedtuple

def k_means(X, K, max_iter=100):
    Result = namedtuple('Result', ['clusters', 'centroids', 'centroids_history', 'best_cost'])
    cur_cost = np.inf
    result = None

    for i in range(max_iter):
        centroids = initialize_cetroids(X, K)
        cur_result = k_means_algo(X, K)
        cost = cost_func(X, cur_result.clusters, cur_result.centroids_history[-1])

        if cost < cur_cost:
            result = cur_result
            cur_cost = cost

    return Result(clusters=result.clusters, centroids=result.centroids_history[-1],
                  centroids_history=result.centroids_history, best_cost=cur_cost)

result = k_means(X, 3)


plt.scatter(X[:, 0], X[:, 1], c=result.clusters)
x1, y1, x2, y2, x3, y3 = [], [], [], [], [], []
for centr in result.centroids_history:
    x1.append(centr[0][0])
    y1.append(centr[0][1])
    x2.append(centr[1][0])
    y2.append(centr[1][1])
    x3.append(centr[2][0])
    y3.append(centr[2][1])
plt.plot(x1, y1, x2, y2, x3, y3, marker='x')
plt.title('Clustering and movement of centroids')
plt.show()


A, = load_file('bird_small.mat', keys=['A'])
print(f'Image shape: {A.shape}')


def compress(A, n_colors=16):
    X = np.reshape(A, [A.shape[0] * A.shape[1], A.shape[2]])
    result = k_means(X, n_colors, max_iter=1)
    clusters = result.clusters
    new_colors = np.round(result.centroids).astype(np.uint8)

    image = X.copy()
    for i in range(X.shape[0]):
        image[i, :] = new_colors[clusters[i]]

    return image.reshape(A.shape)

compressed_A = compress(A)


fig, axs = plt.subplots(ncols=2, figsize=[12, 5])
fig.suptitle("Original and compressed image", fontsize=18)
axs[0].imshow(A)
axs[1].imshow(compressed_A)
plt.show()


import cv2

img = cv2.imread('image_example.jpg', cv2.IMREAD_UNCHANGED)
compressed_img = compress(img)

fig, axs = plt.subplots(ncols=2, figsize=[12, 5])
fig.suptitle("Original and compressed image", fontsize=18)
axs[0].imshow(img)
axs[1].imshow(compressed_img)
plt.show()


from sklearn.cluster import AgglomerativeClustering

def compress_hierarchical_clusters(img, n_colors=16):
    X = np.reshape(img, [img.shape[0] * img.shape[1], img.shape[2]])

    cluster = AgglomerativeClustering(n_clusters=n_colors, affinity='euclidean', linkage='ward')
    labels = cluster.fit_predict(X)
    centroids = update_centroids(X, labels, n_colors).reshape((n_colors, 3))
    new_colors = np.round(centroids).astype(np.uint8)

    image = X.copy()
    for i in range(X.shape[0]):
        image[i, :] = new_colors[labels[i]]

    return image.reshape(img.shape)

compressed_img_hier_cl = compress_hierarchical_clusters(img)

fig, axs = plt.subplots(ncols=2, figsize=[12, 5])
fig.suptitle("Compressed images with K-means and Hierarchical Clustering", fontsize=18)
axs[0].imshow(compressed_img)
axs[1].imshow(compressed_img_hier_cl)
plt.show()


