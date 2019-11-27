import numpy as np
from scipy.io import loadmat
from scipy.sparse.linalg import svds


def load_file(filename, keys=None):
    if keys is None:
        keys = ['X', 'y']
    mat = loadmat(filename)
    ret = tuple([mat[k].reshape(mat[k].shape[0]) if k.startswith('y') else mat[k] for k in keys])
    return ret


Y, R = load_file('ex9_movies.mat', keys=['Y', 'R'])
print(f'Y shape: {Y.shape}')
print(f'R shape: {R.shape}')


N_FACTORS = 20


class Recommender:
    def __init__(self, n_factors=N_FACTORS, learning_rate=0.5, reg_L=0.1, max_steps=1e+3):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_L = reg_L
        self.max_steps = int(max_steps)
        self.cost_history = []
        
    def fit(self, Y, R):
        self.n_m, self.n_u = Y.shape 
        self.X = np.random.rand(self.n_m, self.n_factors)
        self.Theta = np.random.rand(self.n_factors, self.n_u)

        for cur_step in range(self.max_steps):
            self.gradient_descent(Y, R)
            cost = self.cost_func(Y, R)
            self.cost_history.append(cost)

    def cost_func(self, Y, R):
        hypotesis = np.dot(self.X, self.Theta)
        mean_error = R * (hypotesis - Y)
        mean_squared_error = mean_error ** 2
        cost = mean_squared_error.sum() / 2
        regularized_cost = cost + (self.reg_L / 2) * ((self.X ** 2).sum() + (self.Theta ** 2).sum())
        return regularized_cost
    
    def gradient_descent(self, Y, R):
        hypotesis = np.dot(self.X, self.Theta)
        mean_error = R * (hypotesis - Y)
        dX = np.dot(mean_error, self.Theta.T)
        dTheta = np.dot(self.X.T, mean_error)
        regularized_dX = dX + self.reg_L * self.X
        regularized_dTheta = dTheta + self.reg_L * self.Theta
        self.X -= self.learning_rate * regularized_dX
        self.Theta -= self.learning_rate * regularized_dTheta

    def predict(self, user_id, R, top=5):
        predictions = np.dot(self.X, self.Theta)
        user_ratings = (R[:, user_id] != 1) * predictions[:, user_id]
        return user_ratings.argsort()[-top:][::-1]
    
    
rec = Recommender(learning_rate=0.001, reg_L=10, max_steps=1e+2)
rec.fit(Y, R)


my_ratings, presence = np.zeros(Y.shape[0], dtype=int), np.zeros(R.shape[0], dtype=int)
my_ratings[95], presence[95] = 5, 1  # Terminator 2: Judgment Day (1991)
my_ratings[194], presence[194] = 5, 1  # Terminator, The (1984)
my_ratings[585], presence[585] = 5, 1  # Terminal Velocity (1994)
my_ratings[942], presence[942] = 5, 1  # Killing Zoe (1994)
my_ratings[1216], presence[1216] = 5, 1  # Assassins (1995)
my_ratings[312], presence[312] = 1, 1  # Titanic (1997)
my_ratings[318], presence[318] = 1, 1  # Everyone Says I Love You (1996)
my_ratings[725], presence[725] = 1, 1  # Fluke (1995)

my_Y = np.column_stack((Y, my_ratings))
my_R = np.column_stack((R, presence))
user_id = my_Y.shape[1] - 1


rec = Recommender(learning_rate=0.001, reg_L=10, max_steps=1e+3)
rec.fit(my_Y, my_R)
top_movies = rec.predict(user_id, my_R, top=5)

with open('movie_ids.txt', encoding="ISO-8859-1") as f:
    movie_names = [' '.join(line.split(' ')[1:]).replace('\n', '') for line in f.readlines()]

for movie in np.array(movie_names)[top_movies]:
    print(movie)


class SVDRecommender(Recommender): 
    def fit(self, Y, R):
        self.X, _, self.Theta = svds(Y.astype('float64'), k=N_FACTORS)


svd_rec = SVDRecommender()
svd_rec.fit(my_Y, my_R)
top_mov = svd_rec.predict(user_id, my_R, top=5)
for movie in np.array(movie_names)[top_mov]:
    print(movie)
