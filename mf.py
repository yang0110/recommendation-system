#!/usr/bin/env python3
from sklearn.decomposition import NMF
import scipy.sparse as sps

class MF():
    """ Non Negative Matrix Factorization Recommender
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    """
    def __init__(self, train_inter_matrix, dimension, reg_1, K):
        self.train_inter_matrix = train_inter_matrix
        self.dimension = dimension
        self.reg_1 = reg_1 # l1 regualrizer
        self.K = K

    def fit(self):
        nmf_solver = NMF(n_components  = self.dimension, init = 'random', solver = 'multiplicative_update', beta_loss = 'frobenius', random_state = 2021,l1_ratio = self.reg_1, shuffle = True, verbose = verbose, max_iter = 500)
        nmf_solver.fit(self.train_inter_matrix)

        self.item_features = nmf_solver.components_.copy().T
        self.user_features = nmf_solver.transform(self.train_inter_matrix)

    def predict(self, user):
        user_f = self.user_features[user]
        ratings = np.dot(user_f, self.item_features)
        ranking = list(np.argsort(ratings)[::-1])[:K]
        return ranking
