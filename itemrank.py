# create based on arashkhoeini's work
# https://github.com/arashkhoeini/itemrank
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, cosine_similarity
import networkx as nx
from sklearn.preprocessing import normalize

class ItemRank():
	def __init__(self, train_inter_matrix, test_inter_matrix, alpha=0.1, N=20, K=100):
		self.user_num = train_inter_matrix.shape[0]
		self.item_num = train_inter_matrix.shape[1]
		self.train_inter_matrix = train_inter_matrix
		self.test_inter_matrix = test_inter_matrix
		self.item_correlation = np.zeros((self.item_num, self.item_num))
		self.alpha = alpha
		self.N = N
		self.K = K
		self.IR_matrix = np.ones((self.user_num, self.item_num))

	def generate_item_correlation(self):
		self.item_correlation = cosine_similarity(self.train_inter_matrix.T)
		self.item_correlation = normalize(self.item_correlation, norm='l1', axis=1)

	def generate_d(self, user):
		user_d = self.train_inter_matrix[user]
		return user_d 

	def update_IR(self, user):
		user_d = self.train_inter_matrix[user]
		self.IR_matrix[user] = self.alpha * np.dot(self.item_correlation, self.IR_matrix[user]) + (1-self.alpha) * user_d

	def ranking_item(self, user, K=20):
		ranking = list(np.argsort(self.IR_matrix[user])[::-1])[:K]
		return ranking 

	def fit(self):
		self.generate_item_correlation()

	def predict(self, user):
		for user in range(self.user_num):
			for n in range(self.N):
				self.update_IR(user)

		ranking = ranking_item(user)
		return ranking 





