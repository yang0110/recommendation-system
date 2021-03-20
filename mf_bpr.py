import torch 
import torch.nn as nn 
import numpy as np 
import torch.optim as optim
import torch.utils.data as data
from alg_utils import * 

class MF_BPR():
	def __init__(self, user_num, item_num, dimension, train_inter_matrix, epochs, batch_size, K, learning_rate=0.001):
		self.user_num = user_num 
		self.item_num = item_num 
		self.dimension = dimension
		self.train_inter_matrix = train_inter_matrix
		self.epochs = epochs 
		self.batch_size = batch_size
		self.K = K
		self.learning_rate = learning_rate
		self.model = None 
		self.optimizer = None

	def _init_net(self):
		self.model = MF_BPR_NN(self.user_num, self.item_num, self.dimension)
		self.optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=0.01)

	def fit(self):
		train_loader = DataLoader()
		for epoch in range(self.epochs):
			model.train()
			for user, item_i, item_j in train_loader:
				model.zero_grad()
				rating_i, rating_j = self.model(user, item_i, item_j)
				loss = -(rating_i-rating_j).sigmoid().log().sum()
				loss.backward()
				self.optimizer.step()

	def predict(self, user):
		item_embs = self.model.embedding.weights()
		user_emb = self.
		ratings = np.dot(item_embs, user_emb)
		ranking = list(np.argsort(ratings)[::-1])[:self.K]
		return ranking 			


