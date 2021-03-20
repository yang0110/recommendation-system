import torch
import torch.nn as nn
import torch.optim as optim 
from alg_utils import *

class NCF():
    def __init__(self, train_inter_matrix, dimension, learning_rate, K, epochs):
        self.train_inter_matrix = train_inter_matrix
        self.user_num = self.train_inter_matrix.shape[0]
        self.item_num = self.train_inter_matrix.shape[1]
        self.learning_rate = learning_rate
        self.K = K
        self.epochs = epochs
        self.model = None
        self.optimizer = None

    def _init_net(self):
        self.model = NCF_net(self.user_num, self.item_num, self.dimension)
        nn.init.xavier_uniform_(self.model.parameters())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)

    def fit(self):
        train_loader = DataLoader()
        for epoch in range(self.epochs):
            model.train()
            for user, item, label in train_loader:
                model.zero_grad()
                prediction = self.model(user, item)
                loss = nn.BCEWithLogitsLoss(prediction, label)
                loss.backward()
                self.optimizer.step()

    def predict(self, user):
        item_embs = self.model.embedding.weights()
        user_emb = self.
        ratings = np.dot(item_embs, user_emb)
        ranking = list(np.argsort(ratings)[::-1])[:self.K]
        return ranking      


