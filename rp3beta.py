# create based on Cesare Bernardis's work
# https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation/blob/master/GraphBased/RP3betaRecommender.py
from scipy import sparse
from sklearn.preprocessing import normalize
import numpy as np 

class RP3beta():
	def __init__(self, train_inter_matrix, test_inter_matrix, alpha=1.0, beta=0.6, K=100):
		self.train_inter_matrix = train_inter_matrix
		self.test_inter_matrix = test_inter_matrix
		self.user_num = self.train_inter_matrix.shape[0]
		self.item_num = self.train_inter_matrix.shape[1]
		self.alpha = alpha
        self.beta = beta 
		self.K = K 
		self.Pui = np.zeros((self.user_num, self.item_num))
		self.Piu = np.zeros((self.item_num, self.user_num))
		self.RP3 = None


	def generate_pui(self):
		self.Pui = normalize(self.train_inter_matrix, norm='l1', axis=1)
		X_bool = self.train_inter_matrix.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()
        degree = np.zeros(self.train_inter_matrix.shape[1])
        nonZeroMask = X_bool_sum!=0.0
        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)
        self.Piu = normalize(X_bool, norm='l1', axis=1)
   		self.Pui = self.Pui.power(self.alpha)
   		self.Piu = self.Piu.power(self.alpha)

   	def generate_RP3(self):
   		# P3= Pui*Piu*Pui
        block_dim = 200
        d_t = self.Piu
        dataBlock = 10000000
        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)
        numCells = 0
        for current_block_start_row in range(0, self.Pui.shape[1], block_dim):
            if current_block_start_row + block_dim > self.Pui.shape[1]:
                block_dim = self.Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * self.Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.K]
                notZerosMask = row_data[best] != 0.0
                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]
                for index in range(len(values_to_add)):
                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

        self.RP3 = sparse.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(self.Pui.shape[1], self.Pui.shape[1]))
        self.RP3 = normalize(self.P3, norm='l1', axis=1)

    def fit(self):
        self.generate_pui()
        self.generate_RP3()

    def predict(self, user):
		ranking = list(np.argsort(self.RP3[user])[::-1])[:K]
		return ranking 




		
