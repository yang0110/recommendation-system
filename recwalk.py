# Created base don Tim Toebrock's work 
# https://github.com/titoeb/RecModel
import numpy as np
import scipy.sparse
import ctypes
import os
from alg_utils import *

class RecWalk():
	def __init__(self, train_inter_matrix, test_inter_matrix, k_steps, K, alpha, phi, damping, reg_1, reg_2, max_iter, tolerance):
		self.train_inter_matrix = train_inter_matrix
		self.test_inter_matrix = test_inter_matrix
		self.k_steps = k_steps 
		self.K = K
		self.damping = damping
		self.alpha = alpha
		self.phi = phi
		self.reg_1 = reg_1
		self.reg_2 = reg_2
		self.max_iter = max_iter
		self.tolerance = tolerance
		self.user_num = self.train_inter_matrix.shape[0]
		self.item_num = self.train_inter_matrix.shape[1]
		self.A_g = None 
		self.W = None 
		self.M_I = None
		self.M = None 
		self.H = None
		self.P = None

	def generate_A_g(self):
	    zeros_upper_left = scipy.sparse.csr_matrix((self.user_num, self.user_num), dtype=np.float32) 
	    upper_half = scipy.sparse.hstack([zeros_upper_left, self.train_inter_matrix], format='csr')
	    zeros_lower_right = scipy.sparse.csr_matrix((self.item_num, self.item_num), dtype=np.float32)    
	    lower_half = scipy.sparse.hstack([self.train_inter_matrix.T, zeros_lower_right], format='csr')
	    self.A_g = scipy.sparse.vstack([upper_half, lower_half], format='csr')

	def generate_H(self):
	    row_sums = self.A_g.sum(axis=1).A1
	    self.H = scipy.sparse.diags(diagonals=(1 / row_sums), offsets=0, dtype=np.float32, format='csr')

	def generate_M_I(self, W_indptr, W_indices, W_data):
	    W = scipy.sparse.csr_matrix((W_data, W_indices, W_indptr), shape=(self.item_num, self.item_num), dtype=np.float32)
	    W_normalized = W.copy()
	    row_sums = W.sum(axis=1).A1
	    row_sum_max = row_sums.max()
	    W_normalized.data /= row_sum_max
	    diag_mat = scipy.sparse.diags(diagonals = 1 - (row_sums / row_sum_max), offsets=0, dtype=np.float32, format='csr')
	    self.M_I = W_normalized + diag_mat

	def generate_M(self):
		I = scipy.sparse.diags(diagonals=np.full(self.user_num, 1, dtype=np.float32), offsets=0, dtype=np.float32, format='csr')
	    zeros_upper_right = scipy.sparse.csr_matrix((self.user_num, self.item_num), dtype=np.float32)    
	    zeros_lower_left = scipy.sparse.csr_matrix((self.item_num, self.user_num), dtype=np.float32)    
	    upper_half = scipy.sparse.hstack([I, zeros_upper_right], format='csr')
	    lower_half = scipy.sparse.hstack([zeros_lower_left, M_i], format='csr')
	    self.M = scipy.sparse.vstack([upper_half, lower_half], format='csr')

	def generate_P(self)
        train_inter_matrix = train_inter_matrix.tocsr().astype(np.float64)
        train_inter_matrix.sort_indices()
        train_mat_csc = train_inter_matrix.tocsc()
        train_mat_csc.sort_indices()
        indptr, indices, data = train_Slim(X=train_mat_csc, alpha=self.alpha, l1_ratio=self.reg_1, max_iter=self.max_iter, tol=self.tolerance)
        train_mat = fill_empty_row_or_col(train_inter_matrix)
        self.generate_A_g()
        self.generate_H()
        self.generate_M_I(indptr, indices, data)
        self.generate_M()
        self.H.data *= self.phi
        self.M.data *= (1 - self.phi)
        self.P = (self.H + self.M).T

    def fit(self):
    	self.generate_P()

    def predict(self, items, users):
        user_vec = np.zeros(self.item_num + self.user_num, dtype=np.float32)
        user_vec[users] = 1.0  
        if self.eval_method == 'k_step':
            for _ in range(self.k):
                user_vec = scipy.sparse.csr_matrix.dot(self.P, user_vec)
            predictions = user_vec[items + self.num_users]
            
        else:
            vec_out = user_vec.copy()
            for _ in range(self.k):
                vec_out = self.damping * scipy.sparse.csr_matrix.dot(self.P, vec_out) + (1-self.damping) * user_vec
                vec_out = vec_out / (np.linalg.norm(vec_out) + 1e-10)
            predictions = vec_out[items + self.num_users]

        return items[np.argpartition(predictions, list(range(-self.K, 0, 1)))[-self.K:]][::-1]


