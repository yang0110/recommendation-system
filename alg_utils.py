import numpy as np
import scipy.sparse
import ctypes
import os
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 

def fill_empty_row_or_col(matrix, fill_value=1.0):
    matrix=matrix.copy()
    empty_rows = (matrix.sum(axis=1).A1 == 0)
    random_items = np.random.randint(0, matrix.shape[1], empty_rows.sum())
    matrix[empty_rows, random_items] = fill_value
    empty_cols = (matrix.sum(axis=0).A1 == 0)
    if empty_cols.any():
        random_users=np.random.randint(0, matrix.shape[0], empty_cols.sum())
        matrix[random_users, empty_cols] = fill_value
    return matrix


def train_Slim():
    return None


class MF_BPR_net(nn.Module):
    def __init__(self, user_num, item_num, dimension):
        self.dimension = dimension
        self.user_num = user_num
        self.item_num = item_num
        self.user_emb = nn.Embedding(self.user_num, self.dimension)
        self.item_emb = nn.Embedding(self.item_num, self.dimension)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user_f = self.user_emb(user)
        item_f_i = self.item_emb(item_i)
        item_f_j = self.item_emb(item_j)
        rating_i = (user_f*item_f_i).sum(dim=-1)
        rating_j = (user*item_f_j).sum(dim=-1)
        return rating_i, rating_j 

class NCF_net(nn.Module):
    def __init__(self, user_num, item_num, dimension):
        self.dimension = dimension
        self.user_num = user_num
        self.item_num = item_num
        self.user_emb = nn.Embedding(self.user_num, self.dimension)
        self.item_emb = nn.Embedding(self.item_num, self.dimension)
        self.linear_1 = nn.Linear(2*self.dimension, 64)
        self.linear_2 = nn.Linear(64, 32)
        self.relu = nn.ReLU(32, 1)
        
    def forward(self, user, item):
        user_f = self.user_emb(user)
        item_f = self.item_emb(item)
        join = torch.cat((user_f, item_f), -1)
        a = self.linear_1(join)
        b = self.linear_2(a)
        predict = self.relu(b)
        return predict 

