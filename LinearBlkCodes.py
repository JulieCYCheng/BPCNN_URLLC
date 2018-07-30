import numpy as np


class LDPC:
    def __init__(self, N, K, G_file, H_file):
        self.N = N
        self.K = K
        self.G_matrix, self.H_matrix = self.init_LDPC_matrix(G_file, H_file)

    def init_LDPC_matrix(self, G_file, H_file):
        G_matrix_row_col = np.loadtxt(G_file, dtype=np.int32)
        H_matrix_row_col = np.loadtxt(H_file, dtype=np.int32)
        G_matrix = np.zeros([self.K, self.N], dtype=np.int32)
        H_matrix = np.zeros([self.N - self.K, self.N], dtype=np.int32)
        G_matrix[G_matrix_row_col[:, 0], G_matrix_row_col[:, 1]] = 1
        H_matrix[H_matrix_row_col[:, 0], H_matrix_row_col[:, 1]] = 1

        return G_matrix, H_matrix
