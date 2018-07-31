import tensorflow as tf
import numpy as np

class GetMatrixForBPNet:
    def __init__(self, H, nzero_r_c):
        self.H = H
        self.R, self.N = np.shape(H)                # R = 144, N = 576
        self.H_col_weights = np.sum(H, axis=0)      # shape = (576, )
        self.H_row_weights = np.sum(H, axis=1)      # shape = (144, )
        self.nzero_r_c = nzero_r_c                  # shape = (2, E)
        self.E = np.size(self.nzero_r_c[1, :])

        self.nzero_ver = self.nzero_r_c[1, :] * self.N + self.nzero_r_c[0, :]
        self.nzero_ver_sort = np.sort(self.nzero_ver)
        self.nzero_c_r = np.append([np.mod(self.nzero_ver_sort, self.N)], [self.nzero_ver_sort // self.N], axis=0)



class BP_NetDecoder:
    def __init__(self, H):
        _, self.N = np.shape(H)
        rr, cc = np.nonzero(H)
        loc_nzero_r_c = np.array([rr, cc])      # shape = (2, E)
        self.E = len(rr)

