import numpy as np

class LDPC:
    def __init__(self, N, K, G_file, H_file):
        self.N = N
        self.K = K
        self.G_file = G_file
        self.H_file = H_file