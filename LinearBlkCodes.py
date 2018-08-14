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

def encode_and_transmit(G_matrix, SNR, batch_size, noise_io, intf_io, top_config, rng=0):
    K, N = np.shape(G_matrix)

    # Random generated binary message
    if rng == 0:
        x_bits = np.random.randint(0, 2, size=(batch_size, K))
    else:
        x_bits = rng.randint(0, 2, size=(batch_size, K))

    # Encode
    u_coded_bits = np.mod(np.matmul(x_bits, G_matrix), 2)  # shape: (B, N)

    # BPSK
    s_mod = u_coded_bits * (-2) + 1         # 0 to +1, 1 to -1

    # Noise
    ch_noise_normalize = noise_io.generate_noise(batch_size)
    ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
    ch_noise = ch_noise_normalize * ch_noise_sigma

    # Interference
    intf_labels = intf_io.generate_intf_labels(batch_size)                          # shape: (B, 1)
    intf_bits = np.random.randint(0, 2, size=(batch_size, top_config.intf_len))     # shape: (B, F)
    intf_mod = intf_bits * (-2) + 1
    intf_signal = np.zeros((batch_size, N))
    intf_signal[:, 0:top_config.intf_len] = intf_mod[:, :]                          # shape: (B, N)

    y_receive = s_mod + ch_noise + np.multiply(intf_signal, intf_labels)

    LLR = y_receive * 2.0 / (ch_noise_sigma * ch_noise_sigma)
    return x_bits, u_coded_bits, s_mod, ch_noise, intf_labels, y_receive, LLR