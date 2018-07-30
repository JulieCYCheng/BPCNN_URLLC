import numpy as np


def generate_snr_set(top_config):
    SNR_set_end = top_config.SNR_set_start + 0.5 * top_config.SNR_set_size
    SNR_set = np.arange(top_config.SNR_set_start, SNR_set_end, 0.5, dtype=np.float32)

    return SNR_set


def generate_noise_samples(code, top_config):

    G_matrix = code.G_matrix
    H_matrix = code.H_matrix

    top_config.SNR_set_gen_training = generate_snr_set(top_config)



