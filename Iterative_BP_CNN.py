import numpy as np
import BP_Decoder


def generate_snr_set(top_config):
    SNR_set_end = top_config.SNR_set_start + 0.5 * top_config.SNR_set_size
    SNR_set = np.arange(top_config.SNR_set_start, SNR_set_end, 0.5, dtype=np.float32)

    return SNR_set


def generate_noise_samples(code, top_config, train_config, gen_data_for):

    G_matrix = code.G_matrix
    H_matrix = code.H_matrix

    top_config.SNR_set_gen_training = generate_snr_set(top_config)
    print('SNR set for generating training data: %s' % np.array2string(top_config.SNR_set_gen_training))

    if gen_data_for == 'Training':
        batch_size_each_SNR = int(train_config.training_minibatch_size // top_config.SNR_set_size)
        total_batches = int(train_config.training_sample_num // train_config.training_minibatch_size)
    elif gen_data_for == 'Test':
        batch_size_each_SNR = int(train_config.test_minibatch_size // top_config.SNR_set_size)
        total_batches = int(train_config.test_sample_num // train_config.test_minibatch_size)
    else:
        print('Invalid objective of data generation!')
        exit(0)

    # BP iteration
    bp_decoder = BP_Decoder.BP_NetDecoder(H_matrix, batch_size_each_SNR)



