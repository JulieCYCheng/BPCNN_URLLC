import numpy as np
import BP_Decoder
import ConvNet
import tensorflow as tf
import datetime
import os


def generate_snr_set(top_config):
    SNR_set_end = top_config.SNR_set_start + 0.5 * top_config.SNR_set_size
    SNR_set = np.arange(top_config.SNR_set_start, SNR_set_end, 0.5, dtype=np.float32)

    return SNR_set


def generate_noise_samples(code, top_config, train_config, net_config, gen_data_for, bp_iter_num, num_of_cnn, model_id):

    global batch_size_each_SNR
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
        print('>>> Invalid objective of data generation! (ibc.py)')
        exit(0)

    # BP iteration
    if np.size(bp_iter_num) != num_of_cnn + 1:
        print('>>> Error: the length of bp_iter_num is not correct! (ibc.py)')
        exit(0)
    bp_decoder = BP_Decoder.BP_NetDecoder(H_matrix, batch_size_each_SNR)

    conv_net = {}
    denoise_net_in = {}
    denoise_net_out = {}
    intf_net_out = {}

    for net_id in range(num_of_cnn):        # TODO: Doesn't work if num_of_cnn=0
        conv_net[net_id] = ConvNet.ConvNet(net_config, None, net_id)
        denoise_net_in[net_id], denoise_net_out[net_id],  intf_net_out[net_id] = conv_net[net_id].build_network()

    # Init gragh
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Restore cnn networks before the target CNN        # TODO: Doesn't work if num_of_cnn=0
    for net_id in range(num_of_cnn):                    # TODO: Why restore here?
        conv_net[net_id].restore_network_with_model_id(sess, net_config.total_layers, model_id[0:(net_id + 1)])

    start = datetime.datetime.now()

    if gen_data_for == 'Training':
        if not os.path.isdir(train_config.training_folder):
            os.mkdir(train_config.training_folder)
        fout_est_noise = open(train_config.training_feature_file, 'wb')
        fout_real_noise = open(train_config.training_noise_label_file, 'wb')
        fout_real_intf = open(train_config.training_intf_label_file, 'wb')
    elif gen_data_for == 'Test':
        if not os.path.isdir(train_config.test_folder):
            os.mkdir(train_config.test_folder)
        fout_est_noise = open(train_config.test_feature_file, 'wb')
        fout_real_noise = open(train_config.test_noise_label_file, 'wb')
        fout_real_intf = open(train_config.test_intf_label_file, 'wb')
    else:
        print('>>> Invalid objective of data generation! (ibc.py)')
        exit(0)


