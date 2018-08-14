import numpy as np
import BP_Decoder
import ConvNet
import tensorflow as tf
import datetime
import os
import LinearBlkCodes as lbc


def generate_snr_set(top_config):
    SNR_set_end = top_config.SNR_set_start + 0.5 * top_config.SNR_set_size
    SNR_set = np.arange(top_config.SNR_set_start, SNR_set_end, 0.5, dtype=np.float32)

    return SNR_set

def denoising_and_calc_LLR_awgn(res_noise_power, y_receive, output_pre_decoder, net_in, net_out, intf_out, sess):
    # estimate noise with cnn denoising
    noise_before_cnn = y_receive - (output_pre_decoder * (-2) + 1)
    noise_after_cnn = sess.run(net_out, feed_dict={net_in: noise_before_cnn})
    predicted_intf_ind = sess.run(intf_out, feed_dict={net_in: noise_before_cnn})

    # calculate the LLR for next BP decoding
    s_mod_plus_res_noise = y_receive - noise_after_cnn
    LLR = s_mod_plus_res_noise * 2.0 / res_noise_power
    return LLR, predicted_intf_ind

def generate_noise_samples(code, top_config, train_config, net_config, gen_data_for, bp_iter_num, num_of_cnn, model_id,
                           noise_io, intf_io):

    global batch_size_each_SNR, total_batches
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

    # Generating data
    for ik in range(total_batches):
        for SNR in top_config.SNR_set_gen_training:
            x_bits, _, _, ch_noise, intf_labels, y_receive, LLR = lbc.encode_and_transmit(G_matrix, SNR, batch_size_each_SNR, noise_io, intf_io, top_config)

            for iter in range(0, num_of_cnn + 1):
                # BP decoder
                u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), bp_iter_num[iter])

                # CNN
                if iter != num_of_cnn:
                    res_noise_power = conv_net[iter].get_res_noise_power(model_id).get(np.float32(SNR))
                    LLR, predicted_intf_ind = denoising_and_calc_LLR_awgn(res_noise_power, y_receive, u_BP_decoded, denoise_net_in[iter], denoise_net_out[iter], intf_net_out[iter], sess)

            # reconstruct noise
            noise_before_cnn = y_receive - (u_BP_decoded * (-2) + 1)
            noise_before_cnn = noise_before_cnn.astype(np.float32)
            noise_before_cnn.tofile(fout_est_noise)     # write features to file
            ch_noise.tofile(fout_real_noise)            # write noise labels to file
            intf_labels.tofile(fout_real_intf)          # write interference labels to file

    fout_real_noise.close()
    fout_est_noise.close()

    sess.close()
    end = datetime.datetime.now()

    print("Time: %ds" % (end - start).seconds)
    print("Finish generating %s data" % gen_data_for)
