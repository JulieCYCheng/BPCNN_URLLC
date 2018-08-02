import numpy as np

class TopConfig:
    def __init__(self):

        self.function = 'Train'

        # LDPC
        self.N = 576
        self.K = 432
        self.G_file = format('./LDPC_matrix/LDPC_gen_mat_%d_%d.txt' % (self.N, self.K))
        self.H_file = format('./LDPC_matrix/LDPC_chk_mat_%d_%d.txt' % (self.N, self.K))

        # Noise and interference
        self.intf_prob = 1
        self.intf_ratio = 1 / 12
        self.intf_len = np.int(np.floor(self.N * self.intf_ratio))

        # Training
        self.SNR_set_gen_training = np.array([0])
        self.SNR_set_start = 0.0
        self.SNR_set_size = 7

        # CNN Net
        self.feature_length = self.N
        self.conv_layers_num = 4
        self.dense_layers_num = 1
        self.filter_sizes = np.array([9, 3, 3, 15])
        self.feature_map_nums = np.array([64, 32, 16, 1])
        self.currently_trained_net_id = 0           # TODO: What is this?
        self.cnn_net_num = 1
        self.model_id = np.array([0])

        # BP decoding
        self.BP_iter_nums_gen_data = np.array([5])
        self.BP_iter_nums_simu = np.array([5, 5])

    def parse_cmd_line(self, argv):
        if len(argv) == 1:
            return

        ind = 1

        while ind < len(argv):
            if argv[ind] == '-Func':
                self.function = argv[ind + 1]
                print('Function is set to %s' % self.function)
            elif argv[ind] == '-IntfProb':
                self.intf_prob = float(argv[ind + 1])
                print('Interference probability is set to %g' % self.intf_prob)
            elif argv[ind] == '-SnrStart':
                self.SNR_set_start = float(argv[ind + 1])
            elif argv[ind] == '-SnrSetSize':
                self.SNR_set_size = int(argv[ind + 1])
            elif argv[ind] == '-ConvLayNum':
                self.conv_layers_num = int(argv[ind + 1])
                print('Convolution layers number is set to %d' % self.conv_layers_num)
            elif argv[ind] == '-DenseLayNum':
                self.dense_layers_num = int(argv[ind + 1])
                print('Dense layers number is set to %d' % self.dense_layers_num)
            elif argv[id] == '-BP_IterForGenData':
                self.BP_iter_nums_gen_data = np.fromstring(argv[ind + 1], np.int32, sep=' ')
                print('BP iter for gen data is set to: %s' % np.array2string(self.BP_iter_nums_gen_data))
            elif argv[id] == '-BP_IterForSimu':
                self.BP_iter_nums_simu = np.fromstring(argv[ind + 1], np.int32, sep=' ')
                print('BP iter for simulation is set to: %s' % np.array2string(self.BP_iter_nums_simu))

            else:
                print('>>> Command not recognized: %s' % argv[ind])
                exit(0)
            ind = ind + 2


class TrainConfig:
    def __init__(self, top_config):

        self.SNR_set_gen_training = top_config.SNR_set_gen_training
        self.currently_trained_net_id = top_config.currently_trained_net_id

        # Training data info
        self.training_sample_num = 1999200
        self.training_minibatch_size = 1400
        self.training_folder = "./TrainingData"
        self.training_feature_file = format("./TrainingData/EstNoise_before_cnn%d.dat" % self.currently_trained_net_id)
        self.training_noise_label_file = "./TrainingData/RealNoise.dat"
        self.training_intf_label_file = "./TrainingData/RealIntf.dat"

        # Test data info
        self.test_sample_num = 105000
        self.test_minibatch_size = 3500
        self.test_folder = "./TestData"
        self.test_feature_file = format("./TestData/EstNoise_before_cnn%d.dat" % self.currently_trained_net_id)
        self.test_noise_label_file = "./TestData/RealNoise.dat"
        self.test_intf_label_file = "./TestData/RealIntf.dat"


class NetConfig:
    def __init__(self, top_config):

        self.feature_length = top_config.feature_length
        self.conv_layers_num = top_config.conv_layers_num
        self.dense_layers_num = top_config.dense_layers_num
        self.total_layers_num = self.conv_layers_num + self.dense_layers_num
        self.filter_sizes = top_config.filter_sizes
        self.feature_map_nums = top_config.feature_map_nums


