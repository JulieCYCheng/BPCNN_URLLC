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
            else:
                print('>>> Command not recognized: %s' % argv[ind])
                exit(0)
            ind = ind + 2


class TrainConfig:
    def __init__(self, top_config):

        self.SNR_set_gen_training = top_config.SNR_set_gen_training

        # Training data info
        self.training_sample_num = 1999200
        self.training_minibatch_size = 1400

        # Test data info
        self.test_sample_num = 105000
        self.test_minibatch_size = 3500
