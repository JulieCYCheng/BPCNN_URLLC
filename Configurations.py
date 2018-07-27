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

    def parse_cmd_line(self, argv):
        if len(argv) == 1:
            return

        id = 1

        while id < len(argv):
            if argv[id] == '-Func':
                self.function = argv[id + 1]
                print('Function is set to %s' % self.function)
            elif argv[id] == '-IntfProb':
                self.intf_prob = float(argv[id + 1])
                print('Interference probability is set to %g' % self.intf_prob)
            else:
                print('>>> Command not recognized: %s' % argv[id])
                exit(0)
            id = id + 2
