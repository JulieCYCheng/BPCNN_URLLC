import numpy as np
import tensorflow as tf


class NoiseIO:
    def __init__(self, top_config, read_from_file, noise_file, rng_seed=None):
        self.read_from_file = read_from_file
        self.blk_len = top_config.N
        self.rng_seed = rng_seed

        if read_from_file:
            self.fin_noise = open(noise_file, 'rb')
        else:
            self.rng = np.random.RandomState(rng_seed)
            self.awgn_noise = tf.placeholder(dtype=tf.float32, shape=[None, self.blk_len])
            self.sess = tf.Session()

    def generate_noise(self, batch_size):
        if self.read_from_file:
            noise = np.fromfile(self.fin_noise, np.float32, batch_size * self.blk_len)
            noise = np.reshape(noise, [batch_size, self.blk_len])
        else:
            noise_awgn = self.rng.randn(batch_size, self.blk_len)
            noise_awgn = noise_awgn.astype(np.float32)
            noise = self.sess.run(self.awgn_noise, feed_dict={self.awgn_noise: noise_awgn})

        return noise


class IntfIO:
    def __init__(self, top_config, read_from_file, intf_file, rng_seed=None):
        self.read_from_file = read_from_file
        self.rng_seed = rng_seed
        self.intf_prob = top_config.intf_prob

        if read_from_file:
            self.fin_noise = open(intf_file, 'rb')
        else:
            self.rng = np.random.RandomState(rng_seed)
            self.intf_labels_ph = tf.placeholder(dtype=tf.int8, shape=[None, 1])
            self.sess = tf.Session()

    def generate_intf_labels(self, batch_size):
        if self.read_from_file:
            intf_labels = np.fromfile(self.fin_noise, np.float32, batch_size)
            intf_labels = np.reshape(intf_labels, [batch_size, 1])
        else:
            intf_labels_rand = np.random.choice([0, 1], size=[batch_size, 1], p=[1 - self.intf_prob, self.intf_prob])
            intf_labels = self.sess.run(self.intf_labels_ph, feed_dict={self.intf_labels_ph: intf_labels_rand})

        return intf_labels


class TrainingDataIO:
    def __init__(self, feature_filename, label_filename, intf_filename, total_training_samples, feature_length,
                 noise_label_length):
        print("Construct the data IO class for training!\n")
        self.fin_label = open(label_filename, "rb")
        self.fin_feature = open(feature_filename, "rb")
        self.fin_intf = open(intf_filename, "rb")
        self.total_training_samples = total_training_samples
        self.feature_length = feature_length
        self.label_length = noise_label_length

    def __del__(self):
        print(">>> Delete the training data IO class! (DataIO.py)\n")
        self.fin_feature.close()
        self.fin_label.close()
        self.fin_intf.close()

    def load_next_minibatch(self, minibatch_size, factor_of_start_pos=1):
        remain_samples = minibatch_size
        sample_id = np.random.randint(self.total_training_samples - minibatch_size)

        features = np.zeros(0)
        noise_labels = np.zeros(0)
        intf_labels = np.zeros(0)
        if minibatch_size > self.total_training_samples:
            print(">>> Mini batch size should not be larger than total sample size!\n")
        self.fin_feature.seek((self.feature_length * 4) * (sample_id // factor_of_start_pos * factor_of_start_pos),
                              0)  # float32 = 4 bytes = 32 bits
        self.fin_label.seek((self.label_length * 4) * (sample_id // factor_of_start_pos * factor_of_start_pos), 0)
        self.fin_intf.seek((1 * 4) * (sample_id // factor_of_start_pos * factor_of_start_pos), 0)

        while 1:
            new_feature = np.fromfile(self.fin_feature, np.float32, self.feature_length * remain_samples)
            new_noise_label = np.fromfile(self.fin_label, np.float32, self.label_length * remain_samples)
            new_intf_label = np.fromfile(self.fin_intf, np.float32, 1 * remain_samples)

            features = np.concatenate((features, new_feature))
            noise_labels = np.concatenate((noise_labels, new_noise_label))
            intf_labels = np.concatenate((intf_labels, new_intf_label))

            remain_samples -= len(new_feature) // self.feature_length

            if remain_samples == 0:
                break

            self.fin_feature.seek(0, 0)
            self.fin_label.seek(0, 0)
            self.fin_intf.seek(0, 0)

        features = features.reshape((minibatch_size, self.feature_length))
        noise_labels = noise_labels.reshape((minibatch_size, self.label_length))
        intf_labels = intf_labels.reshape((minibatch_size, 1))

        return features, noise_labels, intf_labels


class TestDataIO:
    def __init__(self, feature_filename, label_filename, intf_filename, total_test_samples, feature_length,
                 noise_label_length):
        print("Construct the data IO class for test!\n")
        self.fin_label = open(label_filename, "rb")
        self.fin_feature = open(feature_filename, "rb")
        self.fin_intf = open(intf_filename, "rb")
        self.total_test_samples = total_test_samples
        self.feature_length = feature_length
        self.label_length = noise_label_length
        # self.all_features = np.zeros(0)
        # self.all_labels = np.zeros(0)
        # self.data_position = 0

    def __del__(self):
        print(">>> Delete the training data IO class! (DataIO.py)\n")
        self.fin_feature.close()
        self.fin_label.close()
        self.fin_intf.close()
