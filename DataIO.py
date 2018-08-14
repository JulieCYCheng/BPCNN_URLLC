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

