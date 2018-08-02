import numpy as np
import tensorflow as tf


class ConvNet:
    def __init__(self, net_config, train_config, net_id):
        self.net_config = net_config
        self.train_config = train_config
        self.conv_filter_name = {}
        self.bias_name = {}
        self.conv_filter = {}
        self.bias = {}
        self.best_conv_filter = {}
        self.best_bias = {}
        self.assign_best_conv_filter = {}
        self.assign_best_bias = {}
        self.net_id = net_id
        self.res_noise_power_dict = {}
        self.res_noise_pdf_dict = {}
        self.trade_off = 1.0  # total_loss = noise_loss + (trade_off * intf_loss)

    def build_network(self, built_for_training=False):
        x_in = tf.placeholder(tf.float32, [None, self.net_config.feature_length], name='x_in')
        x_in_reshape = tf.reshape(x_in, (-1, self.net_config.feature_length, 1, 1), name='x_in_reshape')

        layer_output = {}

        for layer in range(self.net_config.conv_layers_num):
            self.conv_filter_name[layer] = format("conv_layer%d" % layer)
            self.bias_name[layer] = format("b%d" % layer)

            if layer == 0:
                layer_input = x_in_reshape
                in_channels = 1

            else:
                layer_input = layer_output[layer - 1]
                in_channels = self.net_config.feature_map_nums[layer - 1]
            out_channels = self.net_config.feature_map_nums[layer]

            if built_for_training:
                # Xavier initialization for training
                self.conv_filter[layer] = tf.get_variable(name=self.conv_filter_name[layer],
                                                          shape=[self.net_config.filter_sizes[layer], 1, in_channels,
                                                                 out_channels],
                                                          dtype=tf.float32,
                                                          initializer=tf.contrib.layers.xavier_initializer())
                self.bias[layer] = tf.get_variable(name=self.bias_name[layer], shape=[out_channels],
                                                   dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                self.best_conv_filter[layer] = tf.Variable(
                    tf.ones([self.net_config.filter_sizes[layer], 1, in_channels, out_channels], tf.float32),
                    dtype=tf.float32)
                self.best_bias[layer] = tf.Variable(tf.ones([out_channels], tf.float32), dtype=tf.float32)
                self.assign_best_conv_filter[layer] = self.best_conv_filter[layer].assign(self.conv_filter[layer])
                self.assign_best_bias[layer] = self.best_bias[layer].assign(self.bias[layer])
            else:
                # just build tensors for testing and their values will be loaded later.
                self.conv_filter[layer] = tf.Variable(
                    tf.random_normal([self.net_config.filter_sizes[layer], 1, in_channels, out_channels], 0, 1,
                                     tf.float32), dtype=tf.float32,
                    name=self.conv_filter_name[layer])
                self.bias[layer] = tf.Variable(tf.random_normal([out_channels], 0, 1, tf.float32), dtype=tf.float32,
                                               name=self.bias_name[layer])

            layer_output[layer] = tf.nn.relu(
                tf.nn.conv2d(layer_input, self.conv_filter[layer], [1, 1, 1, 1], 'SAME') + self.bias[layer])

        # Dense layer
        for layer in range(self.net_config.conv_layers_num, self.net_config.total_layers_num):
            layer_output[layer] = tf.layers.dense(inputs=layer_output[layer - 1], units=self.net_config.feature_length,
                                                  activation=tf.nn.relu)
            layer_output[layer] = tf.reshape(layer_output[layer], [-1, self.net_config.feature_length])

        # Multiple task
        y_out = tf.layers.dense(inputs=layer_output[self.net_config.total_layers_num - 1],
                                units=self.net_config.feature_length)
        y_out = tf.reshape(y_out, [-1, self.net_config.feature_length])

        i_out = tf.layers.dense(inputs=layer_output[self.net_config.total_layers_num - 1], units=1)
        i_out = tf.reshape(i_out, [-1, 1])

        print('CNN network built!')

        return x_in, y_out, i_out

    def restore_network_with_model_id(self, sess_in, restore_layers_num, model_id):
        # restore some layers
        save_dict = {}
        if restore_layers_num > 0:
            for layer in range(restore_layers_num):
                save_dict[self.conv_filter_name[layer]] = self.conv_filter[layer]
                save_dict[self.bias_name[layer]] = self.bias[layer]
            model_id_str = np.array2string(model_id, separator='_', formatter={'int': lambda d: "%d" % d})
            model_id_str = model_id_str[1:(len(model_id_str)-1)]
            model_folder = format("%snetid%d_model%s" % (self.net_config.model_folder, self.net_id, model_id_str))
            restore_model_name = format("%s/model.ckpt" % model_folder)
            saver_restore = tf.train.Saver(save_dict)
            saver_restore.restore(sess_in, restore_model_name)
            print("Restore the first %d layers.\n" % restore_layers_num)
