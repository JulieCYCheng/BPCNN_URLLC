import tensorflow as tf
import numpy as np


class GetMatrixForBPNet:
    def __init__(self, H, nzero_r_c_rs):
        self.H = H
        self.R, self.N = np.shape(H)                # R = 144, N = 576
        self.H_col_weights = np.sum(H, axis=0)      # shape = (576, )
        self.H_row_weights = np.sum(H, axis=1)      # shape = (144, )
        self.nzero_r_c_rs = nzero_r_c_rs            # shape = (2, E), sorted by row
        self.E = np.size(self.nzero_r_c_rs[1, :])

        self.nzero_ver = self.nzero_r_c_rs[1, :] * self.N + self.nzero_r_c_rs[0, :]
        self.nzero_ver_sort = np.sort(self.nzero_ver)
        self.nzero_r_c_cs = np.append([np.mod(self.nzero_ver_sort, self.N)],
                                      [self.nzero_ver_sort // self.N], axis=0)      # shape = (2, E), sorted by column
        self.nzero_hor = self.nzero_r_c_cs[0, :] * self.N + self.nzero_r_c_cs[1, :]
        self.nzero_hor_sort = np.sort(self.nzero_hor)

    def get_matrix_VC(self):
        H_xin_to_var = np.zeros([self.E, self.N], dtype=np.float32)
        H_var_to_chk = np.zeros([self.E, self.E], dtype=np.float32)
        H_var_to_yout = np.zeros([self.N, self.E], dtype=np.float32)
        map_row_to_col = np.zeros([self.E, 1])

        for i in range(self.E):
            map_row_to_col[i] = np.where(self.nzero_ver == self.nzero_ver_sort[i])

        map_H_row_to_col = np.zeros([self.E, self.E], dtype=np.float32)

        for i in range(self.E):
            map_H_row_to_col[i, int(map_row_to_col[i])] = 1

        count = 0
        for i in range(self.N):
            temp = count + self.H_col_weights[i]
            H_var_to_chk[count:temp, count:temp] = 1
            H_var_to_yout[i, count:temp] = 1
            H_xin_to_var[count:temp, i] = 1
            for j in range(self.H_col_weights[i]):
                H_var_to_chk[count + j, count + j] = 0
            count = count + self.H_col_weights[i]
        print('Return VC matrix successfully!')

        return H_xin_to_var, np.matmul(H_var_to_chk, map_H_row_to_col), np.matmul(H_var_to_yout, map_H_row_to_col)
        # shape:    (E, N),     (E, E) = (E, E) x (E, E),                  (N, E) = (N, E) x (E, E)

    def get_matrix_CV(self):
        H_chk_to_var = np.zeros([self.E, self.E], dtype=np.float32)

        map_col_to_row = np.zeros([self.E, 1])

        for i in range(self.E):
            map_col_to_row[i] = np.where(self.nzero_hor == self.nzero_hor_sort[i])

        map_H_col_to_row = np.zeros([self.E, self.E], dtype=np.float32)

        for i in range(self.E):
            map_H_col_to_row[i, int(map_col_to_row[i])] = 1

        count = 0
        for i in range(self.R):
            temp = count + self.H_row_weights[i]
            H_chk_to_var[count:temp, count:temp] = 1
            for j in range(self.H_row_weights[i]):
                H_chk_to_var[count + j, count + j] = 0
            count = count + self.H_row_weights[i]
        print('Return CV matrix successfully!')

        return np.matmul(H_chk_to_var, map_H_col_to_row)    # shape: (E, E) = (E, E) x (E, E)


class BP_NetDecoder:
    def __init__(self, H, batch_size):
        _, self.N = np.shape(H)
        rr, cc = np.nonzero(H)
        nzero_r_c_rs = np.array([rr, cc])      # shape = (2, E)
        self.E = len(rr)

        gm1 = GetMatrixForBPNet(H[:, :], nzero_r_c_rs)

        self.H_chk_to_var = gm1.get_matrix_CV()
        self.H_xin_to_var, self.H_var_to_chk, self.H_var_to_yout = gm1.get_matrix_VC()

        self.batch_size = batch_size

        self.llr_placeholder = tf.placeholder(tf.float32, [batch_size, self.N])
        self.llr_into_bp_net, self.xe_0, self.xe_v2c_pre_iter_assign, self.start_next_iteration, self.dec_out = self.build_bp_network()
        self.llr_assign = self.llr_into_bp_net.assign(tf.transpose(self.llr_placeholder))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()  # open a session
        print('Open a tf session! (BP_Decoder.py/ BP_NetDecoder)')
        self.sess.run(init)

    # def __del__(self):
    #     self.sess.close()
    #     print('Close a tf session!')

    def atanh(self, x):
        x1 = tf.add(1.0, x)
        x2 = tf.subtract((1.0), x)
        x3 = tf.divide(x1, x2)
        x4 = tf.log(x3)
        return tf.divide(x4, (2.0))

    def one_BP_iteration(self, xe_v2c_pre_iter, H_chk_to_var, H_var_to_chk, xe0):
        xe_tanh = tf.tanh(tf.to_double(tf.truediv(xe_v2c_pre_iter, [2.0])))                 # shape: (E, B)
        xe_tanh = tf.to_float(xe_tanh)
        xe_tanh_temp = tf.sign(xe_tanh)
        xe_sum_log_img = tf.matmul(H_chk_to_var, tf.multiply(tf.truediv((1 - xe_tanh_temp), [2.0]), [3.1415926]))   # shape: (E, B)
        xe_sum_log_real = tf.matmul(H_chk_to_var, tf.log(1e-8 + tf.abs(xe_tanh)))           # shape: (E, B)
        xe_sum_log_complex = tf.complex(xe_sum_log_real, xe_sum_log_img)
        xe_product = tf.real(tf.exp(xe_sum_log_complex))
        xe_product_temp = tf.multiply(tf.sign(xe_product), -2e-7)
        xe_pd_modified = tf.add(xe_product, xe_product_temp)
        xe_v_sumc = tf.multiply(self.atanh(xe_pd_modified), [2.0])
        xe_c_sumv = tf.add(xe0, tf.matmul(H_var_to_chk, xe_v_sumc))                         # shape: (E, B)
        return xe_v_sumc, xe_c_sumv

    def build_bp_network(self):
        llr_into_BP_net = tf.Variable(np.ones([self.N, self.batch_size]), dtype=np.float32)     # shape: (N, B)
        xe0 = tf.matmul(self.H_xin_to_var, llr_into_BP_net, name='xe0')                         # shape: (E, B) = (E, N) x (N, B)
        xe_v2c_pre_iter = tf.Variable(np.ones([self.E, self.batch_size]), dtype=np.float32)
        xe_v2c_pre_iter_assign = xe_v2c_pre_iter.assign(xe0)

        # One iteration
        H_chk_to_var = tf.constant(self.H_chk_to_var, dtype=tf.float32)
        H_var_to_chk = tf.constant(self.H_var_to_chk, dtype=tf.float32)
        xe_v_sumc, xe_c_sumv = self.one_BP_iteration(xe_v2c_pre_iter, H_chk_to_var, H_var_to_chk, xe0)

        # Start the next iteration
        start_next_iteration = xe_v2c_pre_iter.assign(xe_v_sumc)

        bp_out_llr = tf.add(llr_into_BP_net, tf.matmul(self.H_var_to_yout, xe_v_sumc))          # shape: (N, B) = (N, B) + (N, E) x (E, B)
        dec_out = tf.transpose(tf.floordiv(1 - tf.to_int32(tf.sign(bp_out_llr)), 2))            # +1 to 0, -1 to 1

        return llr_into_BP_net, xe0, xe_v2c_pre_iter_assign, start_next_iteration, dec_out



