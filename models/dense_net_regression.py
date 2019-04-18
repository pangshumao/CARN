import tensorflow as tf
import numpy as np
# import pylab as plt
import matplotlib.pyplot as plt
from .lossFun import mae_lossFun, mse_lossFun, pearson_lossFun, logHC_lossFun, l1_lossFun, adaptive_lossFun, lspmr_lossFun
from functools import reduce
import os
import time
import shutil
from datetime import timedelta
import math
import numpy as np
import tensorflow as tf
plt.switch_backend('agg')

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class DenseNetRegression:
    def __init__(self, data_provider, growth_rate, depth,
                 total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_save_logs, should_save_model, save_path, logs_path,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 carn_mode=False,
                 cnn_mode=False,
                 lspmr_mode=False,
                 lr=0.1,
                 knn=20,
                 alscmrWeight=0.0,
                 lspmrWeight=0.0,
                 **kwargs):
        """
        Class to implement networks from this paper
        https://arxiv.org/pdf/1611.05552.pdf

        Args:
            data_provider: Class, that have all required data sets
            growth_rate: `int`, variable from paper
            depth: `int`, variable from paper
            total_blocks: `int`, paper value == 3
            keep_prob: `float`, keep probability for dropout. If keep_prob = 1
                dropout will be disables
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
            nesterov_momentum: `float`, momentum for Nesterov optimizer
            model_type: `str`, 'DenseNet' or 'DenseNet-BC'. Should model use
                bottle neck connections or not.
            dataset: `str`, dataset name
            should_save_logs: `bool`, should logs be saved or not
            should_save_model: `bool`, should model be saved or not
            renew_logs: `bool`, remove previous logs for current model
            reduction: `float`, reduction Theta at transition layer for
                DenseNets with bottleneck layers. See paragraph 'Compression'
                https://arxiv.org/pdf/1608.06993v3.pdf#4
            bc_mode: `bool`, should we use bottleneck layers and features
                reduction or not.
        """
        self.display_step = 50
        self.layer = {}
        self.carn_mode = carn_mode
        self.cnn_mode = cnn_mode
        self.lspmr_mode = lspmr_mode
        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        self.depth = depth
        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        # self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.layers_per_block = (depth - (total_blocks + 2)) // total_blocks
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        self._save_path = save_path
        self._logs_path = logs_path
        self.lr = lr
        self.knn = knn
        self.alscmrWeight = alscmrWeight
        self.lspmrWeight = lspmrWeight
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])
        # self.summary_merged = tf.summary.merge(self.summary_list)
        self.summary_merged = tf.summary.merge_all()
        self.train_summary_writer = tf.summary.FileWriter(os.sep.join([self.logs_path, 'train']), self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(os.sep.join([self.logs_path, 'test']))
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
        #     logswriter = tf.train.SummaryWriter
        else:
            self.sess.run(tf.global_variables_initializer())
        #     logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        # self.summary_merged = tf.summary.merge(self.summary_list)
        # # self.summary_merged = tf.summary.merge_all()
        # # self.summary_writer = logswriter(self.logs_path)
        # self.train_summary_writer = tf.summary.FileWriter(self.logs_path + '/train', self.sess.graph)
        # self.test_summary_writer = tf.summary.FileWriter(self.logs_path + '/test')

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
            save_path = os.path.join(save_path, '%s' % self.model_identifier)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
            logs_path = os.path.join(logs_path, '%s' % self.model_identifier)
            # if self.renew_logs:
            #     shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            # if self.renew_logs:
            #     shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return "{}_depth_{}_dataset_{}_lr_{}_knn_{}_alscmrWeight_{}_lspmrWeight_{}".format(
            self.model_type, self.depth, self.dataset_name, self.lr, self.knn , self.alscmrWeight, self.lspmrWeight)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        # try:
        #     self.saver.restore(self.sess, self.save_path + 'something')
        # except Exception as e:
        #     raise IOError("Failed to to load model "
        #                   "from save path: %s" % self.save_path)

        full_path = self.save_path.split(os.sep)
        sub_path = os.sep.join(full_path[:-1])
        if os.path.exists(sub_path):
            try:
                self.saver.restore(self.sess, self.save_path)
                print("Successfully load model from save path: %s" % self.save_path)
            except Exception as e:
                print("Failed to to load model "
                              "from save path: %s" % self.save_path)
                print('training from scratch......')


    def log_loss(self, mae_loss, epoch, prefix,
                          should_print=True, lspmr_loss=None):
        if should_print:
            if lspmr_loss is None:
                print("mean mae_loss: %f" % (mae_loss))
                # summary = tf.Summary(value=[tf.Summary.Value(tag='mae_loss_%s' % prefix, simple_value=float(mae_loss))])
                # self.summary_writer.add_summary(summary, epoch)
            else:
                print("mean mae_loss: %f, lspmr_loss: %f" % (mae_loss, lspmr_loss))
                # summary = tf.Summary(value=[tf.Summary.Value(tag='mae_loss_%s' % prefix, simple_value=float(mae_loss))])
                # self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.recon_labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='recon_labels')
        self.adjacentMatrix = tf.placeholder(
            tf.float32,
            shape=[None, None],
            name='adjacentMatrix')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])
        self.disc_target_real = tf.placeholder(tf.int32, shape=[None])
        self.disc_target_fake = tf.placeholder(tf.int32, shape=[None])
        self.gen_target = tf.placeholder(tf.int32, shape=[None])
        self.k_d = tf.Variable(0.01, trainable=False, name='k_d')
        self.k_g = tf.Variable(0., trainable=False, name='k_g')

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')

            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not self.bc_mode:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output

    def gating(self, _input, composite=None):
        shape = _input.get_shape().as_list()
        if composite is None:
            with tf.variable_scope("gate"):
                conv = self.conv2d(_input, shape[-1], 3)
                # drop = self.dropout(conv)
                bn = self.batch_norm(conv)
                gate = tf.nn.sigmoid(bn)
                output = gate * _input
                return output
        else:
            with tf.variable_scope('gate'):
                # conv = self.conv2d(composite, shape[-1], 3)
                # bn = self.batch_norm(conv)
                # gate = tf.nn.sigmoid(bn)
                # output = gate * _input
                # return output

                conv = self.conv2d(composite, shape[-1], 3)
                bn = self.batch_norm(conv)
                gate = tf.nn.tanh(bn)
                output = gate * _input + _input
                # output = tf.concat(axis=3, values=(gate * _input, _input))
                return output

    def swish(self, x):
        return x * tf.nn.sigmoid(x)

    def add_block(self, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, [2, 2])
        return output

    def transition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        if self.carn_mode is True or self.cnn_mode is True:
            output = _input
        else:
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # output = tf.nn.elu(_input)

        # global average pooling for denseNet
        last_pool_kernel = output.get_shape().as_list()[1:3]

        if self.carn_mode is True or self.cnn_mode is True:
            # average pooling for gcnn or cnn
            # last_pool_kernel[0] /= 2
            # last_pool_kernel[1] /= 2
            last_pool_kernel[0] = 4
            last_pool_kernel[1] = 2

        output = self.avg_pool(output, last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1] * output.get_shape()[-2] * output.get_shape()[-3])

        # shape = output.get_shape().as_list()
        # features_total = reduce(lambda x, y: x * y, shape[1:])

        print('features_total: ', features_total)
        feature = tf.reshape(output, [-1, features_total])

        # W1 = self.weight_variable_xavier(
        #     [features_total, self.n_classes], name='W1')
        # bias1 = self.bias_variable([self.n_classes], name='bias1')
        # output = tf.matmul(feature, W1) + bias1
        output = self.fc(feature, [features_total, self.n_classes], 'fc1', activation=None)
        # fc1 = self.dropout(fc1)
        # output = self.fc(fc1, [256, self.n_classes], 'fc2', activation=None)

        return output, feature

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME', initialize='msra'):
        in_features = int(_input.get_shape()[-1])
        if initialize == 'msra':
            with tf.name_scope('kernel'):
                kernel = self.weight_variable_msra(
                    [kernel_size, kernel_size, in_features, out_features],
                    name='kernel')
                self.variable_summaries(kernel)
        elif initialize == 'xavier':
            with tf.name_scope('kernel'):
                kernel = self.weight_variable_xavier(
                    [kernel_size, kernel_size, in_features, out_features],
                    name='kernel')
                self.variable_summaries(kernel)
        with tf.name_scope('bias'):
            bias = self.bias_variable([out_features], name='bias')
            self.variable_summaries(bias)
        output = tf.nn.bias_add(tf.nn.conv2d(_input, kernel, strides, padding), bias)
        tf.summary.histogram('preactivation', output)
        return output

    def conv2d_withName(self, _input, out_features, name, kernel_size=3,
               strides=[1, 1, 1, 1], padding='SAME', activation='relu'):
        with tf.variable_scope(name):
            conv = self.conv2d(_input, out_features, kernel_size, strides, padding)
            if activation == 'relu':
                bn = self.batch_norm(conv)
                tf.summary.histogram('bn', bn)
                output = tf.nn.relu(bn)
                tf.summary.histogram('relu', output)
                return output
            elif activation == 'linear':
                output = conv
                return output

    def avg_pool(self, _input, kernel_size):
        ksize = [1, kernel_size[0], kernel_size[1], 1]
        strides = [1, kernel_size[0], kernel_size[1], 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding, name='average_pool')
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None, decay=0.99, epsilon=1e-7)
        # output = tf.contrib.layers.batch_norm(
        #     _input, scale=True, is_training=self.is_training,
        #     updates_collections=tf.GraphKeys.UPDATE_OPS, decay=0.99, epsilon=1e-7, renorm=True)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())
# from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer
    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True))

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        # initial = tf.truncated_normal(shape, .0, .01)
        return tf.get_variable(name, initializer=initial)

    def fc(self, x, shape, name, activation=None):
        with tf.variable_scope(name):
            W = self.weight_variable_xavier(
                [shape[0], shape[1]], name='weight')
            bias = self.bias_variable([shape[1]], name='bias')
        x = tf.matmul(x, W) + bias
        if activation is None:
            return x
        elif activation == 'relu':
            # x = self.batch_norm(x)
            return tf.nn.relu(x)
        elif activation == 'sigmoid':
            # x = self.batch_norm(x)
            return tf.nn.sigmoid(x)
        elif activation == 'tanh':
            # x = self.batch_norm(x)
            return tf.nn.tanh(x)
        elif activation == 'elu':
            return tf.nn.elu(x)
        else:
            return x

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse):
            # [30, 128]
            x = self.fc(x, [x.get_shape().as_list()[1], 128], name='fc1', activation='elu')
            # [128, 256]
            x = self.fc(x, [128, 10], name='fc2', activation='elu')
            # [256, 128]
            x = self.fc(x, [10, 128], name='fc3', activation='elu')
            # [128, 2]
            x = self.fc(x, [128, 30], name='fc4', activation='linear')
        return x

    def generator(self, reuse=False):
        if self.carn_mode is False:
            if self.cnn_mode is False:
                ############################################denseNet###################################################
                growth_rate = self.growth_rate
                layers_per_block = self.layers_per_block
                with tf.variable_scope('Generator', reuse=reuse):
                    with tf.variable_scope("Initial_convolution_1"):
                        output = self.conv2d(
                            self.images,
                            out_features=self.first_output_features,
                            kernel_size=7,
                            strides=[1, 2, 2, 1])
                        output = self.batch_norm(output)
                        output = tf.nn.relu(output)
                    with tf.variable_scope("Initial_convolution_2"):
                        output = self.conv2d(
                            output,
                            out_features=self.first_output_features * 2,
                            kernel_size=7,
                            strides=[1, 2, 2, 1])
                    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

                    # add N required blocks
                    for block in range(self.total_blocks):
                        with tf.variable_scope("Block_%d" % block):
                            output = self.add_block(output, growth_rate, layers_per_block)
                            self.layer["Block_%d" % block] = output
                        # last block exist without transition layer
                        if block != self.total_blocks - 1:
                            with tf.variable_scope("Transition_after_block_%d" % block):
                                output = self.transition_layer(output)
                                self.layer["Transition_after_block_%d" % block] = output

                    with tf.variable_scope("Transition_to_regression"):
                        self.prediction, self.feature = self.transition_layer_to_classes(output)
                return self.prediction
            else:
                ################################################ CNN ###################################################
                with tf.variable_scope('Generator', reuse=reuse):
                    root_channels = 8
                    # 512,256,1 -> 256,128,root_channels
                    with tf.variable_scope("Initial_convolution_1"):
                        output = self.conv2d(
                            self.images,
                            out_features=root_channels,
                            kernel_size=7,
                            strides=[1, 2, 2, 1])
                        output = self.batch_norm(output)
                        self.layer['Initial_convolution_1'] = tf.nn.relu(output)
                    # 256,128,root_channels -> 256,128,root_channels * 2
                    self.layer['cnn_1'] = self.conv2d_withName(self.layer['Initial_convolution_1'], root_channels * 2, 'cnn_1')
                    # 256,128,root_channels * 2 -> 128,64,root_channels * 2
                    output = tf.nn.max_pool(self.layer['cnn_1'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='SAME', name='max_1')

                    # 128,64,root_channels * 2 -> 128,64,root_channels * 4
                    self.layer['cnn_2'] = self.conv2d_withName(output, root_channels * 4, 'cnn_2')
                    # 128,64,root_channels * 4 -> 64,32,root_channels * 4
                    output = tf.nn.max_pool(self.layer['cnn_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='SAME', name='max_2')

                    # 64,32,root_channels * 4 -> 64,32,root_channels * 8
                    self.layer['cnn_3'] = self.conv2d_withName(output, root_channels * 8, 'cnn_3')
                    # 64,32,root_channels * 8 -> 32,16,root_channels * 8
                    output = tf.nn.max_pool(self.layer['cnn_3'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='SAME', name='max_3')

                    # 32,16,root_channels * 8 -> 32,16,root_channels * 16
                    self.layer['cnn_4'] = self.conv2d_withName(output, root_channels * 16, 'cnn_4')
                    # 32,16,root_channels * 16 -> 16,8,root_channels * 16
                    output = tf.nn.max_pool(self.layer['cnn_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='SAME', name='max_4')

                    # 16,8,root_channels * 16 -> 16,8,root_channels * 32
                    self.layer['cnn_5'] = self.conv2d_withName(output, root_channels * 32, 'cnn_5')
                    # 16,8,root_channels * 32 -> 8,4,root_channels * 32
                    output = tf.nn.max_pool(self.layer['cnn_5'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='SAME', name='max_5')

                    self.layer['cnn_6'] = self.conv2d_withName(output, root_channels * 64, 'cnn_6')
                    output = self.conv2d_withName(self.layer['cnn_6'], 256, 'cnn_7', kernel_size=1, strides=[1,1,1,1], activation='linear')
                    output = self.batch_norm(output)
                    with tf.variable_scope("Transition_to_regression"):
                        self.prediction, self.feature = self.transition_layer_to_classes(output)
                return self.prediction
        else:
            ################################################ CARN #################################################
            with tf.variable_scope('Generator', reuse=reuse):
                root_channels = 8
                depth = self.depth
                # 512,256,1 -> 256,128,root_channels
                with tf.variable_scope("Initial_convolution_1"):
                    output = self.conv2d(
                        self.images,
                        out_features=root_channels,
                        kernel_size=7,
                        strides=[1, 2, 2, 1])
                    output = self.batch_norm(output)
                    self.layer['Initial_convolution_1'] = tf.nn.relu(output)
                # 256,128,root_channels -> 256,128,root_channels * 2
                self.layer['gcnn_1'], self.layer['gate_1'], self.layer['selected_fea_1'] = self.gated_conv(self.layer['Initial_convolution_1'], root_channels * 2, 'gcnn_1')
                # 256,128,root_channels * 2 -> 128,64,root_channels * 2
                output = tf.nn.max_pool(self.layer['gcnn_1'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME', name='max_1')

                # 128,64,root_channels * 2 -> 128,64,root_channels * 4
                self.layer['gcnn_2'], self.layer['gate_2'], self.layer['selected_fea_2'] = self.gated_conv(output, root_channels * 4, 'gcnn_2')
                # 128,64,root_channels * 4 -> 64,32,root_channels * 4
                output = tf.nn.max_pool(self.layer['gcnn_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_2')

                # 64,32,root_channels * 4 -> 64,32,root_channels * 8
                self.layer['gcnn_3'], self.layer['gate_3'], self.layer['selected_fea_3'] = self.gated_conv(output, root_channels * 8, 'gcnn_3')
                # 64,32,root_channels * 8 -> 32,16,root_channels * 8
                output = tf.nn.max_pool(self.layer['gcnn_3'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_3')

                # 32,16,root_channels * 8 -> 32,16,root_channels * 16
                self.layer['gcnn_4'], self.layer['gate_4'], self.layer['selected_fea_4'] = self.gated_conv(output, root_channels * 16, 'gcnn_4')
                # self.layer['gcnn_4'] = self.dropout(self.layer['gcnn_4'])

                if depth == 4:
                    # 4 AU
                    output = self.conv2d_withName(self.layer['gcnn_4'], 256, 'last_conv', kernel_size=1,
                                                  strides=[1, 1, 1, 1],
                                                  activation='linear')
                elif depth == 5:
                    # 5 AU
                    # 32,16,root_channels * 16 -> 16,8,root_channels * 16
                    output = tf.nn.max_pool(self.layer['gcnn_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_4')

                    self.layer['gcnn_5'], self.layer['gate_5'], self.layer['selected_fea_5'] = self.gated_conv(output, root_channels * 32, 'gcnn_5')
                    output = self.conv2d_withName(self.layer['gcnn_5'], 256, 'last_conv', kernel_size=1,
                                                  strides=[1, 1, 1, 1],
                                                  activation='linear')
                elif depth == 7:
                    # 7 AU
                    output = tf.nn.max_pool(self.layer['gcnn_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='SAME', name='max_4')

                    self.layer['gcnn_5'], self.layer['gate_5'], self.layer['selected_fea_5'] = self.gated_conv(output,
                                                                                                               root_channels * 32,
                                                                                                               'gcnn_5')
                    output = tf.nn.max_pool(self.layer['gcnn_5'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_5')

                    self.layer['gcnn_6'], self.layer['gate_6'], self.layer['selected_fea_6'] = self.gated_conv(output, root_channels * 64, 'gcnn_6')
                    output = tf.nn.max_pool(self.layer['gcnn_6'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_6')

                    self.layer['gcnn_7'], self.layer['gate_7'], self.layer['selected_fea_7'] = self.gated_conv(output, root_channels * 128, 'gcnn_7')

                    output = self.conv2d_withName(self.layer['gcnn_7'], 256, 'last_conv', kernel_size=1, strides=[1, 1, 1, 1],
                                                  activation='linear')
                elif depth == 6:
                    # 6 AU
                    # self.layer['gcnn_5'] = self.dropout(self.layer['gcnn_5'])
                    # 16,8,root_channels * 32 -> 8,4,root_channels * 32
                    # 32,16,root_channels * 16 -> 16,8,root_channels * 16
                    output = tf.nn.max_pool(self.layer['gcnn_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='SAME', name='max_4')

                    # 16,8,root_channels * 16 -> 16,8,root_channels * 32
                    self.layer['gcnn_5'], self.layer['gate_5'], self.layer['selected_fea_5'] = self.gated_conv(output,
                                                                                                               root_channels * 32,
                                                                                                               'gcnn_5')
                    output = tf.nn.max_pool(self.layer['gcnn_5'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                            padding='SAME', name='max_5')

                    self.layer['gcnn_6'], self.layer['gate_6'], self.layer['selected_fea_6'] = self.gated_conv(output,
                                                                                                               root_channels * 64,
                                                                                                               'gcnn_6')

                    output = self.conv2d_withName(self.layer['gcnn_6'], 256, 'last_conv', kernel_size=1,
                                                  strides=[1, 1, 1, 1],
                                                  activation='linear')
                output = self.batch_norm(output)
                with tf.variable_scope("Transition_to_regression"):
                    self.prediction, self.feature = self.transition_layer_to_classes(output)
            return self.prediction


    def gated_conv(self, _input, out_channels, name):
        with tf.variable_scope(name):
            in_channels = _input.get_shape().as_list()[-1]
            linear_out_c = out_channels - in_channels
            nonlinear_out_c = in_channels
            linear_feature = self.conv2d(_input, linear_out_c, kernel_size=3, initialize='xavier')
            # linear_feature = self.conv2d_withName(_input, linear_out_c, kernel_size=3, name='control_feature', activation='relu')
            # linear_feature = self.conv2d_withName(_input, linear_out_c, kernel_size=3, name='control_feature',
            #                                       activation='linear')
            with tf.variable_scope('gate'):
                conv = self.conv2d(linear_feature, nonlinear_out_c, 3, initialize='xavier')
                bn = self.batch_norm(conv)
                # bn = conv
                gate = tf.nn.tanh(bn)
                tf.summary.histogram('gate', gate)
                selected_feature = gate * _input + _input
                tf.summary.histogram('nonlinear_feature', selected_feature)
                output = tf.concat(axis=3, values=(selected_feature, tf.nn.relu(self.batch_norm(linear_feature))))
                # output = tf.concat(axis=3, values=(nonlinear_feature, linear_feature))
                with tf.variable_scope('out'):
                    bn = self.batch_norm(output)
                    tf.summary.histogram('out', bn)
                    shape = bn.get_shape().as_list()
                    tf.summary.image('out', tf.reshape(bn[:, :, :, 0], [-1, shape[1], shape[2], 1]), 10)
                    return bn, gate, selected_feature

    def _build_graph(self):
        self.generator(reuse=False)
        # Losses
        self.mae_loss = mae_lossFun(self.prediction, self.labels)
        self.alscmr_loss = mae_lossFun(self.prediction, self.recon_labels)

        self.l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # self.l2_loss = tf.add_n(
        #     [tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name])

        # self.l2_loss = tf.add_n(
        #     [tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'kernel' in var.name])
        # for var in tf.trainable_variables():
        #     if 'weight' in var.name:
        #         group_lasso = tf.reduce_sum(tf.norm(var, ord=2, axis=1))
        #         self.l2_loss = tf.add(self.l2_loss, 0.01 * group_lasso)

        self.summary_list = []
        # self.summary_list.append(tf.summary.scalar('group_lasso', group_lasso))
        self.summary_list.append(tf.summary.scalar('mae_loss', self.mae_loss))
        self.summary_list.append(tf.summary.scalar('l2_loss', self.l2_loss * self.weight_decay))
        if self.alscmrWeight >= 100.0:
            print('Using adaptive local shape-constrained manifold regularization (ALSCMR)................................................')
            self.total_loss = adaptive_lossFun(self.prediction, self.labels, self.recon_labels,
                                               self.data_provider.sigma, self.alscmrWeight - 100.0) + self.l2_loss * self.weight_decay
        else:
            print('Using local shape-constrained manifold regularization (LSCMR).....................................................')
            self.total_loss = self.mae_loss + self.l2_loss * self.weight_decay + self.alscmr_loss * self.alscmrWeight
            self.summary_list.append(tf.summary.scalar('alscmr_loss', self.alscmr_loss * self.laeWeight))
        if self.lspmr_mode is True:
            self.lspmr_loss = lspmr_lossFun(self.feature, self.adjacentMatrix)
            self.total_loss += self.lspmr_loss * self.lspmrWeight
            self.summary_list.append(tf.summary.scalar('lspmr_loss', self.lspmr_loss * self.lspmrWeight))
        self.summary_list.append(tf.summary.scalar('total_loss', self.total_loss))

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)

        self.train_step = optimizer.minimize(self.total_loss)

    def visualization(self, name, data, batch=2, channels=10):
        logs_path = self.logs_path.split(os.sep)
        pre_path = os.sep.join(logs_path[:-2])
        if not os.path.exists(os.sep.join([pre_path, 'visualization'])):
            os.mkdir(os.sep.join([pre_path, 'visualization']))
        vis_path = os.sep.join([pre_path, 'visualization', self.model_identifier])
        if not os.path.exists(vis_path):
            os.mkdir(vis_path)

        if data.shape[0] < batch:
            batch = data.shape[0]
        layerImages = self.sess.run(self.layer[name], feed_dict={self.images: data[:batch], self.is_training: False})
        layerImages = self.visual_scale(layerImages)
        if layerImages.shape[-1] < channels:
            channels = layerImages.shape[-1]
        rows = batch
        cols = channels
        height = layerImages.shape[1]
        width = layerImages.shape[2]
        gap = 10
        totalImages = np.zeros((rows * height + (rows - 1) * gap, cols * width + (cols - 1) * gap))
        for i in range(rows):
            for j in range(cols):
                totalImages[i * (height + gap): i * (height + gap) + height,
                j * (width + gap): j * (width + gap) + width] = layerImages[i, :, :, j]
        plt.figure()
        plt.imshow(totalImages, cmap='gray')
        plt.axis('off')
        plt.title(name)
        # plt.show()
        plt.savefig(os.sep.join([vis_path, name]) + ".png")
        plt.close()

    def visual_scale(self, x):
        eps = 1e-7
        num, height, width, channels = x.shape
        if height * width == 1:
            return x * 255
        else:
            image_out = np.zeros(x.shape)
            for i in range(num):
                image = x[i]
                for j in range(channels):
                    image_min = x[i, :, :, j].min()
                    image_max = x[i, :, :, j].max()
                    image_out[i, :, :, j] = (x[i,:,:,j] - image_min) / (image_max - image_min + eps) * 255
        return image_out

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        # mean
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # std
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # record std, max, min of var
        tf.summary.scalar('std', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # histogram
        tf.summary.histogram('histogram', var)

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        # learning_rate = train_params['initial_learning_rate']
        learning_rate = self.lr
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()
        total_train_mae_loss = []
        total_train_lspmr_loss = []
        total_val_mae_loss = []
        total_val_lspmr_loss = []
        full_path = self.logs_path
        sub_path_pre = os.sep.join(full_path.split(os.sep)[:-2])
        sub_path_last = full_path.split(os.sep)[-1]
        if not os.path.exists(os.sep.join([sub_path_pre, 'loss'])):
            os.mkdir(os.sep.join([sub_path_pre, 'loss']))
            os.mkdir(os.sep.join([sub_path_pre, 'loss', sub_path_last]))
        else:
            if not os.path.exists(os.sep.join([sub_path_pre, 'loss', sub_path_last])):
                os.mkdir(os.sep.join([sub_path_pre, 'loss', sub_path_last]))
            elif os.path.exists(os.sep.join([sub_path_pre, 'loss', sub_path_last, 'loss.npz'])):
                lossData = np.load(os.sep.join([sub_path_pre, 'loss', sub_path_last, 'loss.npz']))
                total_train_mae_loss = lossData['train_mae_loss'].tolist()
                total_val_mae_loss = lossData['val_mae_loss'].tolist()
                if self.lspmr_mode:
                    total_train_lspmr_loss = lossData['train_lspmr_loss'].tolist()
                    total_val_lspmr_loss = lossData['val_lspmr_loss'].tolist()

        for epoch in range(1, n_epochs + 1):
            displayStep = self.display_step
            if epoch % displayStep == 0:
                print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2 or epoch == (reduce_lr_epoch_1+reduce_lr_epoch_2) or epoch == (2*reduce_lr_epoch_1+reduce_lr_epoch_2):
            # if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 10
                print("Decrease learning rate, new lr = %f" % learning_rate)
            if epoch % displayStep == 0:
                print("Training...")
            if self.lspmr_mode is True:
                train_mae_loss, train_lspmr_loss = self.train_one_epoch(
                    self.data_provider.train, batch_size, learning_rate)
                total_train_mae_loss.append(train_mae_loss)
                total_train_lspmr_loss.append(train_lspmr_loss)
            else:
                train_mae_loss = self.train_one_epoch(
                    self.data_provider.train, batch_size, learning_rate)
                total_train_mae_loss.append(train_mae_loss)
            if self.should_save_logs:
                if epoch % displayStep == 0:
                    batch = self.data_provider.train.next_batch(batch_size)
                    images, labels, recon_labels, adjacentMatrix = batch
                    feed_dict = {
                        self.images: images,
                        self.labels: labels,
                        self.recon_labels: recon_labels,
                        self.adjacentMatrix: adjacentMatrix,
                        self.learning_rate: learning_rate,
                        self.is_training: True,
                    }
                    summary = self.sess.run(self.summary_merged, feed_dict=feed_dict)
                    self.train_summary_writer.add_summary(summary, epoch-1)
                    self.train_summary_writer.flush()
                    # self.train_summary_writer.close()
                    if self.lspmr_mode:
                        self.log_loss(train_mae_loss, epoch, prefix='train', lspmr_loss=train_lspmr_loss)
                    else:
                        self.log_loss(train_mae_loss, epoch, prefix='train')
                    if self.carn_mode is False:
                        if self.cnn_mode is False:
                            for block in range(0, self.total_blocks):
                                name = 'Block_%d' % block
                                self.visualization(name, self.data_provider.visualization.images, batch=4, channels=10)
                            for block in range(0, self.total_blocks-1):
                                name = 'Transition_after_block_%d' % block
                                self.visualization(name, self.data_provider.visualization.images, batch=4, channels=10)
                        else:
                            name = 'Initial_convolution_1'
                            self.visualization(name, self.data_provider.visualization.images, batch=4, channels=10)
                            for i in range(1, 7):
                                name = 'cnn_%d' % i
                                self.visualization(name, self.data_provider.visualization.images, batch=4, channels=10)
                    else:
                        name = 'Initial_convolution_1'
                        self.visualization(name, self.data_provider.visualization.images, batch=4, channels=10)
                        for i in range(1,self.depth + 1):
                            name = 'gcnn_%d' % i
                            self.visualization(name, self.data_provider.visualization.images, batch=4, channels=10)

            if train_params.get('validation_set', False):
                if epoch % displayStep == 0:
                    print("Validation...")
                    batch = self.data_provider.test.next_batch(batch_size=batch_size)
                    images, labels, recon_labels, adjacentMatrix = batch
                    feed_dict = {
                        self.images: images,
                        self.labels: labels,
                        self.recon_labels: recon_labels,
                        self.adjacentMatrix: adjacentMatrix,
                        self.learning_rate: learning_rate,
                        self.is_training: False,
                    }
                    summary = self.sess.run(self.summary_merged, feed_dict=feed_dict)
                    self.test_summary_writer.add_summary(summary, epoch - 1)
                    self.test_summary_writer.flush()
                if self.lspmr_mode is True:
                    val_mae_loss, val_lspmr_loss = self.test(
                        self.data_provider.validation, batch_size)
                    if self.should_save_logs:
                        total_val_mae_loss.append(val_mae_loss)
                        total_val_lspmr_loss.append(val_lspmr_loss)
                        if epoch % displayStep == 0:
                            self.log_loss(val_mae_loss, epoch, prefix='valid', lspmr_loss=val_lspmr_loss)
                else:
                    val_mae_loss = self.test(
                        self.data_provider.validation, batch_size)
                    if self.should_save_logs:
                        total_val_mae_loss.append(val_mae_loss)
                        if epoch % displayStep == 0:
                            self.log_loss(val_mae_loss, epoch, prefix='valid')


            if epoch % displayStep == 0:
                np.savez(os.sep.join([sub_path_pre, 'loss', sub_path_last, 'loss.npz']),
                         train_mae_loss=total_train_mae_loss, val_mae_loss=total_val_mae_loss,
                         train_lspmr_loss=total_train_lspmr_loss, val_lspmr_loss=total_val_lspmr_loss)
                x = np.arange(len(total_train_mae_loss))
                indNum = 40
                plt.figure()
                p1, = plt.plot(x[indNum:], total_train_mae_loss[indNum:])
                # p2, = plt.plot(x, total_val_mae_loss)
                plt.xlabel('epoch')
                plt.ylabel('MAE')
                plt.legend(handles=[p1], labels=['train'], loc='best')
                # plt.show()
                plt.savefig(os.sep.join([sub_path_pre, 'loss', sub_path_last, 'mae_loss.png']))
                plt.close()

                if self.lspmr_mode:
                    plt.figure()
                    p1, = plt.plot(x[indNum:], total_train_lspmr_loss[indNum:])
                    p2, = plt.plot(x[indNum:], total_val_lspmr_loss[indNum:])
                    plt.xlabel('epoch')
                    plt.ylabel('lspmr_loss')
                    plt.legend(handles=[p1, p2, ], labels=['train', 'val'], loc='best')
                    plt.legend(handles=[p1, p2, ], labels=['train', 'val'], loc='best')
                    plt.savefig(os.sep.join([sub_path_pre, 'loss', sub_path_last, 'lspmr_loss.png']))
                    plt.close()

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            if epoch % displayStep == 0:
                print("Time per epoch: %s, Est. complete in: %s" % (
                    str(timedelta(seconds=time_per_epoch)),
                    str(timedelta(seconds=seconds_left))))
            if self.should_save_model:
                if epoch % displayStep == 0:
                    self.save_model()


        total_training_time = time.time() - total_start_time
        if epoch % displayStep == 0:
            print("\nTotal training time: %s" % str(timedelta(
                seconds=total_training_time)))

    def train_one_epoch(self, data, batch_size, learning_rate):
        if self.lspmr_mode is True:
            num_examples = data.num_examples
            total_mae = []
            total_lspmr = []
            for i in range(num_examples // batch_size):
                batch = data.next_batch(batch_size)
                images, labels, recon_labels, adjacentMatrix = batch
                feed_dict = {
                    self.images: images,
                    self.labels: labels,
                    self.recon_labels: recon_labels,
                    self.adjacentMatrix: adjacentMatrix,
                    self.learning_rate: learning_rate,
                    self.is_training: True,
                }
                fetches = [self.train_step, self.mae_loss, self.lspmr_loss]
                _, mae, lspmr_loss = self.sess.run(fetches, feed_dict=feed_dict)
                total_mae.append(mae)
                total_lspmr.append(lspmr_loss)
                if self.should_save_logs:
                    self.batches_step += 1
                    self.log_loss(mae, self.batches_step, prefix='per_batch',should_print=False, lspmr_loss=lspmr_loss,)
            mean_mae = np.mean(total_mae)
            mean_lspmr = np.mean(total_lspmr)
            return mean_mae, mean_lspmr
        else:
            num_examples = data.num_examples
            total_loss = []
            for i in range(num_examples // batch_size):
                batch = data.next_batch(batch_size)
                images, labels, recon_labels, adjacentMatrix = batch
                feed_dict = {
                    self.images: images,
                    self.labels: labels,
                    self.recon_labels: recon_labels,
                    self.adjacentMatrix: adjacentMatrix,
                    self.learning_rate: learning_rate,
                    self.is_training: True,
                }
                fetches = [self.train_step, self.mae_loss]
                result = self.sess.run(fetches, feed_dict=feed_dict)
                _, loss = result
                total_loss.append(loss)
                if self.should_save_logs:
                    self.batches_step += 1
                    self.log_loss(
                        loss, self.batches_step, prefix='per_batch',
                        should_print=False)
            mean_loss = np.mean(total_loss)
            return mean_loss


    def test(self, data, batch_size):
        if self.lspmr_mode is True:
            num_examples = data.num_examples
            total_mae = []
            total_lspmr_loss = []
            batch_num = num_examples // batch_size
            for i in range(batch_num):
                batch = data.next_batch(batch_size)
                images, labels, recon_labels, adjacentMatrix = batch
                feed_dict = {
                    self.images: images,
                    self.labels: labels,
                    self.recon_labels: recon_labels,
                    self.adjacentMatrix: adjacentMatrix,
                    self.is_training: False,
                }
                mae, lspmr_loss = self.sess.run([self.mae_loss, self.lspmr_loss], feed_dict=feed_dict)
                total_mae.append(mae)
                total_lspmr_loss.append(lspmr_loss)
            mean_mae = np.mean(total_mae)
            mean_lspmr = np.mean(total_lspmr_loss)
            return mean_mae, mean_lspmr
        else:
            num_examples = data.num_examples
            total_loss = []
            for i in range(num_examples // batch_size):
                batch = data.next_batch(batch_size)
                images, labels, recon_labels, adjacentMatrix = batch
                feed_dict = {
                    self.images: images,
                    self.labels: labels,
                    self.recon_labels: recon_labels,
                    self.adjacentMatrix: adjacentMatrix,
                    self.is_training: False,
                }
                loss = self.sess.run(self.mae_loss, feed_dict=feed_dict)
                total_loss.append(loss)
            mean_loss = np.mean(total_loss)
            return mean_loss

    def predict(self, data, batch_size):
        num_examples = data.num_examples
        batch_num = math.ceil(num_examples / batch_size)
        images = data.images
        labels = data.labels
        pre = np.zeros(labels.shape)
        for i in range(batch_num):
            batch_x = images[i * batch_size : (i+1) * batch_size]
            batch_y = labels[i * batch_size : (i+1) * batch_size]
            feed_dict = {
                self.images: batch_x,
                self.labels: batch_y,
                self.is_training: False,
            }
            pre[i * batch_size : (i+1) * batch_size] = self.sess.run(self.prediction, feed_dict=feed_dict)
        return pre

    def get_embedding(self, data, batch_size):
        num_examples = data.num_examples
        batch_num = math.ceil(num_examples / batch_size)
        images = data.images
        fea = self.sess.run(self.feature, feed_dict={self.images:images[0:2], self.is_training:False})
        total_fea = np.zeros((images.shape[0], fea.shape[1]))
        for i in range(batch_num):
            batch_x = images[i * batch_size: (i + 1) * batch_size]
            feed_dict = {
                self.images: batch_x,
                self.is_training: False,
            }
            total_fea[i * batch_size: (i + 1) * batch_size] = self.sess.run(self.feature, feed_dict=feed_dict)
        return total_fea

    def get_feature_map(self, images, layerName):
        featureMap = self.sess.run(self.layer[layerName], feed_dict={self.images: images, self.is_training:False})
        return featureMap
