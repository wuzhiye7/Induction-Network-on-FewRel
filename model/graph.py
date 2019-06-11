# -*- coding: utf-8 -*-
"""
Created on: 2019/5/27 11:08
@Author: zsfeng
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from model.layer_module import neural_tensor_layer, self_attention, dynamic_routing
from model.base import Base
import numpy as np


class InductionGraph(Base):
    def __init__(self, N, K, Q, **kwds):
        """       
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        """
        Base.__init__(self, kwds)
        self.num_classes = N
        self.support_num_per_class = K
        self.query_num_per_class = Q

        self.build()

    def forward(self):
        with tf.name_scope("EncoderModule"):
            embedded_words = self.get_embedding()  # (k*c,seq_len,emb_size)

            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)  # forward direction cell
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)  # backward direction cell
            if self.keep_prob is not None:
                lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.keep_prob)
                lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.keep_prob)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_words,
                                                         dtype=tf.float32)
            output_rnn = tf.concat(outputs, axis=2)  # [k*c,sequence_length,hidden_size*2]
            encoder = self_attention(output_rnn)  # (k*c,hidden_size*2)
            support_encoder = tf.slice(encoder, [0, 0],
                                       [self.num_classes * self.support_num_per_class, self.hidden_size * 2])
            query_encoder = tf.slice(encoder, [self.num_classes * self.support_num_per_class, 0],
                                     [self.num_classes * self.query_num_per_class, self.hidden_size * 2])

        with tf.name_scope("InductionModule"):
            b_IJ = tf.constant(
                np.zeros([self.num_classes, self.support_num_per_class], dtype=np.float32))

            class_vector = dynamic_routing(
                tf.reshape(support_encoder, [self.num_classes, self.support_num_per_class, -1]),
                b_IJ)  # (k,hidden_size*2)

        with tf.name_scope("RelationModule"):
            self.probs = neural_tensor_layer(class_vector, query_encoder)

    def build_loss(self):
        with tf.name_scope("loss"):
            labels_one_hot = tf.one_hot(self.query_label, self.num_classes, dtype=tf.float32)
            losses = tf.losses.mean_squared_error(labels=labels_one_hot, predictions=self.probs)

            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            self.loss = losses + l2_losses
