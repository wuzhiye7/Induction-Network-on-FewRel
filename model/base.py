# -*- coding: utf-8 -*-
"""
Created on: 2019/5/30 15:22
@Author: zsfeng
"""
import tensorflow as tf
from os.path import exists, join
from os import makedirs


class Base:
    def __init__(self, kwds):
        self.sequence_length = kwds.get("sequence_length", 40)
        self.vocab_size = kwds.get("vocab_size", None)
        self.embed_size = kwds.get("embed_size", 50)
        self.hidden_size = kwds.get("hidden_size", 100)
        self.is_training = kwds.get("is_training", True)
        self.learning_rate = kwds.get("learning_rate", 0.001)
        self.initializer = kwds.get("initializer", tf.random_normal_initializer(stddev=0.1))
        self.decay_steps = kwds.get("decay_steps", 100)
        self.decay_rate = kwds.get("decay_rate", 0.9)
        self.l2_lambda = kwds.get("l2_lambda", 0.0001)
        self.embed = kwds.get("pred_embed", None)
        # self.epoch_num = kwds.get("epoch_num", )
        self.pos_embedding_dim = 5
        self.keepProb = kwds.get("keep_prob", 0.9)

    def build(self):
        self.initial_params()
        self.forward()
        self.build_predict()
        self.build_accuracy()
        self.build_loss()
        self.build_optimize()
        self.build_summary()

    def initial_params(self):
        # step
        self.global_step = tf.Variable(name="global_step", initial_value=0, trainable=False)

        # input
        self.input_words = tf.placeholder(name="input_words", shape=[None, self.sequence_length], dtype=tf.int32)
        self.input_pos1 = tf.placeholder(name="input_pos1", shape=[None, self.sequence_length], dtype=tf.int32)
        self.input_pos2 = tf.placeholder(name="input_pos2", shape=[None, self.sequence_length], dtype=tf.int32)
        self.query_label = tf.placeholder(name="query_label", shape=[None], dtype=tf.int32)  # y [None,num_classes]
        self.keep_prob = tf.placeholder(name="keep_probx", dtype=tf.float32)

        # embedding matrix
        with tf.name_scope("embedding"):
            if self.embed is not None:
                self.word_embedding = tf.Variable(self.embed, trainable=False)
            else:
                self.word_embedding = tf.get_variable(name="word_embedding", shape=[self.vocab_size, self.embed_size],
                                                      initializer=self.initializer, trainable=True)
            self.pos1_embedding = tf.get_variable(name="pos1_embedding",
                                                  shape=[2 * self.sequence_length, self.pos_embedding_dim],
                                                  initializer=self.initializer, trainable=True)
            self.pos2_embedding = tf.get_variable(name="pos2_embedding",
                                                  shape=[2 * self.sequence_length, self.pos_embedding_dim],
                                                  initializer=self.initializer, trainable=True)

    def get_embedding(self):
        embedded_words = tf.nn.embedding_lookup(self.word_embedding, self.input_words)
        embedded_pos1 = tf.nn.embedding_lookup(self.pos1_embedding, self.input_pos1)
        embedded_pos2 = tf.nn.embedding_lookup(self.pos2_embedding, self.input_pos2)
        return tf.concat([embedded_words, embedded_pos1, embedded_pos2], axis=2)

    def forward(self):
        raise NotImplementedError

    def build_loss(self):
        raise NotImplementedError

    def build_optimize(self):
        # based on the loss, use SGD to update parameter
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                        self.decay_rate, staircase=True)
        self.optimize = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step,
                                                        learning_rate=self.learning_rate, optimizer="Adam")

    def build_predict(self):
        self.predict_test = tf.argmax(name="predictions", input=self.probs, axis=1)
        self.predict = tf.round(self.probs)

    def build_accuracy(self):
        correct_prediction = tf.equal(tf.cast(self.predict_test, tf.int32), self.query_label)
        self.accuracy = tf.reduce_mean(name="accuracy", input_tensor=tf.cast(correct_prediction, tf.float32))

    def build_summary(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('learning_rate', self.learning_rate)
        self.summary = tf.summary.merge_all()

    def train(self, dataloader, model_dir_path, model_name="inductionNetwork",
              train_iter=30000, val_iter=1000, val_step=2000, test_iter=3000, ):
        # 资源配置，自增长
        train_data_loader, val_data_loader = dataloader
        if not exists(model_dir_path):
            makedirs(model_dir_path)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            train_writer = tf.summary.FileWriter(join(model_dir_path, "train"), sess.graph)
            # val_writer = tf.summary.FileWriter(join(model_dir_path, "val"), sess.graph)

            sess.run(tf.global_variables_initializer())

            curr_iter = 0
            best_acc = 0.0
            not_best_count = 0  # Stop training after several epochs without improvement.

            print("training start ..")
            iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0
            for it in range(curr_iter, curr_iter + train_iter):
                inputs, query_label = train_data_loader.next_one_tf(self.num_classes,
                                                                    self.support_num_per_class,
                                                                    self.query_num_per_class)

                curr_loss, curr_acc, _, curr_summary, global_step = sess.run(
                    [self.loss, self.accuracy, self.optimize, self.summary, self.global_step],
                    feed_dict={self.input_words: inputs['word'],
                               self.input_pos1: inputs['pos1'],
                               self.input_pos2: inputs['pos2'],
                               self.query_label: query_label,
                               self.keep_prob: self.keepProb}
                )
                train_writer.add_summary(curr_summary, global_step)
                iter_loss += curr_loss
                iter_right += curr_acc
                iter_sample += 1
                if it % 100 == 0:
                    print(
                        '[train] step:{0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1,
                                                                                          iter_loss / iter_sample,
                                                                                          100 * iter_right / iter_sample) + '\r')

                if it % val_step == 0:
                    iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0

                if (it + 1) % val_step == 0:  # 验证集验证
                    print("evaluate start ..")
                    acc_val = self.eval(val_data_loader, sess, val_iter)
                    if acc_val > best_acc:
                        print("Best checkpoint.[EVAL] accuracy :{}".format(acc_val))
                        ckpt_dir = join(model_dir_path, "checkpoint")
                        if not exists(ckpt_dir):
                            makedirs(ckpt_dir)
                        save_path = join(ckpt_dir, model_name)
                        saver.save(sess, save_path, global_step=global_step)
                        best_acc = acc_val

            print("\n####################\n")
            print("Finish training :" + model_name)

            test_acc = self.eval(val_data_loader, sess, test_iter)
            print("Test accuracy: {}".format(test_acc))

    def eval(self, val_data_loader, sess, val_iter):

        iter_right_val, iter_sample_val = 0.0, 0.0
        for it_val in range(val_iter):
            inputs_val, query_label_val = val_data_loader.next_one_tf(self.num_classes,
                                                                      self.support_num_per_class,
                                                                      self.query_num_per_class)
            curr_loss_val, curr_acc_val, curr_summary_val = sess.run(
                [self.loss, self.accuracy, self.summary],
                feed_dict={self.input_words: inputs_val['word'],
                           self.input_pos1: inputs_val['pos1'],
                           self.input_pos2: inputs_val['pos2'],
                           self.query_label: query_label_val,
                           self.keep_prob: 1}
            )
            # val_writer.add_summary(curr_summary_val, it_val)
            iter_right_val += curr_acc_val
            iter_sample_val += 1
            print(
                '[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it_val + 1,
                                                                  100 * iter_right_val / iter_sample_val) + '\r')
            acc_val = iter_right_val / iter_sample_val
            return acc_val
