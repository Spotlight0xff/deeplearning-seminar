#!/usr/bin/env python3
from datetime import datetime
import numpy as np
import tensorflow as tf
import prettytensor as pt
import sys
import os
from numpy.random import uniform,random_sample
from progressbar import ETA, Bar, Percentage, ProgressBar


class GAN(object):
    """Generative adversarial networks implementation in TensorFlow/PrettyTensor

    Just for educational purposes, generator + discriminator implemented using MLP.
    """

    HYPERPARAMS = {
            "batch_size": 128,
            "learning_rate": 1e-2,
            "beta1": 0.5, # for AdamOptimizer
            "z_dim": 1, # dimension of noise prior
            "h_dim": 5, # dim of hidden layer in D and G
            "k": 1, # updates of discriminator
            "data_dim": 1, # dimensionality of generated data
            "nonlin": tf.nn.relu, # TODO LeakyReLU
            "squash": tf.nn.tanh,
            }

    def __init__(self,
            hyper_params = {}, # update to HYPERPARAMS
            models_dir = "models/",
            ):
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir)
        self.models_dir = models_dir

        tf.reset_default_graph()

        # insert updated hyper parameters into class
        self.__dict__.update(GAN.HYPERPARAMS, **hyper_params)

        print('Building tensorflow graph')
        self.sess = tf.Session()
        self._build_graph()

        self.constructed = datetime.now().strftime("%y%m%d_%H%M")
        logs_path = os.path.join("logs", "run_"+self.constructed)
        self.writer = tf.train.SummaryWriter(logs_path, self.sess.graph)
        self._build_summaries()

        self.sess.run(tf.initialize_all_variables())

    def _build_summaries(self):
            summary_d = [
                tf.scalar_summary('loss_d_real', self.loss_d_real),
                tf.scalar_summary('loss_d_fake', self.loss_d_fake),
                tf.scalar_summary('loss_d', self.loss_d),
                ]
            summary_g = [
                tf.scalar_summary('loss_g', self.loss_g)
            ]
            self.merged_d = tf.merge_summary(summary_d)
            self.merged_g = tf.merge_summary(summary_g)

    def _build_graph(self):
        """Build tensorflow graph (discriminator + generator)"""
        self.data_tensor = tf.placeholder(np.float32, shape=(self.batch_size, self.data_dim))
        self.noise_tensor = tf.placeholder(np.float32, shape=(self.batch_size, self.z_dim))


        self.output_g = self._build_generator(self.noise_tensor)
        self.output_d_real = self._build_discriminator(self.data_tensor)
        self.output_d_fake = self._build_discriminator(self.output_g)

        # discriminator loss (for both cases)
        # max log(D(x)) + log(1 - D(z))
        self.loss_d_real = -tf.reduce_mean(tf.log(self.output_d_real + 1e-12))
        self.loss_d_fake = -tf.reduce_mean(tf.log((1 - self.output_d_fake) + 1e-12))
        self.loss_d = self.loss_d_real + self.loss_d_fake

        # generator loss
        # max D(G(z))
        self.loss_g = -tf.reduce_mean(tf.log(self.loss_d_fake + 1e-12))

        vars_d = tf.get_collection('discriminator')
        vars_g = tf.get_collection('generator')
        print(len(vars_d))
        print(len(vars_g))
        self.opt_d = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1)\
                    .minimize(self.loss_d, var_list = vars_d)

        self.opt_g = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1)\
                    .minimize(self.loss_g, var_list = vars_g)



    def _build_discriminator(self, input_data):
        """Build discriminator network to distinguish between 'fake' and 'real' data.

        Args:
            input_data(tf.Tensor): input data to be discriminated
        """
        with pt.defaults_scope(activation_fn=tf.nn.tanh, variable_collections=['discriminator']):
            return (pt.wrap(input_data).flatten()
                   .fully_connected(self.h_dim)
                   .fully_connected(1, activation_fn=tf.nn.sigmoid)
                   ).tensor

    def _build_generator(self, input_data):
        """Build generator network to generate fake samples from the noise prior."""
        with pt.defaults_scope(activation_fn=tf.nn.tanh, variable_collections=['generator']):
            return (pt.wrap(input_data)
                   .flatten()
                   .fully_connected(self.h_dim)
                   .fully_connected(self.data_dim)).tensor

    def _gen_prior(self):
        """return z ~ prior

        normalized to [-1,1]
        """
        return random_sample((self.batch_size, self.z_dim))
        # low = -2
        # high = 4
        # return uniform(low, high, (self.batch_size,self.z_dim))

    def decode(self, zs = None):
        """Sample x ~ G(z)
        """
        prior = self._gen_prior() if zs is None else zs
        return self.sess.run(self.output_g, {self.noise_tensor: prior})
    


    def train(self, X, num_epochs = 2000, modulo = 100, output = True):
        now = datetime.now().isoformat()[11:]
        if output:
            print("[**] Begin training: {}".format(now))
        count = 0
        l_d = 0
        summary_d = None
        n_k = self.k

        for epoch in range(0, num_epochs):
            # update discriminator k times
            for k in range(0, n_k):
                prior = self._gen_prior()
                l_d,summary_d, _ = self.sess.run([self.loss_d, self.merged_d, self.opt_d], {
                    self.data_tensor: X,
                    self.noise_tensor: prior,
                    })

            # update generator once
            prior = self._gen_prior()
            l_g,summary_g, _ = self.sess.run([self.loss_g, self.merged_g, self.opt_g], {
                self.noise_tensor: prior
                })


            # write summaries
            if epoch % modulo == 0:
                l_d, l_g, summary_d, summary_g = \
                        self.sess.run([
                            self.loss_d,
                            self.loss_g,
                            self.merged_d,
                            self.merged_g
                        ], {
                            self.data_tensor: X,
                            self.noise_tensor: prior,
                        })
                self.writer.add_summary(summary_d, count)
                self.writer.add_summary(summary_g, count)
                if output:
                    print('{:>5}@{:<2} loss_d: {:.3}\tloss_g: {:.3}'.format(count, n_k, l_d, l_g))
                count += 1


        now = datetime.now().strftime("%y%m%d_%H%M")
        if output:
            print("[***] Training end: {}".format(now))
