# inspired by https://github.com/fastforwardlabs/vae-tf
from datetime import datetime
import numpy as np
import tensorflow as tf
import prettytensor as pt
import sys
import os
import logging
from Layer.Dense import Dense
from Utils import compose_all
import pickle


class VAE(object):
    """Variational Autoencoder implementation in TensorFlow using PrettyTensor

    TODO:
    - batch normalization
    - convolutional network
    - inverse normalizing flow stuff
    - dropout
    """

    HYPERPARAMS = {
            "batch_size": 128,
            "learning_rate": 1e-3,
            "nonlin": tf.nn.relu, # TODO relu
            "squash": tf.nn.sigmoid, # TODO explore (softplus?)
            "arch": [784,512,512,50],
            }

    def __init__(self,
            hyper_params = {}, # update to HYPERPARAMS
            models_dir = "models",
            k = 1, # number of forward passes
            name = None, # model name
            **kwargs
            ):
        self.k_passes = k
        # self.arch = network_architecture
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir)
        self.models_dir = models_dir
        # insert updated hyper parameters into class
        tf.reset_default_graph()
        self.__dict__.update(VAE.HYPERPARAMS, **kwargs)
        self.sess = tf.Session()
        self.saver = None
        self.name = name
        

        # Saves to $models_dir/$name/$name-.*.{meta,index,...}
        # if name:
            # self.meta_graph_path = os.path.join(self.models_dir, self.name, "{}.meta".format(name))
            # if not os.path.isdir(os.path.join(models_dir, name)):
                # os.makedirs(os.path.join(models_dir, name))
            # else:
                # print('Trying to restore model "{}"'.format(name))
                # if os.path.isfile(self.meta_graph_path):
                    # saver = tf.train.import_meta_graph(self.meta_graph_path)
                    # saver.restore(self.sess, tf.train.latest_checkpoint('{}/{}/'))
                    # restored_vars = tf.trainable_variables()
                    # train_vars = tf.trainable_variables()
                    # print('Restored {} trainable variables:'.format(len(train_vars)))
                    # for v in train_vars:
                        # print("\t{}".format(v.name))
                    # return

        print('Building tensorflow graph with the following hyper parameters:')
        for k in VAE.HYPERPARAMS:
            print('\t{}: {}'. format(k, self.__dict__[k]))
        print('With the following network architecture: [{}]'.format(",".join(map(str,self.arch))))
        # build tf graph according to hyperparams
        self._build_graph()
        self.saver = tf.train.Saver()


        self.constructed = datetime.now().strftime("%y%m%d_%H%M")
        logs_path = os.path.join("logs", "run_"+self.constructed)
        self.summary_writer = tf.summary.FileWriter(logs_path, self.sess.graph)
        self.writer = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())

        print('Done constructing network')

    def _build_recognition_graph(self):
        # Graph input
        self.x_in = tf.placeholder(tf.float32, shape=[None, self.arch[0]], name="x")

        layer = pt.wrap(self.x_in).sequential()
        layer.flatten()
        for i in self.arch[1:-1]:
            layer.fully_connected(i, activation_fn = self.nonlin, weights = tf.contrib.layers.xavier_initializer())

        self.z_mean = layer.as_layer().fully_connected(self.arch[-1], name = 'z_mean', weights = tf.contrib.layers.xavier_initializer(), activation_fn = None).tensor
        self.z_log_sigma = layer.as_layer().fully_connected(self.arch[-1], name = 'z_log_sigma', weights = tf.contrib.layers.xavier_initializer(), activation_fn = None).tensor

    def _build_decoding_layers(self):
        dec_layer = pt.template('input').sequential()
        dec_layer.flatten()
        for count, i in enumerate(reversed(self.arch[1:-1])):
            dec_layer.fully_connected(i, activation_fn = self.nonlin, weights = tf.contrib.layers.xavier_initializer())
        dec_layer.fully_connected(self.arch[0], activation_fn=self.squash, weights = tf.contrib.layers.xavier_initializer())
        return dec_layer


    def _build_decoding_graph(self):
        self.z = self.sample_gaussian(self.z_mean, self.z_log_sigma)
        self.x_reconstructed = self.dec_layers.construct(input = self.z)


    def _build_loss_graph(self):
        # reconstruction loss, mismatch between input and reconstructed
        self.rec_loss = VAE.rec_error(self.x_reconstructed, self.x_in)

        # determine kullback-leibler divergence (approx. to true posterior)
        self.kl_loss = VAE.kl_divergence(self.z_mean, self.z_log_sigma)

        self.cost = tf.reduce_mean(self.rec_loss + self.kl_loss, name = "vae_cost")


    def _build_generator_graph(self):
        """Build the decoupled generator network
        for latent space exploration, manifold plotting etc.
        """
        self.z_ = tf.placeholder_with_default(tf.random_normal([1, self.arch[-1]]),
                shape = [None, self.arch[-1]],
                name = "latent_in")
        self.x_reconstructed_ = self.dec_layers.construct(input = self.z_).tensor

    def _build_graph(self):
        """
        Build tensorflow graph.
        First build (both reconstruction and generator network).

        Args:
            self.arch(list[int]): number of layers
        """

        # Build first autoencoder graph
        # recognition graph
        self._build_recognition_graph()


        # build template for the decoding network (in autoencoder)
        # and generator network (decoupled for explicit latent prior exploration)
        self.dec_layers = self._build_decoding_layers()

        # decoding graph
        self._build_decoding_graph()

        self._build_loss_graph()


        # optimizer
        tf.summary.scalar('vae_cost', self.cost)
        with tf.name_scope("Adam_optimizer"):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        # decoupled generator network for latent space exploration
        self._build_generator_graph()


    def sample_gaussian(self,mu, log_sigma):
        """Sample from normal distribution using reparameterization trick.
        First sample epsilon ~ N(mu, I*sigma**2).
        Then return z = mu + epsilon*exp(log_sigma)

        Args:
            mu: mean of distribution
            log_sigma: log(sigma) for distribution
        Returns:
            sample from provided gaussian (shape (1,k))
        """
        with tf.name_scope("sample_gaussian"):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(log_sigma), name = "epsilon")
            return mu + epsilon * tf.exp(log_sigma)

    @staticmethod
    def rec_error(obs, actual, offset=1e-7):
        """Reconstruction error per sample

        Args:
            obs(tf.Tensor): observed tensor
            actual(tf.Tensor): actual tensor
            offset(tf.Tensor): offset, to avoid evaluating log(0)
        Returns:
            tf.Tensor: sum of reconstruction error
        """
        with tf.name_scope("rec_error"):
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs_) +
                                  (1 - actual) * tf.log(1 - obs_), 1)
    @staticmethod
    def kl_divergence(mu, log_sigma):
        with tf.name_scope("KL_divergence"):
            return -0.5 * tf.reduce_sum(1 + 2* log_sigma - mu**2 -
                                      tf.exp(2*log_sigma), 1)

    def encode(self, x):
        feed_dict = {self.x_in: x}
        return self.sess.run([self.z_mean, self.z_log_sigma], feed_dict=feed_dict)

    def decode(self, zs = None):
        feed_dict = dict()
        if zs is not None:
            is_tensor = lambda x: hasattr(x, "eval")
            zs = (self.sess.run(zs) if is_tensor(zs) else zs) # --> np.array
            feed_dict.update({self.z_: zs})
        return self.sess.run(self.x_reconstructed_, feed_dict = feed_dict)

    def end_to_end(self, x):
        """End-to-end pass for the VAE"""
        return self.decode(self.sample_gaussian(*self.encode(x)))



    def train(self, X, num_epochs = 75):
        now = datetime.now().isoformat()[11:]
        print("[**] Begin training: {}".format(now))
        i = 0
        num_samples_train = X.train.num_examples
        num_batches_train = int(num_samples_train / self.batch_size)
        avg_error_train = 0

        num_samples_test = X.test.num_examples
        num_batches_test = int(num_samples_test / self.batch_size)
        avg_error_test = 0

        for epoch in range(num_epochs):

            ### Training
            avg_error_train = 0
            for i in range(num_batches_train):
                # get mini-batch
                x, _ = X.train.next_batch(self.batch_size)

                x_reconstructed, cost, summary, _ = self.sess.run([
                    self.x_reconstructed,
                    self.cost,
                    self.writer,
                    self.train_op,],
                    {
                        self.x_in: x,
                    })
                avg_error_train += cost / num_samples_train * self.batch_size

            ### Testing
            avg_error_test = 0
            for i in range(num_batches_test):
                test_set = X.test.next_batch(self.batch_size)
                _, test_error = self.sess.run([
                    self.x_reconstructed,
                    self.cost],
                    {
                        self.x_in: x,
                    })
                avg_error_test += test_error / num_samples_test * self.batch_size



            print("epoch {}: train cost: {} | test cost: {}".format(epoch, avg_error_train, avg_error_test))

            yield [epoch, avg_error_train, avg_error_test]

        now = datetime.now().strftime("%y%m%d_%H%M")
        print("[***] Training end: {}".format(now))
        print("After {} epochs of training:")
        print("Train error: {}".format(avg_error_train))
        print("Test error: {}".format(avg_error_test))
        return avg_error_train

    def load_hyper(self, path):
        new_hyper = pickle.load(open(path, 'rb'))
        self.__dict__.update(new_hyper)

    def save_hyper(self, path):
        with open(path, 'wb') as output:
            hyperparams = {k: self.__dict__[k] for k in self.HYPERPARAMS}
            pickle.dump(hyperparams, output, pickle.HIGHEST_PROTOCOL)

    def save(self,epoch):
        if not self.name:
            return
            # self.name = "-".join(map(str,self.arch))

        model_path = os.path.join(self.models_dir, self.name, "epoch_{:05d}".format(epoch))
        self.saver.save(self.sess, model_path, write_meta_graph=True)
        # if not os.path.isfile(self.meta_graph_path):
            # tf.train.export_meta_graph(self.meta_graph_path)
            # print('Save metagraph at ' + self.meta_graph_path)

        instance = os.path.join(self.models_dir, self.name, "hyperparams.pkl")
        if not os.path.isfile(instance):
            self.save_hyper(instance)
            print('Saved instance at ' + instance)

        print('Saved model @ epoch {}: {}'.format(epoch, model_path))
