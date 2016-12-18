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
import matplotlib.gridspec as gridspec


from progressbar import ETA, Bar, Percentage, ProgressBar
import matplotlib.pyplot as plt
from IPython import display

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
            }

    def __init__(self,
            network_architecture = [], # list of nodes per layer
            hyper_params = {}, # update to HYPERPARAMS
            models_dir = "models/",
            k = 1, # number of forward passes
            **kwargs
            ):
        self.k_passes = k
        self.arch = network_architecture
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir)
        self.models_dir = models_dir
        # insert updated hyper parameters into class
        tf.reset_default_graph()
        self.__dict__.update(VAE.HYPERPARAMS, **kwargs)
        self.sess = tf.Session()

        print('Building tensorflow graph with the following hyper parameters:')
        for k in VAE.HYPERPARAMS:
            print('\t{}: {}'. format(k, self.__dict__[k]))
        print('With the following network architecture: [{}]'.format(",".join(map(str,self.arch))))


        # build tf graph according to hyperparams
        self._build_graph()

        self.constructed = datetime.now().strftime("%y%m%d_%H%M")
        logs_path = os.path.join("logs", "run_"+self.constructed)
        self.summary_writer = tf.summary.FileWriter(logs_path, self.sess.graph)
        self.writer = tf.summary.merge_all()
        # self.saver = tf.train.Saver()

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

    def plot_reconstruction(self, data, save_dir = False):
        """Try to reconstruct input data using the VAE Network

        Args:
            data(Dataset): input data, where we will get
                           one batch (self.batch_size large) for reconstruction
            save_dir(bool): If true, save all reconstructions (and originals)
                            in the directory "out/run_"+self.constructed
                            Otherwise, print one single image and its reconstruction
        """
        # setup plots
        x,_ = data.validation.next_batch(self.batch_size)
        fetches = [self.x_reconstructed, self.cost]
        feed_dict = {self.x_in: x}
        x_reconstructed, cost = self.sess.run(fetches, feed_dict)
        print("cost for validation batch: {}".format(cost))
        f, (plot_left, plot_right) = plt.subplots(1,2, sharey=True)
        for plot in (plot_left, plot_right):
            plot.set_yticks([])
            plot.set_xticks([])
        if save_dir == False:
            img_orig = x[0]
            img_rec = x_reconstructed[0]
            plot_left.imshow(img_orig.reshape(28,28), cmap=plt.cm.gray)
            plot_right.imshow(img_rec.reshape(28,28), cmap=plt.cm.gray)
        else:
            dir = os.path.join("out","run_" + self.constructed)
            if not os.path.exists(dir):
                os.makedirs(dir)
            print("write images to " + dir)
            pbar = ProgressBar(max_value = self.batch_size)
            pbar.start()
            for i in range(0,self.batch_size):
                img_orig = x[i]
                img_rec = x_reconstructed[i]
                plot_left.imshow(img_orig.reshape(28,28), cmap=plt.cm.gray)
                plot_right.imshow(img_rec.reshape(28,28), cmap=plt.cm.gray)
                path = os.path.join(dir, "img_{}.png".format(i))
                f.savefig(path)
                pbar.update(i)
            plt.close('all')
        print("done")

    def plot_manifold(self, range_x=(-4,4), range_y=(-4,4), axis = (0,1), single_axis = False, num_rowdigits = 20, output="manifold.pdf"):
        x1 = np.linspace(range_x[0], range_x[1], num = num_rowdigits)
        if not single_axis:
            x2 = np.linspace(range_y[0], range_y[1], num = num_rowdigits)
        else:
            x2 = np.zeros((1, num_rowdigits))

        manifold = np.zeros(shape=(len(x1)*28, len(x2)*28))
        for i_x, x in enumerate(x1):
            for i_y, y in enumerate(x2):
                input = np.zeros(shape=(1, self.arch[-1]))
                input[0][axis[0]] = x
                if not single_axis:
                    input[0][axis[1]] = y
                img = self.decode(input)
                manifold[i_x*28:(i_x+1)*28,i_y*28:(i_y+1)*28] = img.reshape(28,28)
        plt.imshow(manifold, cmap=plt.cm.gray)
        plt.axis('off')
        plt.savefig(output)

    def plot_z_mean(self, data, num_samples = 5000):
        x, y = data.test.next_batch(num_samples)
        z_mean, _ = self.encode(x)
        if self.arch[-1] == 2:
            plt.scatter(z_mean[:,0], z_mean[:,1], c=np.argmax(y, 1), alpha=1, edgecolors='black')
            plt.show()
        elif self.arch[-1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(z_mean[:,0], z_mean[:,1], z_mean[:,2], c=np.argmax(y,1), s = 40, marker='.')
            plt.show()



    def train(self, X, num_epochs = 75, plot_manifold = False):
        avg_train_error = 0
        now = datetime.now().isoformat()[11:]
        print("[**] Begin training: {}".format(now))
        cost_test = []
        i = 0
        num_samples = X.train.num_examples
        num_batches = int(num_samples / self.batch_size)
        avg_error_train = 0

        for epoch in range(num_epochs):
            avg_error_train = 0
            for i in range(num_batches):
                # get mini-batch
                x, _ = X.train.next_batch(self.batch_size)

                feed_dict = {self.x_in: x}
                x_reconstructed, cost, summary, _ = self.sess.run([
                    self.x_reconstructed,
                    self.cost,
                    self.writer,
                    self.train_op,],
                    {
                        self.x_in: x,
                    })
                avg_error_train += cost / num_samples * self.batch_size

                # write summary every batch
                self.summary_writer.add_summary(summary, epoch)
            # model_path = os.path.join(self.models_dir, "model_{}_epoch_{:03d}.cpkt".format(self.constructed, epoch))
            # save_path = self.saver.save(self.sess, model_path)
            print("epoch {}: avg cost: {}".format(epoch, avg_error_train))
            if plot_manifold:
                self.plot_manifold(range_x=(-1,1), range_y=(-1,1), output = "manifold_{:03}.pdf".format(epoch))
                print("done plotting manifold")

        now = datetime.now().strftime("%y%m%d_%H%M")
        print("[***] Training end: {}".format(now))
        print("average cost after {} epochs achieved: {}".format(num_epochs, avg_error_train))
        return avg_error_train
