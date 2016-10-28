# inspired by https://github.com/fastforwardlabs/vae-tf
from datetime import datetime
import numpy as np
import tensorflow as tf
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
    """ Variational Autoencoder implementation in TensorFlow"""

    HYPERPARAMS = {
            "batch_size": 128,
            "learning_rate": 1e-3,
            "dropout": 1.0,
            "nonlin": tf.nn.relu, # TODO relu
            "squash": tf.nn.sigmoid, # TODO explore (softplus?)
            }

    def __init__(self,
            network_architecture = [], # list of nodes per layer
            hyper_params = {}, # update to HYPERPARAMS
            ):
        # insert updated hyper parameters into class
        tf.reset_default_graph()
        self.__dict__.update(VAE.HYPERPARAMS, **hyper_params)
        self.sess = tf.Session()

        print('Building tensorflow graph')

        # build tf graph according to network_architecture and hyperparams
        handles = self._build_graph(network_architecture)

        self.constructed = datetime.now().strftime("%y%m%d_%H%M")
        logs_path = os.path.join("logs", "run_"+self.constructed)
        self.summary_writer = tf.train.SummaryWriter(logs_path, self.sess.graph)
        self.writer = tf.merge_all_summaries()

        self.sess.run(tf.initialize_all_variables())
        (self.x_in, self.dropout_, self.z_mean, self.z_log_sigma,
         self.x_reconstructed, self.z_, self.x_reconstructed_,
         self.cost, self.train_op) = handles

        print('Done constructing network')

    def _build_recognition_graph(self, architecture):
        # Graph input
        x_in = tf.placeholder(tf.float32, shape=[None, architecture[0]], name="x")


        # setup dropout placeholder for dense layers
        dropout = tf.placeholder_with_default(self.dropout, shape=[], name = "dropout")

        # encoding (q(z|x)), build latent space
        encoding = [Dense("encoding", hidden_size, dropout, self.nonlin) for hidden_size in reversed(architecture[1:-1])]

        # compose all dense layers
        h_encoded = compose_all(encoding)(x_in)

        # latent dist: z ~ N(z_mean, np.exp(z_log_sigma)**2)
        z_mean = Dense("z_mean", architecture[-1], dropout)(h_encoded)
        z_log_sigma = Dense("z_log_sigma", architecture[-1], dropout)(h_encoded)
        return (x_in, dropout, z_mean, z_log_sigma)

    def _build_decoding_graph(self, architecture, dropout, vars):
        (x_in, z_mean, z_log_sigma) = vars
        # sample from gaussian
        z = self.sample_gaussian(z_mean, z_log_sigma)


        decoding = [Dense("decoding", hidden_size, dropout, self.nonlin) for hidden_size in architecture[1:-1]]

        # prepend squashing func for reconstructed x
        decoding.insert(0, Dense("x_decoding", architecture[0], dropout, self.squash))

        x_reconstructed = tf.identity(compose_all(decoding)(z), name = "x_reconstructed")

        tf.image_summary('x_reconstructed', tf.reshape(x_reconstructed,[self.batch_size, 28, 28, 1]))



        # aux. ops to explore latent space, prior z ~ N(0,I)
        z_ = tf.placeholder_with_default(tf.random_normal([1, architecture[-1]]),
                shape = [None, architecture[-1]],
                name = "latent_in")
        x_reconstructed_ = compose_all(decoding)(z_)

        # reconstruction loss, mismatch between input and reconstructed
        rec_loss = VAE.rec_error(x_reconstructed, x_in)

        # determine kullback-leibler divergence (approx. to true posterior)
        kl_loss = VAE.kl_divergence(z_mean, z_log_sigma)

        return (decoding, x_reconstructed, z_, x_reconstructed_, rec_loss, kl_loss)


    def _build_graph(self, architecture):
        """
        Build tensorflow graph.
        First build (both reconstruction and generator network).

        Args:
            architecture(list[int]): number of layers
        """
        
        # recognition graph
        (x_in, dropout, z_mean, z_log_sigma) = \
                self._build_recognition_graph(architecture)

        # decoding graph
        (decoding, x_reconstructed, z_, x_reconstructed_, rec_loss, kl_loss) = \
                self._build_decoding_graph(architecture, dropout, (x_in, z_mean, z_log_sigma))


        tf.histogram_summary('rec_loss', rec_loss)
        tf.histogram_summary('kl_loss', kl_loss)
        # loss for this graph
        with tf.name_scope("cost"):
            cost = tf.reduce_mean(rec_loss + kl_loss, name = "vae_cost")
        tf.scalar_summary('vae_cost', cost)

        # optimizer
        with tf.name_scope("Adam_optimizer"):
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        with tf.name_scope("latent_in"):
             z_ = tf.placeholder_with_default(tf.random_normal([1, architecture[-1]]),
                     shape = [None, architecture[-1]],
                     name = "latent_in")
             x_reconstructed_ = compose_all(decoding)(z_)
        return (x_in, dropout, z_mean, z_log_sigma, x_reconstructed, z_, x_reconstructed_,
                cost, train_op)

    def sample_gaussian(self,mu, log_sigma):
        """Sample from normal distribution using reparameterization trick.
        First sample epsilon ~ N(mu, I*sigma**2).
        Then return z = mu + epsilon*exp(log_sigma)

        Args:
            mu: mean of distribution
            log_sigma: log(sigma) for distribution
        Returns:
            sample from provided gaussian
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

    def plot_manifold(self, range_x=20, range_y=20, output="manifold.pdf"):
        mid_x = range_x // 2
        mid_y = range_y // 2
        x1 = range(range_x)
        x2 = range(range_y)
        plt.figure(figsize=(range_x, range_y))
        gs = gridspec.GridSpec(len(x1), len(x2))
        gs.update(wspace=0, hspace=0)
        pbar = ProgressBar(max_value = len(x1)*len(x2))
        pbar.start()
        i=0
        for x in x1:
            for y in x2:
                ax = plt.subplot(gs[x,y])
                ax.set_yticks([])
                ax.set_xticks([])
                img = self.decode([[x - mid_x,y - mid_y]])
                ax.imshow(img.reshape(28,28), cmap=plt.cm.gray)
                pbar.update(i)
                i += 1
        plt.savefig(output)


    
    def train(self, X, num_epochs = np.inf):
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

                feed_dict = {self.x_in: x, self.dropout_: self.dropout}
                x_reconstructed, cost, summary, _ = self.sess.run([
                    self.x_reconstructed,
                    self.cost,
                    self.writer,
                    self.train_op,],
                    {
                        self.x_in: x,
                        self.dropout_: self.dropout,
                    })
                avg_error_train += cost / num_samples * self.batch_size

                # write summary every batch
                self.summary_writer.add_summary(summary, epoch)
            print("epoch {}: avg cost: {}".format(epoch, avg_error_train))

        now = datetime.now().strftime("%y%m%d_%H%M")
        print("[***] Training end: {}".format(now))
        print("average cost after {} epochs achieved: {}".format(num_epochs, avg_error_train))
        self.writer.close()
        return avg_error_train
