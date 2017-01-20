#!/usr/bin/env python3
from Network import VAE
import numpy as np
import matplotlib.pyplot as plt
from random import choice,choices
import os
import input_data
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)




class Model:
    def __init__(self, arch):
        self.arch = arch
        self.latent_dim = arch[-1]
        self.model_name = "-".join(map(str,arch))
        results_dir = "results"
        self.results_dir = os.path.join(results_dir, self.model_name)
        print('Using results directory: ' + self.results_dir)

        # create results directory if needed
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)

        self.v = VAE.VAE(arch=arch, name = self.model_name)

    def train(self, num_epochs):
        loss = np.array(list(self.v.train(mnist, num_epochs = num_epochs)))
        return loss

    def get_pdf(self, name):
        return os.path.join(self.results_dir, name) + ".pdf"

    def plot_loss(self, loss):
        sns.set_style("darkgrid")
        train_loss, = plt.plot(loss[:,0], loss[:,1], label = 'Train loss')
        test_loss, = plt.plot(loss[:,0], loss[:,2], label = 'Test loss')
        plt.xlabel('Epochs')
        plt.ylabel('Negative log-likelihood')
        plt.legend(handles=[train_loss, test_loss])
        formatter = FormatStrFormatter('%d')
        plt.gca().get_xaxis().set_major_formatter(formatter)
        plt.savefig(self.get_pdf('loss'))
        sns.set_style(None)
        # also save loss to file
        np.savetxt(os.path.join(self.results_dir, 'loss.np'), loss)
        plt.close()

    def plot_manifold(self, axes = (0,1)):
        """Plots a manifold along the specified `axes`
        Please only do this if you have a two-dimensional latent space.
        To get good manifold in higher dimensions, you probably want to
        sample from a n-dimensional z-sphere."""
        # plot 2D manifold
        range_x1 = (-3, 3)
        range_x2 = (-3, 3)
        num_rowdigits = 20
        x1 = np.linspace(range_x1[0], range_x1[1], num = num_rowdigits)
        x2 = np.linspace(range_x2[0], range_x2[1], num = num_rowdigits)
        output = self.get_pdf("manifold_{}_{}".format(axes[0], axes[1]))
        print('Saving to ' + output)

        # preallocate manifold matrix
        manifold = np.zeros(shape=(len(x1)*28, len(x2)*28))
        for i_x, x in enumerate(x1):
            for i_y, y in enumerate(x2):
                input = np.zeros(shape=(1, latent_dim))
                input[0][axes[0]] = x
                input[0][axes[1]] = y
                img = self.v.decode(input)
                manifold[i_x*28:(i_x+1)*28,i_y*28:(i_y+1)*28] = img.reshape(28,28)
        plt.imshow(manifold, cmap=plt.cm.gray_r)
        plt.axis('off')
        plt.savefig(output)
        plt.close()

    def plot_generated(self):
        """Generate a plot with generated samples (from latent space)"""
        x1 = range(0,5)
        x2 = range(0,5)
        output = self.get_pdf("generated_samples")
        print('Saving to ' + output)

        manifold = np.zeros(shape=(len(x1)*28, len(x2)*28))
        for i_x, x in enumerate(x1):
            for i_y, y in enumerate(x2):
                img = self.v.decode()
                manifold[i_x*28:(i_x+1)*28,i_y*28:(i_y+1)*28] = img.reshape(28,28)
        plt.imshow(manifold, cmap=plt.cm.gray_r)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output)
        plt.close()

    def plot_mnist(self):
        x1 = range(0,5)
        x2 = range(0,5)
        output = self.get_pdf("mnist_samples")
        print('Saving to ' + output)

        manifold = np.zeros(shape=(len(x1)*28, len(x2)*28))
        for i_x, x in enumerate(x1):
            for i_y, y in enumerate(x2):
                manifold[i_x*28:(i_x+1)*28,i_y*28:(i_y+1)*28] = choice(mnist.test.images).reshape(28,28)
        plt.imshow(manifold, cmap=plt.cm.gray_r)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output)
        plt.close()


    def plot_reconstructions(self, n = 7):
        """Plot a plot with reconstructions.
        Top row is ground truth, bottom is reconstruction
        Parameter `n` specifies the number of samples
        """
        mnist.test.next_batch(n)
        images = choices(mnist.test.images,k=n)
        output = self.get_pdf("reconstructions")
        print('Saving to ' + output)

        f = plt.figure(figsize=(7, 2))
        for i in range(n):
            reconstructed = self.v.end_to_end(images[i].reshape(1,784))
            
            # display original
            ax = plt.subplot(2, n, i + 1)
            ax.axis('off')
            plt.imshow(images[i].reshape(28, 28), cmap=plt.cm.gray_r)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            ax.axis('off')
            plt.imshow(reconstructed.reshape(28, 28), cmap=plt.cm.gray_r)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.savefig(output)
        plt.close()


if __name__ == '__main__':
    print('Starting MetaTrain')
    num_epochs = 300
    for hidden_dim in [128,256,512,1024]:
        for latent_dim in [1,2,5,10,15,20,30,50,80,100,200]:
            arch = [784, hidden_dim, hidden_dim, latent_dim]
            model = Model(arch)
            loss = model.train(num_epochs)
            model.plot_loss(loss)
            if latent_dim == 2:
                model.plot_manifold()
            model.plot_generated()
            model.plot_mnist()
            model.plot_reconstructions()

