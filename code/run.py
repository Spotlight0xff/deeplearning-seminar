#!/usr/bin/env python3

from Network import VAE
import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
arch = [784, 200, 200, 20]
v = VAE.VAE(arch)
v.train(mnist, num_epochs=20)
