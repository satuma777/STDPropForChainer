#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import time
import sys

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer import iterators

import numpy as np

import stdprop

class MLP(chainer.Chain):

    # Multi-Layer Perceptron for MNIST dataset.
    # This network has 20 hidden layers.
    # Activation function is ReLU.

    def __init__(self, n_in, n_hidden, n_out):
        super(MLP, self).__init__(
            l1 = L.Linear(n_in, n_hidden),
            l2 = L.Linear(n_hidden, n_hidden),
            l3 = L.Linear(n_hidden, n_hidden),
            l4 = L.Linear(n_hidden, n_hidden),
            l5 = L.Linear(n_hidden, n_hidden),
            l6 = L.Linear(n_hidden, n_hidden),
            l7 = L.Linear(n_hidden, n_hidden),
            l8 = L.Linear(n_hidden, n_hidden),
            l9 = L.Linear(n_hidden, n_hidden),
            l10 = L.Linear(n_hidden, n_hidden),
            l11 = L.Linear(n_hidden, n_hidden),
            l12 = L.Linear(n_hidden, n_hidden),
            l13 = L.Linear(n_hidden, n_hidden),
            l14 = L.Linear(n_hidden, n_hidden),
            l15 = L.Linear(n_hidden, n_hidden),
            l16 = L.Linear(n_hidden, n_hidden),
            l17 = L.Linear(n_hidden, n_hidden),
            l18 = L.Linear(n_hidden, n_hidden),
            l19 = L.Linear(n_hidden, n_hidden),
            l20 = L.Linear(n_hidden, n_hidden),
            l21 = L.Linear(n_hidden, n_hidden),
            l22 = L.Linear(n_hidden, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        h5 = F.relu(self.l5(h4))
        h6 = F.relu(self.l6(h5))
        h7 = F.relu(self.l7(h6))
        h8 = F.relu(self.l8(h7))
        h9 = F.relu(self.l9(h8))
        h10 = F.relu(self.l10(h9))
        h11 = F.relu(self.l11(h10))
        h12 = F.relu(self.l12(h11))
        h13 = F.relu(self.l13(h12))
        h14 = F.relu(self.l14(h13))
        h15 = F.relu(self.l15(h14))
        h16 = F.relu(self.l16(h15))
        h17 = F.relu(self.l17(h16))
        h18 = F.relu(self.l18(h17))
        h19 = F.relu(self.l19(h18))
        h20 = F.relu(self.l20(h19))
        h21 = F.relu(self.l21(h20))

        return self.l22(h21)

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=50,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports gaussian nagative log-likelihood loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(28*28, args.unit, 10))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = stdprop.STDProp(alpha=0.001, gamma=0.99, eps=1e-5)
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.GradientNoise(eta=0.01))

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
