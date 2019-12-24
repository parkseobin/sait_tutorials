""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size


def _visualize_omniglot(inputa, inputb, labela, labelb, filename='output.jpg'):
    '''
        inputa: (32, 5, 784[=28*28])
        labela: (32, 5, 5)
    '''
    plt.rc('font', size=5)
    fig = plt.figure()
    col = 5
    row = 2
    for i in range(1, row*col + 1):
        ax = fig.add_subplot(row, col, i)
        if(i <= 5):
            img = inputa[0, i-1, :].reshape((28, 28))
            ax.set_title(str(labela[0, i-1, :]))
        else:
            img = inputb[0, i-6, :].reshape((28, 28))
            #ax.set_title(str(labelb[0, i-6, :]))
            title_string = '['
            for j in range(5):
                title_string += '{:.2f}, '.format(labelb[0, i-6, j])
            title_string = title_string[:-2] + ']'
            ax.set_title(title_string)
        #ax.set_yticklabels([])
        #ax.set_xticklabels([])
        plt.axis('off')
        plt.imshow(img)

    plt.savefig(filename, bbox_inches='tight')
    print('>> {} saved!'.format(filename))


def visualize_omniglot(inputa, inputb, labela, labelb, num_classes, filename='output.jpg'):
    '''
        inputa: (32, 5, 784[=28*28])
        labela: (32, 5, 5)
    '''
    plt.rc('font', size=5)
    fig = plt.figure()
    #plt.subplots_adjust(wspace=.1, hspace=.1)
    #fig.tight_layout()
    col = 2
    row = num_classes
    for i in range(1, row*col + 1):
        ax = fig.add_subplot(row, col, i)
        if(i % 2 == 1):
            ind = i//2
            img = inputa[0, ind, :].reshape((28, 28))
            ax.set_title(str(labela[0, ind, :]))
        else:
            ind = (i-1)//2
            img = inputb[0, ind, :].reshape((28, 28))
            #ax.set_title(str(labelb[0, i-6, :]))
            title_string = '['
            for j in range(num_classes):
                title_string += '{:.2f}, '.format(labelb[0, ind, j])
            title_string = title_string[:-2] + ']'
            ax.set_title(title_string)
        #ax.set_yticklabels([])
        #ax.set_xticklabels([])
        plt.axis('off')
        plt.imshow(img)
        plt.tight_layout()

    plt.savefig(filename, bbox_inches='tight')
    print('>> {} saved!'.format(filename))


def visualize_sinusoid(amp, phase, outputa, outputb, pointsa, filename='output.jpg'):
    '''
        range of x: -5 ~ 5
        amp, phase: scalars
    '''
    
    plt.rc('font', size=5)
    fig = plt.figure()
    x = np.arange(-5, 5, 0.1)
    sinusoid = amp * np.sin(x - phase)
    # plot ground truth sinusoid
    plt.plot(x, sinusoid, color='C1')
    # plot learned sinusoid
    plt.plot(outputb[0], outputb[1], color='mediumblue')
    plt.plot(outputa[0], outputa[1], linestyle=':', color='mediumblue')
    # plot given points
    pointsa_sin = amp * np.sin(pointsa - phase)
    plt.plot(pointsa, pointsa_sin, '^', color='red')

    plt.savefig(filename, bbox_inches='tight')
    print('>> {} saved!'.format(filename))


