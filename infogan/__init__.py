import argparse
import time

import progressbar
import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

from tensorflow.examples.tutorials.mnist import input_data

def leaky_rectify(x, leakiness=0.01):
    assert leakiness <= 1
    ret = tf.maximum(x, leakiness * x)
    return ret

def load_dataset():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    pixel_height = 28
    pixel_width = 28
    n_channels = 1
    for dset in [mnist.train, mnist.validation, mnist.test]:
        num_images = len(dset.images)
        dset.images.shape = (num_images, pixel_height, pixel_width, n_channels)
    return mnist


def create_progress_bar(message):
    widgets = [
        message,
        progressbar.Counter(),
        ' ',
        progressbar.Percentage(),
        ' ',
        progressbar.Bar(),
        progressbar.AdaptiveETA()
    ]
    pbar = progressbar.ProgressBar(widgets=widgets)
    return pbar

def generator_forward(z, reuse=None, name="generator"):
    with tf.variable_scope(name, reuse=reuse):
        z_shape = tf.shape(z)
        out = layers.fully_connected(
            z,
            num_outputs=1568,
            activation_fn=leaky_rectify
        )
        out = tf.reshape(
            out,
            tf.pack([
                z_shape[0], 7, 7, 32
            ])
        )
        out = layers.convolution2d_transpose(
            out,
            num_outputs=64,
            kernel_size=4,
            stride=2,
            activation_fn=leaky_rectify
        )
        out = layers.convolution2d_transpose(
            out,
            num_outputs=32,
            kernel_size=4,
            stride=2,
            activation_fn=leaky_rectify
        )
        out = layers.convolution2d(
            out,
            num_outputs=1,
            kernel_size=1,
            stride=1,
            activation_fn=tf.nn.sigmoid
        )
    return out

def discriminator_forward(img, reuse=None, name="discriminator"):
    with tf.variable_scope(name, reuse=reuse):
        # size is 28, 28, 8
        out = layers.convolution2d(
            img,
            num_outputs=8,
            kernel_size=2,
            stride=1
        )
        # size is 14, 14, 16
        out = layers.convolution2d(
            out,
            num_outputs=16,
            kernel_size=4,
            stride=2
        )
        # size is 7, 7, 32
        out = layers.convolution2d(
            out,
            num_outputs=32,
            kernel_size=4,
            stride=2
        )
        # size is 1568
        out = layers.flatten(out)
        prob = layers.fully_connected(
            out,
            num_outputs=1,
            activation_fn=tf.nn.sigmoid
        )
    return prob


def parse_args():
    paarser = ér

def variables_in_current_scope():
    return tf.get_collection(tf.GraphKeys.VARIABLES, scope=tf.get_variable_scope().name)

def scope_variables(name):
    with tf.variable_scope(name):
        return variables_in_current_scope()

def train():
    np.random.seed(1234)
    mnist = load_dataset()
    batch_size = 64
    n_epochs = 100
    z_size = 7 * 7 * 2
    discriminator_lr = tf.get_variable("discriminator_lr", (), initializer=tf.constant_initializer(0.00005))
    generator_lr =  tf.get_variable("generator_lr", (), initializer=tf.constant_initializer(0.001))
    pixel_height = 28
    pixel_width = 28
    n_channels = 1

    discriminator_lr_placeholder = tf.placeholder(tf.float32, ())
    generator_lr_placeholder = tf.placeholder(tf.float32, ())
    assign_discriminator_lr_op = discriminator_lr.assign(discriminator_lr_placeholder)
    assign_generator_lr_op = generator_lr.assign(generator_lr_placeholder)

    X = mnist.train.images
    n_images = len(X)
    idxes = np.arange(n_images, dtype=np.int32)

    ## begin model

    true_images = tf.placeholder(tf.float32, [None, pixel_height, pixel_width, n_channels])
    z_vectors = tf.placeholder(tf.float32, [None, z_size])

    fake_images = generator_forward(z_vectors, name="generator")
    prob_fake = discriminator_forward(fake_images, name="discriminator")
    prob_true = discriminator_forward(true_images, reuse=True, name="discriminator")

    # discriminator should maximize:
    ll_believing_fake_images_are_fake = tf.log(1.0 - prob_fake)
    ll_true_images = tf.log(prob_true)
    discriminator_obj = (
        tf.reduce_mean(ll_believing_fake_images_are_fake) +
        tf.reduce_mean(ll_true_images)
    )

    # generator should maximize:
    ll_believing_fake_images_are_real = tf.reduce_mean(tf.log(prob_fake))

    discriminator_solver = tf.train.AdamOptimizer(learning_rate=discriminator_lr, beta1=0.5)
    generator_solver = tf.train.AdamOptimizer(learning_rate=generator_lr, beta1=0.5)

    discriminator_variables = scope_variables("discriminator")
    generator_variables = scope_variables("generator")

    train_generator = generator_solver.minimize(-ll_believing_fake_images_are_real, var_list=generator_variables)
    train_discriminator = discriminator_solver.minimize(-discriminator_obj, var_list=discriminator_variables)

    tf.image_summary("fake images", fake_images, max_images=4)
    summary_op = tf.merge_all_summaries()
    journalist = tf.train.SummaryWriter("MNIST_v1_log", flush_secs=10)

    iters = 0

    with tf.Session() as sess:
        # pleasure
        sess.run(tf.initialize_all_variables())
        # content
        for epoch in range(n_epochs):
            disc_epoch_obj = 0.0
            gen_epoch_obj = 0.0

            np.random.shuffle(idxes)
            pbar = create_progress_bar("epoch %d >> " % (epoch,))

            for idx in pbar(range(0, n_images, batch_size)):
                batch = X[idxes[idx:idx + batch_size]]
                # train discriminator
                noise = np.random.standard_normal(size=(batch_size, z_size))
                _, disc_obj = sess.run(
                    [train_discriminator, discriminator_obj],
                    feed_dict={true_images:batch, z_vectors:noise}
                )
                disc_epoch_obj += disc_obj

                # train generator
                noise = np.random.standard_normal(size=(batch_size, z_size))
                _, gen_obj = sess.run(
                    [train_generator, ll_believing_fake_images_are_real],
                    feed_dict={z_vectors:noise}
                )
                gen_epoch_obj += gen_obj

                iters += 1

                if iters % 100 == 0:
                    current_summary = sess.run(summary_op, {z_vectors:noise})
                    journalist.add_summary(current_summary)
                    journalist.flush()

            print("epoch %d >> discriminator LL %.2f (lr=%.6f), generator LL %.2f (lr=%.6f)" % (
                    epoch,
                    disc_epoch_obj / iters, sess.run(discriminator_lr),
                    gen_epoch_obj / iters, sess.run(generator_lr)
                )
            )

            if disc_epoch_obj / iters > np.log(0.7):
                sess.run(
                    assign_discriminator_lr_op,
                    {discriminator_lr_placeholder: sess.run(discriminator_lr) * 0.5}
                )
            elif disc_epoch_obj / iters < np.log(0.4):
                sess.run(
                    assign_discriminator_lr_op,
                    {discriminator_lr_placeholder: sess.run(discriminator_lr) * 2.0}
                )


if __name__ == "__main__":
    train()
