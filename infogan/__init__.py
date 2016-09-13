import argparse

import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

from .categorical_grid_plots import CategoricalPlotter
from tensorflow.examples.tutorials.mnist import input_data
from infogan.tf_utils import (
    scope_variables, NOOP, leaky_rectify, identity,
    conv_batch_norm, load_mnist_dataset
)
from infogan.misc_utils import (
    next_unused_name,
    add_boolean_cli_arg,
    create_progress_bar
)
from infogan.noise_utils import (
    create_infogan_noise_sample,
    create_gan_noise_sample
)

TINY = 1e-6


def generator_forward(z, image_size, is_training, reuse=None, name="generator", use_batch_norm=True):
    with tf.variable_scope(name, reuse=reuse):
        z_shape = tf.shape(z)

        if use_batch_norm:
            fc_batch_norm = layers.batch_norm
            cnn_batch_norm = conv_batch_norm
        else:
            fc_batch_norm = None
            cnn_batch_norm = None

        out = layers.fully_connected(
            z,
            num_outputs=1024,
            activation_fn=tf.nn.relu,
            normalizer_fn=fc_batch_norm,
            normalizer_params={"is_training": is_training, "updates_collections": None}
        )
        out = layers.fully_connected(
            out,
            num_outputs=(image_size // 4) * (image_size // 4) * 128,
            activation_fn=tf.nn.relu,
            normalizer_fn=fc_batch_norm,
            normalizer_params={"is_training": is_training, "updates_collections": None}
        )
        out = tf.reshape(
            out,
            tf.pack([
                z_shape[0], image_size // 4, image_size // 4, 128
            ])
        )
        out = layers.convolution2d_transpose(
            out,
            num_outputs=64,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.relu,
            normalizer_fn=cnn_batch_norm,
            normalizer_params={"is_training": is_training}
        )
        out = layers.convolution2d_transpose(
            out,
            num_outputs=1,
            kernel_size=4,
            stride=2,
            activation_fn=tf.nn.sigmoid
        )
    return out


def discriminator_forward(img, is_training, reuse=None, name="discriminator", use_batch_norm=True):
    with tf.variable_scope(name, reuse=reuse):

        if use_batch_norm:
            fc_batch_norm = layers.batch_norm
            cnn_batch_norm = conv_batch_norm
        else:
            fc_batch_norm = None
            cnn_batch_norm = None

        out = layers.convolution2d(
            img,
            num_outputs=64,
            kernel_size=4,
            stride=2,
            activation_fn=leaky_rectify,
            scope="my_conv1"
        )
        out = layers.convolution2d(
            out,
            num_outputs=128,
            kernel_size=4,
            stride=2,
            normalizer_params={"is_training": is_training},
            normalizer_fn=cnn_batch_norm,
            activation_fn=leaky_rectify,
            scope="my_conv2"
        )
        out = layers.flatten(out)
        out = layers.fully_connected(
            out,
            num_outputs=1024,
            activation_fn=leaky_rectify,
            normalizer_params={"is_training": is_training, "updates_collections": None},
            normalizer_fn=fc_batch_norm,
            scope="my_fc1"
        )
        prob = layers.fully_connected(
            out,
            num_outputs=1,
            activation_fn=tf.nn.sigmoid,
            scope="my_fc2"
        )
    return {"prob":prob, "hidden":out}


def reconstruct_mutual_info(true_categorical,
                            true_continuous,
                            fix_std,
                            hidden,
                            is_training,
                            reuse=None,
                            name="mutual_info"):
    with tf.variable_scope(name, reuse=reuse):
        out = layers.fully_connected(
            hidden,
            num_outputs=128,
            activation_fn=leaky_rectify,
            normalizer_fn=layers.batch_norm,
            normalizer_params={"is_training":is_training}
        )

        num_categorical = true_categorical.get_shape()[1].value
        num_continuous = true_continuous.get_shape()[1].value

        out = layers.fully_connected(
            out,
            num_outputs=num_categorical + (num_continuous if fix_std else (num_continuous * 2)),
            activation_fn=tf.identity
        )

        # distribution logic
        prob_categorical = tf.nn.softmax(out[:, :num_categorical])
        ll_categorical = tf.reduce_sum(tf.log(prob_categorical + TINY) * true_categorical, reduction_indices=1)

        mean_contig = out[:, num_categorical:num_categorical + num_continuous]

        if fix_std:
            std_contig = tf.ones_like(mean_contig)
        else:
            std_contig = tf.sqrt(tf.exp(out[:, num_categorical + num_continuous:num_categorical + num_continuous * 2]))

        epsilon = (true_continuous - mean_contig) / (std_contig + TINY)
        ll_continuous = tf.reduce_sum(
            - 0.5 * np.log(2 * np.pi) - tf.log(std_contig + TINY) - 0.5 * tf.square(epsilon),
            reduction_indices=1,
        )
        mutual_info_lb = ll_categorical + ll_continuous
    return {
        "mutual_info": tf.reduce_mean(mutual_info_lb),
        "ll_categorical": tf.reduce_mean(ll_categorical),
        "ll_continuous": tf.reduce_mean(ll_continuous),
        "std_contig": tf.reduce_mean(std_contig)
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--generator_lr", type=float, default=1e-3)
    parser.add_argument("--discriminator_lr", type=float, default=2e-4)
    parser.add_argument("--num_categorical", type=int, default=10)
    parser.add_argument("--num_continuous", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--style_size", type=int, default=62)
    parser.add_argument("--plot_every", type=int, default=200,
                        help="How often should plots be made (note: slow + costly).")
    add_boolean_cli_arg("infogan", default=True, help="control whether to train GAN or InfoGAN")
    add_boolean_cli_arg("use_batch_norm", default=True, help="Use batch normalization (convergence++).")
    add_boolean_cli_arg("fix_std", default=True, help="fix the standard deviation of continuous variables to 1.")
    return parser.parse_args()


def train():
    args = parse_args()

    np.random.seed(args.seed)
    mnist = load_mnist_dataset()

    batch_size = args.batch_size
    n_epochs = args.epochs
    use_batch_norm = args.use_batch_norm
    fix_std = args.fix_std
    plot_every = args.plot_every
    use_entropy_obj = args.entropy_penalty
    use_infogan = args.infogan
    style_size = args.style_size
    num_categorical = args.num_categorical
    num_continuous = args.num_continuous

    if use_infogan:
        z_size = style_size + num_categorical + num_continuous
        sample_noise = create_infogan_noise_sample(
            num_categorical,
            num_continuous,
            style_size
        )
    else:
        z_size = style_size
        sample_noise = create_gan_noise_sample(style_size)

    discriminator_lr = tf.get_variable("discriminator_lr", (), initializer=tf.constant_initializer(args.discriminator_lr))
    generator_lr = tf.get_variable("generator_lr", (), initializer=tf.constant_initializer(args.generator_lr))
    pixel_height = 28
    pixel_width = 28
    n_channels = 1

    discriminator_lr_placeholder = tf.placeholder(tf.float32, (), name="discriminator_lr")
    generator_lr_placeholder = tf.placeholder(tf.float32, (), name="generator_lr")
    assign_discriminator_lr_op = discriminator_lr.assign(discriminator_lr_placeholder)
    assign_generator_lr_op = generator_lr.assign(generator_lr_placeholder)

    X = mnist.train.images
    n_images = len(X)
    idxes = np.arange(n_images, dtype=np.int32)

    ## begin model
    true_images = tf.placeholder(
        tf.float32,
        [None, pixel_height, pixel_width, n_channels],
        name="true_images"
    )
    z_vectors = tf.placeholder(
        tf.float32,
        [None, z_size],
        name="z_vectors"
    )
    is_training_discriminator = tf.placeholder(
        tf.bool,
        [],
        name="is_training_discriminator"
    )
    is_training_generator = tf.placeholder(
        tf.bool,
        [],
        name="is_training_generator"
    )

    fake_images = generator_forward(
        z_vectors,
        image_size=pixel_height,
        is_training=is_training_generator,
        name="generator"
    )
    discriminator_fake = discriminator_forward(
        fake_images,
        is_training_discriminator,
        name="discriminator",
        use_batch_norm=use_batch_norm
    )
    prob_fake = discriminator_fake["prob"]
    discriminator_true = discriminator_forward(
        true_images,
        is_training_discriminator,
        reuse=True,
        name="discriminator",
        use_batch_norm=use_batch_norm
    )
    prob_true = discriminator_true["prob"]

    # discriminator should maximize:
    ll_believing_fake_images_are_fake = tf.log(1.0 - prob_fake + TINY)
    ll_true_images = tf.log(prob_true + TINY)
    discriminator_obj = (
        tf.reduce_mean(ll_believing_fake_images_are_fake) +
        tf.reduce_mean(ll_true_images)
    )

    # generator should maximize:
    ll_believing_fake_images_are_real = tf.reduce_mean(tf.log(prob_fake + TINY))
    generator_obj = ll_believing_fake_images_are_real

    discriminator_solver = tf.train.AdamOptimizer(
        learning_rate=discriminator_lr,
        beta1=0.5
    )
    generator_solver = tf.train.AdamOptimizer(
        learning_rate=generator_lr,
        beta1=0.5
    )

    discriminator_variables = scope_variables("discriminator")
    generator_variables = scope_variables("generator")

    if use_infogan:
        q_output = reconstruct_mutual_info(
            z_vectors[:, :num_categorical],
            z_vectors[:, num_categorical:num_categorical+num_continuous],
            fix_std=fix_std,
            use_entropy_obj=use_entropy_obj,
            hidden=discriminator_fake["hidden"],
            is_training=is_training_discriminator,
            name="mutual_info"
        )
        mutual_info_objective = q_output["mutual_info"]
        mutual_info_variables = scope_variables("mutual_info")
        train_mutual_info = generator_solver.minimize(
            -mutual_info_objective,
            var_list=generator_variables + discriminator_variables + mutual_info_variables
        )
        ll_categorical = q_output["ll_categorical"]
        ll_continuous = q_output["ll_continuous"]
        std_contig = q_output["std_contig"]
        entropy = q_output["entropy"]
    else:
        mutual_info_objective = NOOP
        train_mutual_info = NOOP
        ll_categorical = NOOP
        ll_continuous = NOOP
        std_contig = NOOP
        entropy = NOOP

    train_discriminator = discriminator_solver.minimize(-discriminator_obj, var_list=discriminator_variables)
    train_generator = generator_solver.minimize(-generator_obj, var_list=generator_variables)

    discriminator_obj_summary = tf.scalar_summary("discriminator_objective", discriminator_obj)
    generator_obj_summary = tf.scalar_summary("generator_objective", generator_obj)
    if use_infogan:
        mutual_info_obj_summary = tf.scalar_summary("mutual_info_objective", mutual_info_objective)
        ll_categorical_obj_summary = tf.scalar_summary("ll_categorical_objective", ll_categorical)
        ll_continuous_obj_summary = tf.scalar_summary("ll_continuous_objective", ll_continuous)
        std_contig_summary = tf.scalar_summary("std_contig", std_contig)
        generator_obj_summary = tf.merge_summary([
            generator_obj_summary,
            mutual_info_obj_summary,
            ll_categorical_obj_summary,
            ll_continuous_obj_summary,
            std_contig_summary
        ])

    log_dir = next_unused_name("MNIST_v1_log/%s" % ("infogan" if use_infogan else "gan"))
    journalist = tf.train.SummaryWriter(
        log_dir,
        flush_secs=10
    )
    print("Saving tensorboard logs to %r" % (log_dir,))

    img_summaries = {}
    if use_infogan:
        plotter = CategoricalPlotter(
            num_categorical=num_categorical,
            num_continuous=num_continuous,
            style_size=style_size,
            journalist=journalist,
            generate=lambda sess, x: sess.run(
                fake_images,
                {z_vectors:x, is_training_discriminator:False, is_training_generator:False}
            )
        )
    else:
        image_placeholder = None
        plotter = None
        img_summaries["fake_images"] = tf.image_summary("fake images", fake_images, max_images=10)
    image_summary_op = tf.merge_summary(list(img_summaries.values())) if len(img_summaries) else NOOP

    iters = 0
    with tf.Session() as sess:
        # pleasure
        sess.run(tf.initialize_all_variables())
        # content
        for epoch in range(n_epochs):
            disc_epoch_obj = []
            gen_epoch_obj = []
            infogan_epoch_obj = []

            np.random.shuffle(idxes)
            pbar = create_progress_bar("epoch %d >> " % (epoch,))

            for idx in pbar(range(0, n_images, batch_size)):
                batch = X[idxes[idx:idx + batch_size]]
                # train discriminator
                noise = sample_noise(batch_size)
                _, summary_result1, disc_obj, infogan_obj = sess.run(
                    [train_discriminator, discriminator_obj_summary, discriminator_obj, nll_mutual_info],
                    feed_dict={
                        true_images:batch,
                        z_vectors:noise,
                        is_training_discriminator:True,
                        is_training_generator:True
                    }
                )

                disc_epoch_obj.append(disc_obj)

                if use_infogan:
                    infogan_epoch_obj.append(infogan_obj)

                # train generator
                noise = sample_noise(batch_size)
                _, _, summary_result2, gen_obj, infogan_obj = sess.run(
                    [train_generator, train_mutual_info, generator_obj_summary, generator_obj, nll_mutual_info],
                    feed_dict={
                        z_vectors:noise,
                        is_training_discriminator:True,
                        is_training_generator:True
                    }
                )

                journalist.add_summary(summary_result1, iters)
                journalist.add_summary(summary_result2, iters)
                journalist.flush()
                gen_epoch_obj.append(gen_obj)

                if use_infogan:
                    infogan_epoch_obj.append(infogan_obj)

                iters += 1

                if iters % plot_every == 0:
                    if use_infogan:
                        plotter.generate_images(sess, 10)
                    else:
                        noise = sample_noise(batch_size)
                        current_summary = sess.run(
                            image_summary_op,
                            {
                                z_vectors:noise,
                                is_training_discriminator:False,
                                is_training_generator:False
                            }
                        )
                        journalist.add_summary(current_summary, iters)
                    journalist.flush()

            msg = "epoch %d >> discriminator LL %.2f (lr=%.6f), generator LL %.2f (lr=%.6f)" % (
                epoch,
                np.mean(disc_epoch_obj), sess.run(discriminator_lr),
                np.mean(gen_epoch_obj), sess.run(generator_lr)
            )
            if use_infogan:
                msg = msg + ", infogan loss %.2f" % (np.mean(infogan_epoch_obj),)
            print(msg)
