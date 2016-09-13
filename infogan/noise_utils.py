import numpy as np

from infogan.numpy_utils import make_one_hot


def create_continuous_noise(num_continuous, style_size, size):
    continuous = np.random.uniform(-1.0, 1.0, size=(size, num_continuous))
    style = np.random.standard_normal(size=(size, style_size))
    return np.hstack([continuous, style])


def create_infogan_noise_sample(num_categorical, num_continuous, style_size):
    def sample(batch_size):
        categorical = make_one_hot(
            np.random.randint(0, num_categorical, size=(batch_size,)),
            size=num_categorical
        )
        return np.hstack(
            [
                categorical,
                create_continuous_noise(num_continuous, style_size, size=batch_size)
            ]
        )
    return sample


def create_gan_noise_sample(style_size):
    def sample(batch_size):
        return np.random.standard_normal(size=(batch_size, style_size))
    return sample
