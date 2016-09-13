import numpy as np
import tensorflow as tf

from PIL import Image

from infogan.numpy_utils import make_one_hot
from infogan.noise_utils import create_continuous_noise


def create_image_strip(images, zoom=1, gutter=5):
    num_images, image_height, image_width, channels = images.shape

    if channels == 1:
        images = images.reshape(num_images, image_height, image_width)

    # add a gutter between images
    effective_collage_width = num_images * (image_width + gutter) - gutter

    # use white as background
    start_color = (255, 255, 255)

    collage = Image.new('RGB', (effective_collage_width, image_height), start_color)
    offset = 0
    for image_idx in range(num_images):
        to_paste = Image.fromarray(
            (images[image_idx] * 255).astype(np.uint8)
        )
        collage.paste(
            to_paste,
            box=(offset, 0, offset + image_width, image_height)
        )
        offset += image_width + gutter

    if zoom != 1:
        collage = collage.resize(
            (
                int(collage.size[0] * zoom),
                int(collage.size[1] * zoom)
            ),
            Image.NEAREST
        )
    return np.array(collage)


class CategoricalPlotter(object):
    def __init__(self,
                 journalist,
                 num_categorical,
                 num_continuous,
                 style_size,
                 generate,
                 row_size=10,
                 zoom=2.0,
                 gutter=3):
        self._journalist = journalist
        self._gutter = gutter
        self.num_categorical = num_categorical
        self.style_size = style_size
        self.num_continuous = num_continuous
        self._generate = generate
        self._zoom = zoom

        self._placeholders = {}
        self._image_summaries = {}

    def generate_categorical_variations(self, session, row_size):
        images = []
        continuous_noise = create_continuous_noise(
            num_continuous=self.num_continuous,
            style_size=self.style_size,
            size=row_size
        )
        for i in range(self.num_categorical):
            categorical_noise = make_one_hot(np.ones(row_size, dtype=np.int32) * i, size=self.num_categorical)
            zs = np.hstack([categorical_noise, continuous_noise])
            images.append(
                (
                    create_image_strip(
                        self._generate(session, zs),
                        zoom=self._zoom, gutter=self._gutter
                    ),
                    "categorical variable %d" % (i,)
                )
            )
        self._add_image_summary(session, images)

    def _get_placeholder(self, name):
        if name not in self._placeholders:
            self._placeholders[name] = tf.placeholder(tf.uint8, [None, None, 3])
        return self._placeholders[name]

    def _get_image_summary_op(self, names):
        joint_name = "".join(names)
        if joint_name not in self._image_summaries:
            summaries = []
            for name in names:
                image_placeholder = self._get_placeholder(name)
                decoded_image = tf.expand_dims(image_placeholder, 0)
                image_summary_op = tf.image_summary(
                    name,
                    decoded_image, max_images=1
                )
                summaries.append(image_summary_op)
            self._image_summaries[joint_name] = tf.merge_summary(summaries)
        return self._image_summaries[joint_name]

    def _add_image_summary(self, session, images):
        feed_dict = {}
        for image, placeholder_name in images:
            placeholder = self._get_placeholder(placeholder_name)
            feed_dict[placeholder] = image

        summary_op = self._get_image_summary_op(
            [name for _, name in images]
        )
        summary = session.run(
            summary_op, feed_dict=feed_dict
        )

        self._journalist.add_summary(summary)
        self._journalist.flush()

    def generate_continuous_variations(self, session, row_size, variations=3):
        categorical_fixed = make_one_hot(
            np.random.randint(0, self.num_categorical, size=(variations,)),
            size=self.num_categorical
        )
        continuous_fixed = create_continuous_noise(
            num_continuous=self.num_continuous,
            style_size=self.style_size,
            size=variations
        )
        linear_variation = np.linspace(-2.0, 2.0, row_size)
        images = []

        for contig_idx in range(self.num_continuous):
            for var_idx in range(variations):
                zs = np.zeros(
                    (
                        row_size,
                        self.num_continuous + self.style_size + self.num_categorical
                    )
                )
                # apply the fixed noise for the full row:
                zs[:, self.num_categorical:] = continuous_fixed[var_idx, :]
                zs[:, :self.num_categorical] = categorical_fixed[var_idx, :]
                # make this continuous variable vary linearly over the row:
                zs[:, self.num_categorical + contig_idx] = linear_variation

                images.append(
                    (
                        create_image_strip(
                            self._generate(session, zs),
                            zoom=self._zoom, gutter=self._gutter
                        ),
                        "continuous variable %d, variation %d" % (
                            contig_idx,
                            var_idx
                        )
                    )
                )

        self._add_image_summary(
            session, images
        )

    def generate_images(self, session, row_size):
        self.generate_categorical_variations(session, row_size)
        self.generate_continuous_variations(session, row_size, variations=3)

