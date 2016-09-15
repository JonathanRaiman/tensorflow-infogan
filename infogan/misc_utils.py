import random

from os.path import exists, join
from os import listdir, walk

from PIL import Image

import numpy as np
import progressbar



OPS =  [
   ('+', lambda a, b: a+b),
   ('-', lambda a, b: a-b),
   ('*', lambda a, b: a*b),
   ('x', lambda a, b: a*b),
   ('/', lambda a, b: a//b),
]


def parse_math(s):
   for operator, f in OPS:
       try:
           idx = s.index(operator)
           return f(parse_math(s[:idx]), parse_math(s[idx+1:]))
       except ValueError:
           pass
   return int(s)

def next_unused_name(name):
    save_name = name
    name_iteration = 0
    while exists(save_name):
        save_name = name + "-" + str(name_iteration)
        name_iteration += 1
    return save_name


def add_boolean_cli_arg(parser, name, default=False, help=None):
    parser.add_argument(
        "--%s" % (name,),
        action="store_true",
        default=default,
        help=help
    )
    parser.add_argument(
        "--no%s" % (name,),
        action="store_false",
        dest=name
    )


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


def find_files_with_extension(path, extensions):
    for basepath, directories, fnames in walk(path):
        for fname in fnames:
            name = fname.lower()
            if any(name.endswith(ext) for ext in extensions):
                yield join(basepath, fname)



def load_image_dataset(path,
                       desired_height=None,
                       desired_width=None,
                       value_range=None,
                       max_images=None,
                       force_grayscale=False):
    image_paths = list(find_files_with_extension(path, [".png", ".jpg", ".jpeg"]))
    limit_msg = ''
    if max_images is not None and len(image_paths) > max_images:
        image_paths = random.sample(image_paths, max_images)
        limit_msg = " (limited to %d images by command line argument)" % (max_images,)

    print("Found %d images in %s%s." % (len(image_paths), path, limit_msg))

    pb = create_progress_bar("Loading dataset ")


    storage = None

    image_idx = 0
    for fname in pb(image_paths):
        image = Image.open(join(path, fname))
        width, height = image.size
        if desired_height is not None and desired_width is not None:
            if width != desired_width or height != desired_height:
                image = image.resize((desired_width, desired_height), Image.BILINEAR)
        else:
            desired_height = height
            desired_width = width

        if force_grayscale:
            image = image.convert("L")

        img = np.array(image)

        if len(img.shape) == 2:
            # extra channel for grayscale images
            img = img[:, :, None]

        if storage is None:
            storage = np.empty((len(image_paths), img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

        storage[image_idx] = img

        image_idx += 1

    if value_range is not None:
        storage = (
            value_range[0] + (storage / 255.0) * (value_range[1] - value_range[0])
        )
    print("dataset loaded.", flush=True)
    return storage

