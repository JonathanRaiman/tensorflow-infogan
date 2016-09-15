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
                       value_range=None):
    data = []
    for fname in find_files_with_extension(path, [".png", ".jpg", ".jpeg"]):
        image = Image.open(join(path, fname))
        width, height = image.size
        if desired_height is not None and desired_width is not None:
            if width != desired_width or height != desired_height:
                image = image.resize((desired_width, desired_height), Image.BILINEAR)
        else:
            desired_height = height
            desired_width = width
        data.append(np.array(image)[None])
    concatenated = np.concatenate(data)
    if value_range is not None:
        concatenated = (
            value_range[0] +
            (concatenated / 255.0) * (value_range[1] - value_range[0])
        )
    return concatenated

