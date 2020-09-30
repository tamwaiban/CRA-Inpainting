import argparse

import tensorflow as tf
import cv2

from Utils.loader import ImageDataLoader
from Utils.utils import create_generator

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--__gexists__", default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="Load Pre-Existing Generator")
    PARSER.add_argument("--__gpath__", type=str, default="./Models", help="Pre-Existing Generator Filepath")
    PARSER.add_argument("--__flist__", type=str, default="./Data/Images",
                        help="Images Dataset Filepath")
    PARSER.add_argument("--__mlist__", type=str, default="./Data/Masks", help="Mask Dataset Filepath")
    PARSER.add_argument('--__gen_masks__', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Generate Masks on the fly")
    PARSER.add_argument("--__mask_type__", type=str, default="free_form", help="Generate free-form or bbox masks")
    PARSER.add_argument("--__box_count__", type=int, default=10, help="Number of bbox masks")
    PARSER.add_argument("--__activation__", type=str, default="elu", help="Model Activation")
    PARSER.add_argument("--__width__", type=int, default=512, help="Width of inputs")
    PARSER.add_argument("--__height__", type=int, default=512, help="Height of inputs")

    OPT = PARSER.parse_args()
    generator = create_generator(OPT)
    LOADER = ImageDataLoader(OPT)
    for step, input_ in enumerate(LOADER.data):
        r_, g_, b_, mask = tf.split(input_, 4, axis=1)
        img = tf.concat((r_, g_, b_), axis=1)
        first_output, second_output = generator(input_, training=False)
        fusion_fake1 = img * (1 - mask) + first_output * mask
        fusion_fake2 = img * (1 - mask) + second_output * mask
        masked_input = img * (1 - mask) + mask
        img_save = tf.concat((img, masked_input, first_output, second_output, fusion_fake1, fusion_fake2), axis=3)
        img_save = tf.cast((img_save + 1) * 127.5, tf.uint8)
        img_save = tf.transpose(img_save, perm=[0, 2, 3, 1])[0, :, :, :]
        jpeg_string = tf.io.encode_jpeg(img_save)
        tf.io.write_file("Examples/%s_output.jpg" % step, jpeg_string)
        img = cv2.imread("Examples/%s_output.jpg" % step)
        cv2.imshow("Output", img)
        cv2.waitKey(0)
