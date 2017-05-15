import tensorflow as tf
import pkg_resources


def load():
    so = pkg_resources.resource_filename(__name__, 'rmsd/librmsd.so')
    mod = tf.load_op_library(so)
    return mod
