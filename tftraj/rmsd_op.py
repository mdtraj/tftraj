import tensorflow as tf
import pkg_resources
import warnings
import os


def load():
    so = pkg_resources.resource_filename(__name__, 'rmsd/librmsd.Release.so')
    if not os.path.isfile(so):
        so = pkg_resources.resource_filename(__name__, 'rmsd/librmsd.Debug.so')
        if not os.path.isfile(so):
            raise FileNotFoundError("Could not find the RMSD op shared library. "
                                    "Make sure you compile it!")
        warnings.warn("Using debug build of RMSD Op. This will be slow!")
    mod = tf.load_op_library(so)
    return mod
