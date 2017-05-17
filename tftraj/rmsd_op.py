import tensorflow as tf
import pkg_resources
import warnings
import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops


def load(debug=False):
    so = pkg_resources.resource_filename(__name__, 'rmsd/librmsd.Release.so')
    if debug or not os.path.isfile(so):
        so = pkg_resources.resource_filename(__name__, 'rmsd/librmsd.Debug.so')
        if not os.path.isfile(so):
            raise FileNotFoundError("Could not find the RMSD op shared library. "
                                    "Make sure you compile it!")
        warnings.warn("Using debug build of RMSD Op. This will be slow!")
    mod = tf.load_op_library(so)
    return mod


@ops.RegisterGradient("PairwiseMSD")
def _pairwise_msd_grad(op, grad, rot_grad):
    # TODO: Throw error if rot_grad is non-zero?
    confs1, confs2 = op.inputs
    N1 = int(confs1.get_shape()[0])
    N2 = int(confs2.get_shape()[0])
    n_atom = float(int(confs1.get_shape()[1]))
    confs1 = tf.expand_dims(confs1, axis=1)
    confs2 = tf.expand_dims(confs2, axis=0)

    big_confs1 = tf.tile(confs1, [1, N2, 1, 1])
    big_confs2 = tf.tile(confs2, [N1, 1, 1, 1])

    grad = tf.expand_dims(tf.expand_dims(grad, axis=-1), axis=-1)

    rots = op.outputs[1]
    dxy = grad * (confs1 - tf.matmul(big_confs2, rots))
    sum_dxy = 2 * tf.reduce_sum(dxy, axis=1) / n_atom

    dyx = grad * (confs2 - tf.matmul(big_confs1, rots))
    sum_dyx = 2 * tf.reduce_sum(dyx, axis=0) / n_atom

    return sum_dxy, sum_dyx
