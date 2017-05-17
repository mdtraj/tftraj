import tensorflow as tf
import tftraj.rmsd_op
import mdtraj as md
import numpy as np
import tftraj.rmsd
import pytest





def test_gradients():
    sess = tf.Session()

    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    rmsd = tftraj.rmsd_op.load()
    inds = [5, 19, 234, 235]
    target = np.array(traj.xyz[inds])
    target = tf.Variable(target)
    prmsd, _ = rmsd.pairwise_msd(traj.xyz, target)
    sess.run(tf.global_variables_initializer())
    result = sess.run(prmsd)
    print(result.shape)

    grad = tf.gradients(prmsd, [target])[0]
    grad_result = sess.run(grad)
    print(grad_result.shape)
    print(grad_result)

def test_gradients_vs_tensorflow():
    sess = tf.Session()

    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    rmsd = tftraj.rmsd_op.load()
    inds = [5, 19, 234, 235]
    target = np.array(traj.xyz[inds])
    target = tf.Variable(target)
    sess.run(tf.global_variables_initializer())

    rmsd_from_op, _ = rmsd.pairwise_msd(tf.constant(traj.xyz), target)
    grad_from_op = tf.gradients(rmsd_from_op, [target])[0]
    grad_from_op = sess.run(grad_from_op)

    rmsd_from_tf = tftraj.rmsd.pairwise_msd(tf.constant(traj.xyz), target)
    grad_from_tf = tf.gradients(rmsd_from_tf, [target])[0]
    grad_from_tf = sess.run(grad_from_tf)

    np.testing.assert_almost_equal(grad_from_op, grad_from_tf, decimal=3)

