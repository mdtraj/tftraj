import mdtraj as md
import numpy as np
import tensorflow as tf

import tftraj.rmsd
import tftraj.rmsd_op


def test_works(sess, traj):
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


def test_vs_tensorflow_target(sess):
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


def test_vs_tensorflow_target_few_atoms(sess):
    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    traj.atom_slice(np.arange(traj.n_atoms)[::10])
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


def test_vs_tensorflow_traj(sess):
    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    rmsd = tftraj.rmsd_op.load()
    inds = [5, 19, 234, 235]
    target = np.array(traj.xyz[inds])
    target = tf.constant(target)
    traj = tf.Variable(traj.xyz)
    sess.run(tf.global_variables_initializer())

    rmsd_from_op, _ = rmsd.pairwise_msd(traj, target)
    grad_from_op = tf.gradients(rmsd_from_op, [traj])[0]
    grad_from_op = sess.run(grad_from_op)

    rmsd_from_tf = tftraj.rmsd.pairwise_msd(traj, target)
    grad_from_tf = tf.gradients(rmsd_from_tf, [traj])[0]
    grad_from_tf = sess.run(grad_from_tf)

    np.testing.assert_almost_equal(grad_from_op, grad_from_tf, decimal=3)


def test_vs_tensorflow_both_input(sess):
    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    rmsd = tftraj.rmsd_op.load()
    inds = [5, 19, 234, 235]
    target = np.array(traj.xyz[inds])
    traj = tf.Variable(traj.xyz)
    target = tf.Variable(target)
    sess.run(tf.global_variables_initializer())

    rmsd_from_op, _ = rmsd.pairwise_msd(traj, target)
    grad_from_op = tf.gradients(rmsd_from_op, [traj, target])
    grad_from_op = sess.run(grad_from_op)

    rmsd_from_tf = tftraj.rmsd.pairwise_msd(traj, target)
    grad_from_tf = tf.gradients(rmsd_from_tf, [traj, target])
    grad_from_tf = sess.run(grad_from_tf)

    assert len(grad_from_tf) == 2
    assert len(grad_from_op) == 2
    for i in range(2):
        print(i)
        np.testing.assert_almost_equal(grad_from_op[i], grad_from_tf[i], decimal=3)
