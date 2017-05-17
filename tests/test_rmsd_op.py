import mdtraj as md
import mdtraj.geometry.alignment
import numpy as np
import pytest
import tensorflow as tf

import tftraj.rmsd_op


def test_load():
    rmsd = tftraj.rmsd_op.load()
    assert rmsd is not None


def test_rank_error():
    rmsd = tftraj.rmsd_op.load()
    with pytest.raises(ValueError):
        rmsd.pairwise_msd(np.random.randn(11, 3), np.random.randn(11, 3))


def test_n_atoms_error(sess):
    rmsd = tftraj.rmsd_op.load()
    prmsd, _ = rmsd.pairwise_msd(np.random.randn(11, 12, 3), np.random.randn(11, 13, 3))
    with pytest.raises(tf.errors.InvalidArgumentError):
        sess.run(prmsd)


def test_against_mdtraj(sess):
    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    rmsd = tftraj.rmsd_op.load()
    prmsd, _ = rmsd.pairwise_msd(traj.xyz, traj.xyz)
    result = sess.run(prmsd)
    print(result.shape)

    md_result = [
        md.rmsd(traj, traj, i) ** 2
        for i in range(traj.n_frames)
    ]
    md_result = np.array(md_result)
    np.testing.assert_almost_equal(result, md_result, decimal=5)


def test_against_mdtraj_diff_xy(sess):
    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    rmsd = tftraj.rmsd_op.load()
    inds = [5, 19, 234]
    target = np.array(traj.xyz[inds])
    prmsd, _ = rmsd.pairwise_msd(traj.xyz, target)
    result = sess.run(prmsd)
    print(result.shape)

    md_result = [
        md.rmsd(traj, traj, i) ** 2
        for i in inds
    ]
    md_result = np.array(md_result).T
    np.testing.assert_almost_equal(result, md_result, decimal=5)




def test_transpose(sess):
    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    rmsd = tftraj.rmsd_op.load()
    inds = [5, 19, 234]
    target = np.array(traj.xyz[inds])
    prmsd1, _ = rmsd.pairwise_msd(traj.xyz, target)
    r1 = sess.run(prmsd1)
    prmsd2, _ = rmsd.pairwise_msd(target, traj.xyz)
    r2 = sess.run(prmsd2)

    np.testing.assert_almost_equal(r1, r2.T, decimal=5)
