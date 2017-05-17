import tensorflow as tf
import tftraj.rmsd_op
import mdtraj as md
import mdtraj.geometry.alignment
import numpy as np
import pytest


@pytest.fixture(scope='module')
def sess():
    sess = tf.Session()
    yield sess
    sess.close()


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


def test_rotations_should_be_zero(sess):
    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    inds = [5, 19, 234, 235]
    rmsd = tftraj.rmsd_op.load()

    for k in range(4):
        traj.superpose(traj, inds[k])
        target = np.array(traj.xyz[inds])
        prmsd, rots = rmsd.pairwise_msd(traj.xyz, target)
        rots_result = sess.run(rots)

        should_be = np.array([np.eye(3, 3) for _ in range(500)])
        np.testing.assert_almost_equal(rots_result[:, k], should_be, decimal=3)
        assert np.all(np.abs(rots_result[:, k] - should_be) < 1e-3)
        assert not np.all(np.abs(rots_result[:, (k + 1) % 4] - should_be) < 1e-3)


def test_against_mdtraj_rotations(sess):
    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    traj.center_coordinates()
    rmsd = tftraj.rmsd_op.load()
    inds = [5, 19, 234, 235]
    target = np.array(traj.xyz[inds])
    prmsd, rots = rmsd.pairwise_msd(traj.xyz, target)
    result = sess.run(rots)

    def compute_rot(x, y):
        trans, rot = mdtraj.geometry.alignment.compute_translation_and_rotation(x, y)
        return rot

    md_result = [[None] * 4] * 500
    for i in range(500):
        for j in range(4):
            md_result[i][j] = compute_rot(traj.xyz[i], target[j])
    md_result = np.array(md_result)

    np.testing.assert_almost_equal(result, md_result, decimal=2)


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
