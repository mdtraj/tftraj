import tensorflow as tf
import tftraj.rmsd_op
import mdtraj as md
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
    prmsd = rmsd.pairwise_msd(np.random.randn(11, 12, 3), np.random.randn(11, 13, 3))
    with pytest.raises(tf.errors.InvalidArgumentError):
        sess.run(prmsd)


def test_against_mdtraj(sess):
    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    rmsd = tftraj.rmsd_op.load()
    prmsd = rmsd.pairwise_msd(traj.xyz, traj.xyz)
    result = sess.run(prmsd)
    print(result.shape)

    md_result = [
        md.rmsd(traj, traj, i) ** 2
        for i in range(traj.n_frames)
    ]
    md_result = np.array(md_result)
    np.testing.assert_almost_equal(result, md_result, decimal=5)
