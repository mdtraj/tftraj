import pytest
import tensorflow as tf
import mdtraj as md


@pytest.fixture(scope='session')
def sess():
    sess = tf.Session()
    yield sess
    sess.close()


@pytest.fixture()
def traj():
    t = md.load('fs_peptide/trajectory-9.xtc', top='fs_peptide/fs-peptide.pdb', stride=21)
    return t
