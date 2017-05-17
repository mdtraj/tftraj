import mdtraj as md
import mdtraj.geometry.alignment
import numpy as np

import tftraj.rmsd_op

from Bio.SVDSuperimposer import SVDSuperimposer


def test_rotations_should_be_identity(sess):
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


def test_against_biopython(sess):
    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    traj.center_coordinates()
    rmsd = tftraj.rmsd_op.load()
    inds = [5, 19, 234, 235]
    traj = np.array(traj.xyz)
    target = np.array(traj[inds])
    prmsd, rots = rmsd.pairwise_msd(traj, target)
    result = sess.run(rots)

    sup = SVDSuperimposer()

    def rot(x, y):
        sup.set(x, y)
        sup.run()
        u, tran = sup.get_rotran()
        return u.T

    ref_result = []
    for x in traj:
        for y in target:
            ref_result += [rot(x, y)]
    ref_result = np.array(ref_result)
    ref_result = ref_result.reshape((len(traj), len(target), 3, 3))

    np.testing.assert_almost_equal(result, ref_result, decimal=3)


def test_against_mdtraj_rotations(sess):
    traj = md.load('fip35.500.xtc', top='fip35.pdb')
    traj.center_coordinates()
    traj = np.array(traj.xyz)
    rmsd = tftraj.rmsd_op.load()
    inds = [5, 19, 234, 235]
    target = np.array(traj[inds])
    prmsd, rots = rmsd.pairwise_msd(traj, target)
    result = sess.run(rots)

    def compute_rot(x, y):
        trans, rot = mdtraj.geometry.alignment.compute_translation_and_rotation(x, y)
        return rot

    ref_result = []
    for x in traj:
        for y in target:
            ref_result += [compute_rot(x, y)]
    ref_result = np.array(ref_result)
    ref_result = ref_result.reshape((len(traj), len(target), 3, 3))

    np.testing.assert_almost_equal(result, ref_result, decimal=3)
