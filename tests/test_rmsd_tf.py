import mdtraj as md
import numpy as np
import tensorflow as tf

import tftraj.rmsd


def test_against_mdtraj_diff_xy(sess, traj):
    inds = [5, 19, 234]
    target = np.array(traj.xyz[inds])

    frames = tf.constant(traj.xyz)
    target = tf.constant(target)
    prmsd = tftraj.rmsd.pairwise_msd(frames, target)
    result = sess.run(prmsd)
    print(result.shape)

    md_result = [
        md.rmsd(traj, traj, i) ** 2
        for i in inds
    ]
    md_result = np.array(md_result).T

    np.testing.assert_almost_equal(result, md_result, decimal=5)
