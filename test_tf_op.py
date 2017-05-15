import tensorflow as tf
import tftraj.rmsd_op
import mdtraj as md

rmsd = tftraj.rmsd_op.load()

sess = tf.Session()

import numpy as np
try:
    prmsd_error = rmsd.pairwise_msd(np.random.randn(11,3), np.random.randn(11,3))
except ValueError:
    pass


prmsd_error2 = rmsd.pairwise_msd(np.random.randn(11,12,3), np.random.randn(11,13, 3))
try:
    sess.run(prmsd_error2)
except Exception:
    pass


traj = md.load('fip35.500.xtc', top='fip35.pdb')
prmsd = rmsd.pairwise_msd(traj.xyz, traj.xyz)
result = sess.run(prmsd)
print(result.shape)

md_result = [
    md.rmsd(traj, traj, i)**2
    for i in range(traj.n_frames)
]
md_result = np.array(md_result)
np.testing.assert_almost_equal(result, md_result, decimal=5)
