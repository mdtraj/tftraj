import timeit

import mdtraj as md
import numpy as np
import tensorflow as tf

import tftraj.rmsd_op
import tftraj.rmsd

results = {}
sess = tf.Session()
traj = md.load(['fs_peptide/trajectory-{}.xtc'.format(i + 1) for i in range(28)], top='fs_peptide/fs-peptide.pdb')
traj = traj[::100]
traj_xyz = np.array(traj.xyz)
traj_target = traj[::100]
traj_target_xyz = np.array(traj_target.xyz)
print(len(traj_xyz), len(traj_target_xyz))

rmsd = tftraj.rmsd_op.load()
prmsd, _ = rmsd.pairwise_msd(traj_xyz, traj_target_xyz)

results['tf-cpu'] = timeit.timeit('sess.run(prmsd)', number=30, globals=globals()) / 30
results['mdtraj'] = timeit.timeit('[md.rmsd(traj, traj_target, i) ** 2 for i in range(traj_target.n_frames)]',
                                  number=30, globals=globals()) / 30

tfnative = tftraj.rmsd.pairwise_msd(tf.constant(traj_xyz), tf.constant(traj_target_xyz))
results['tf-native'] = timeit.timeit('sess.run(tfnative)', number=1, globals=globals())

print("{:10s} {:7s}".format("Algo", "time/ms"))
print("{:10s} {:7s}".format('-' * 10, '-' * 7))
for k in sorted(results):
    print("{:10s} {:7.1f}".format(k, 1000 * results[k]))
