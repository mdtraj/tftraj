import timeit

import mdtraj as md
import tensorflow as tf

import tftraj.rmsd_op

results = {}
sess = tf.Session()
traj = md.load(['fs_peptide/trajectory-{}.xtc'.format(i + 1) for i in range(28)], top='fs_peptide/fs-peptide.pdb')
traj = traj[::100]
print(len(traj))
rmsd = tftraj.rmsd_op.load()
prmsd = rmsd.pairwise_msd(traj.xyz, traj.xyz)
results['tf-cpu'] = timeit.timeit('sess.run(prmsd)', number=3, globals=globals())
results['mdtraj'] = timeit.timeit('[md.rmsd(traj, traj, i) ** 2 for i in range(traj.n_frames)]',
                                  number=3, globals=globals())

for k in sorted(results):
    print("{:10s} {:5.2f}".format(k, results[k]))
