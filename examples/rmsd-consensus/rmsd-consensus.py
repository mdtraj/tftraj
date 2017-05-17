import mdtraj as md
import tensorflow as tf
import tftraj.rmsd_op
import numpy as np
import math


def main():
    rmsd_op = tftraj.rmsd_op.load()
    traj = md.load(['../../fs_peptide/trajectory-{}.xtc'.format(i + 1) for i in range(28)],
                   top='../../fs_peptide/fs-peptide.pdb')
    traj = traj[::10]
    print("The trajectory has {} frames".format(len(traj)))

    target = tf.Variable(tf.truncated_normal((1, traj.xyz.shape[1], 3), stddev=0.3), name='target')
    msd, rot = rmsd_op.pairwise_msd(traj.xyz, target)
    loss = tf.reduce_mean(msd, axis=0)

    optimizer = tf.train.AdamOptimizer(1e-3)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    conformations = []
    print("{:>5s}{:>15s}{:>15s}".format("step", "loss", "rmsd (A)"))
    for step in range(2500):
        if step % 10 == 0:
            _loss = sess.run(loss)[0]
            _rmsd = math.sqrt(_loss) * 10
            print("{:5d}{:15.5f}{:15.5f}".format(step, _loss, _rmsd))
            conformations += [sess.run(target)]
        sess.run(train)

    conformations = np.array(conformations)
    assert conformations.shape[1] == 1
    conformations = conformations[:,0]
    new_traj = md.Trajectory(xyz=conformations, topology=traj.topology)
    new_traj.save('consensus.nc')


if __name__ == '__main__':
    main()
