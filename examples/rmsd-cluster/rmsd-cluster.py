import mdtraj as md
import tensorflow as tf
import tftraj.rmsd_op
import numpy as np
import math
from matplotlib import pyplot as plt


def main(n_clusters=2, cluster_diff_multiplier=1.0):
    rmsd_op = tftraj.rmsd_op.load()
    traj = md.load(['../../fs_peptide/trajectory-{}.xtc'.format(i + 1) for i in range(28)],
                   top='../../fs_peptide/fs-peptide.pdb')
    traj = traj[::10]
    print("The trajectory has {} frames".format(len(traj)))

    target = tf.Variable(tf.truncated_normal((n_clusters, traj.xyz.shape[1], 3), stddev=0.3), name='target')
    msd, rot = rmsd_op.pairwise_msd(traj.xyz, target)
    nearest_cluster = msd * tf.nn.softmax(-msd)
    cluster_dist = tf.reduce_mean(nearest_cluster, axis=(0, 1))
    cluster_diff, _ = rmsd_op.pairwise_msd(target, target)
    assert n_clusters == 2, 'This example only works for n_clusters=2 right now'
    cluster_diff = cluster_diff[0, 1]
    loss = cluster_dist - tf.tanh(cluster_diff*10)

    optimizer = tf.train.AdamOptimizer(5e-3)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    conformations = []
    print("{:>5s}{:>15s}{:>15s}{:>15s}".format("step", "loss", "mindist (A)", "uniqueness (A)"))
    for step in range(1000):
        if step % 10 == 0:
            _loss = sess.run(loss)
            _cluster_diff = math.sqrt(sess.run(cluster_diff)) * 10
            _cluster_dist = math.sqrt(sess.run(cluster_dist)) * 10
            print("{:5d}{:15.5f}{:15.5f}{:15.5f}".format(step, _loss, _cluster_dist, _cluster_diff))
            conformations += [sess.run(target)]
        sess.run(train)

    rmsd = np.sqrt(sess.run(msd))*10
    plt.hexbin(rmsd[:,0], rmsd[:,1], bins='log', mincnt=1, cmap='magma_r')
    plt.xlabel('Distance to centroid 0', fontsize=18)
    plt.ylabel('Distance to centroid 1', fontsize=18)
    plt.tight_layout()
    plt.savefig('hist.png')

    conformations = np.array(conformations)
    assert conformations.shape[1] == n_clusters
    for i in range(n_clusters):
        conf = conformations[:, i]
        new_traj = md.Trajectory(xyz=conf, topology=traj.topology)
        new_traj.save('cluster-{}.nc'.format(i))



if __name__ == '__main__':
    main()
