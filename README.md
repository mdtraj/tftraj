TFTraj
======

Molecular dynamics analysis routines implemented in Tensorflow.

See this [blog post](https://medium.com/@pandelab/tensorflow-rmsd-using-tensorflow-for-things-it-was-not-designed-to-do-ada4c9aa0ea2) for
motivations and an overview of the RMSD computation.

## RMSD

RMSD is implemented via composition of existing Tensorflow Ops in `tftraj.rmsd`. 

```python
traj = md.load('trajectory.xtc', top='topology.pdb')
inds = [5, 19, 234]
target = np.array(traj.xyz[inds])

frames = tf.constant(traj.xyz)
target = tf.constant(target)
prmsd = tftraj.rmsd.pairwise_msd(frames, target)
result = sess.run(prmsd)
```

This might be somewhat slow, so there is a native CPU operation that is very fast. 

```python
traj = md.load('trajectory.xtc', top='topology.pdb')
rmsd = tftraj.rmsd_op.load()
prmsd, _ = rmsd.pairwise_msd(traj.xyz, traj.xyz)
result = sess.run(prmsd)
```

### Benchmarks

The benchmark consists of running a pairwise RMSD calculation among 
[fs peptide](https://figshare.com/articles/Fs_MD_Trajectories/1030363)
trajectories. Specifically, between 2800 (stride = 100) frames and 28 targets
(stride = 100 * 100).


Algorithm | Time / ms
----------|----------
mdtraj    |   33.3
tf-cpu    |    1.6
tf-native | 22843.6

The code between the `tf-cpu` custom op and `mdtraj` is largely the same, although
there is much less Python overhead in the `tf` case because all of our looping
is done in c++ (with openmp parallelization) and we deal with the `xyz` numpy
arrays directly instead of the convenience `md.Trajectory` objects.

## Building

The custom op requires a working c++ compiler and CMake. 
Running `python setup.py develop` should automatically invoke CMake. 
It will use `cmake-build-release/` as a scratch (build) directory. To change
CMake options, change into that directory and use `cmake ..` or `ccmake ..`.
The final shared library is copied back into the source tree, much to CMake's
chagrin. This is so we can package it up in python. The `rmsd_op.py` file
is in charge of finding the shared library and calling the Tensorflow function
to load it.

The build script needs to know where the tensorflow headers are installed. It uses

```bash
python -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
```

automatically. Make sure `python` is the right one when you run CMake.

The package requires compiler and cpu support for SSE instructions and OpenMP. 

### Running tests

I've tested this package on Linux and Python 3.5. Other configurations
may work.

The tests require the following molecular dyanmics dataset:

https://figshare.com/articles/Fs_MD_Trajectories/1030363

Download and extract the `fs_peptide` directory, and make
sure it's at the top level of the source directory.

To run the tests, `cd` to the source directory and use:

```bash
pytest
```

You can also use `conda-build` to build and run tests in a pristine environment. Make
sure you have the `conda-forge` channel added to your `.condarc` and run

```bash
conda build conda-recipe/
```

### Examples

There are a couple examples in the `examples/` folder.
