import multiprocessing
import os
import subprocess

from setuptools import setup, find_packages, Command
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def run(self):
        self.run_command('cmake')
        _build_ext.run(self)


class build_cmake(Command):
    description = 'run CMake to build Tensorflow extensions'
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run command."""
        try:
            os.mkdir('cmake-build-release')
        except OSError:
            pass
        print("Running cmake")
        subprocess.check_call(['cmake', '-DCMAKE_BUILD_TYPE=Release', '..'], cwd='cmake-build-release')
        subprocess.check_call(['make', '-j{}'.format(multiprocessing.cpu_count())], cwd='cmake-build-release')


setup(name='tftraj',
      author='Matthew Harrigan',
      author_email='matthew.harrigan@outlook.com',
      description="Molecular dynamics trajectory utilities in Tensorflow",
      version=1,
      license='MIT',
      url='http:/github.com/mdtraj/tftraj',
      platforms=['Linux'],
      packages=find_packages(),
      cmdclass={'cmake': build_cmake,
                'build_ext': build_ext},
      zip_safe=False,
      package_data={
          'tftraj': ['rmsd/librmsd.Release.so'],
      },
      )
