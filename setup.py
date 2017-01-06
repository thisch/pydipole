from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

__version__ = '0.1.0'

setup(
    author='Thomas Hisch',
    author_email='t.hisch@gmail.com',
    name='dipole',
    packages=find_packages(),
    ext_modules=cythonize("dipole/field.pyx"),
    include_dirs=[numpy.get_include()],
    platforms='Any',
    requires=['python (>=3.4.0)'],
    version=__version__,
)
