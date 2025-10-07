# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("./convolution.pyx"),
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-fopenmp"],  # For GCC/Clang
    extra_link_args=["-fopenmp"],  # For GCC/Clang
)
