from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("cum_sum_until.pyx"), include_dirs=[numpy.get_include()],zip_safe=False
)