"""
Add C extensions by Cython module (.pyx)

"""
from setuptools import setup

from Cython.Build import cythonize

setup(
    # from Dane-Hillard's <<Publishing-python-packages>>
    ext_modules=cythonize("gamgin/cythonpkg/harmonic_mean.pyx"),
)
