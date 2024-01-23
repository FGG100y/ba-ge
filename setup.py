"""
Add C extensions by Cython module (.pyx)

Cython is a compiler and language for creating Python C extensions. The Cython
language is a superset of Python and, at its most basic, can be used to speed
up some Python code without requiring sweeping changes. The Cython compiler
converts Cython source code to optimized C code, which will then be compiled
during a package’s build process.

NOTE that `setup.py` file has been a part of the Python packaging ecosystem for
a long time ago (Introduced in PEP-229 in 2000), it even predates `Setuptools`.
(its goal was to centralize where packaging configuration happens) A wide
variety of packages still use it. New build workflow and tools (pyproject.toml,
setup.cfg, etc) are intended to replace setup.py in the long term (so their
co-exists would still last for while).

For pure-Python packages that don't need to detemine any dynamic information at
build time (such as C extensions), one can just use pyproject.toml to define
the build and setup.cfg for configuration when using `Setuptools` as build
backend.


# Credit: Dane-Hillard's <<Publishing-python-packages>>, chap4

#  # outdated code block from book:
#  from Cython.Build import cythonize
#  from setuptools import setup

#  setup(
#      ext_modules=cythonize("src/cythonpkg/harmonic_mean.pyx"),
#  )
"""

from Cython.Build import cythonize
from setuptools import Extension, setup

ext_modules = [
    Extension(
        name="cythonpkg.harmonic_mean",
        sources=["src/cythonpkg/harmonic_mean.pyx"],
    ),
    #  Extension(
    #      name="cythonpkg.mymodule",
    #      sources=["src/cythonpkg/cython_module.pyx"],
    #  )
]

# NOTE with one million inputs, the harmonic_mean calcaulation in Python
# might take some time, it's good time to get help from Cython:
setup(
    name="gamgin",
    ext_modules=cythonize(ext_modules),
)
