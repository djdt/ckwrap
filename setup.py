from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy as np


sources = [
    "ckwrap/_ckwrap.pyx",
    "Ckmeans.1d.dp/src/dynamic_prog.cpp",
    "Ckmeans.1d.dp/src/EWL2_dynamic_prog.cpp",
    "Ckmeans.1d.dp/src/EWL2_fill_log_linear.cpp",
    "Ckmeans.1d.dp/src/EWL2_fill_quadratic.cpp",
    "Ckmeans.1d.dp/src/EWL2_fill_SMAWK.cpp",
    "Ckmeans.1d.dp/src/fill_log_linear.cpp",
    "Ckmeans.1d.dp/src/fill_quadratic.cpp",
    "Ckmeans.1d.dp/src/fill_SMAWK.cpp",
    "Ckmeans.1d.dp/src/select_levels.cpp",
    "Ckmeans.1d.dp/src/weighted_select_levels.cpp",
]

ckwrap = Extension(
    name="_ckwrap",
    sources=sources,
    language="c++",
    include_dirs=["Ckmeans.1d.dp/src", np.get_include()],
    extra_compile_args=["-std=c++11"],
)

setup(
    name="ckwrap",
    version="0.1.1",
    description="Python wrapper for Ckmeans.1d.dp, 4.3.2.",
    packages=["ckwrap"],
    ext_modules=cythonize(ckwrap),
    license="LGPL",
    author="djdt",
    install_requires=["numpy", "Cython"],
)