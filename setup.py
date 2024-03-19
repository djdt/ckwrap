from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

# Extension definition remains similar to your current setup.py
ckwrap = Extension(
    name="_ckwrap",
    sources=[
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
    ],
    language="c++",
    include_dirs=["Ckmeans.1d.dp/src", np.get_include()],
    extra_compile_args=["-std=c++11", "-g0"],
)

setup(
    ext_modules=cythonize(ckwrap),
)
