from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy as np

with open("README.md") as fp:
    long_description = fp.read()

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
    extra_compile_args=["-std=c++11", "-g0"],
)


setup(
    name="ckwrap",
    version="0.1.9",
    description="Python wrapper for Ckmeans.1d.dp, 4.3.3.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="djdt",
    license="LGPL",
    url="https://github.com/djdt/ckwrap",
    packages=["ckwrap"],
    package_data={"ckwrap": ["*.pxd"]},
    ext_modules=cythonize(ckwrap),
    install_requires=["numpy", "Cython"],
    tests_require=["pytest", "scipy"],
)
