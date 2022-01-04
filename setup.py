from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy as np

with open("README.md") as fp:
    long_description = fp.read()


ckwrap = Extension(
    name="_ckwrap",
    sources=["ckwrap/_ckwrap.pyx"],
    language="c++",
    include_dirs=["Ckmeans.1d.dp/src", np.get_include()],
    extra_compile_args=["-std=c++11", "-g0"],
)


setup(
    name="ckwrap",
    version="0.1.8",
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
