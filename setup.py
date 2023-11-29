from Cython.Build import cythonize
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def run(self):
        # defer import
        import numpy as np
        self.include_dirs.append(np.get_include())
        super().run()


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
    include_dirs=["Ckmeans.1d.dp/src"],
    extra_compile_args=["-std=c++11", "-g0"],
)


setup(
    name="ckwrap",
    version="0.1.10",
    description="Python wrapper for Ckmeans.1d.dp, 4.3.4.",
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
    cmdclass={'build_ext': build_ext},
)
