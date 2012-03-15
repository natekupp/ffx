import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("core_utils", ["ffx/core_utils.pyx"]),
               Extension("bases", ["ffx/bases.pyx"])]

setup(
    name = "ffx",
    version = "1.3.3",
    author = "Trent McConaghy",
    author_email = "gtrent@gmail.com",
    maintainer = "Nate Kupp",
    maintainer_email ="nathan.kupp@gmail.com",
    description = ("Fast Function Extraction: A fast, scalable, and deterministic symbolic regression tool."),
    license = "See readme",
    keywords = "symbolic regression machine learning",
    url = "https://github.com/natekupp/ffx",
    packages=['ffx', 'ffx/example-datasets', 'tests'],
    scripts=['ffx/bin/runffx'],
    cmdclass = {'build_ext': build_ext},
    include_dirs = [np.get_include()],
    ext_modules = ext_modules,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: Other/Proprietary License",
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.5',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
)
