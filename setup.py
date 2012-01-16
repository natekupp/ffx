import os
from setuptools import setup
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
