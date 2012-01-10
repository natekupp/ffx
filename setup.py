import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "ffx",
    version = "1.3",
    author = "Trent McConaghy",
    author_email = "gtrent@gmail.com",
    description = ("Fast Function Extraction: A fast, scalable, and deterministic symbolic regression tool."),
    license = "See readme",
    keywords = "symbolic regression machine learning",
    url = "http://packages.python.org/an_example_pypi_project",
    packages=['ffx', 'tests'],
    long_description=read('Readme.md'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: Other/Proprietary License"
    ],
)