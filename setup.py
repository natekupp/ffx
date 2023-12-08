from setuptools import find_packages, setup

setup(
    name='ffx',
    version='2.0.2',
    author='Trent McConaghy',
    author_email='gtrent@gmail.com',
    maintainer='Nate Kupp',
    maintainer_email='nathan.kupp@gmail.com',
    description=(
        'Fast Function Extraction: A fast, scalable, and deterministic symbolic regression tool.'
    ),
    license='See LICENSE',
    keywords='symbolic regression machine learning',
    url='https://github.com/natekupp/ffx',
    packages=find_packages(exclude=['ffx_tests']),
    entry_points={'console_scripts': ['ffx = ffx.cli:main']},
    install_requires=['click>=5.0', 'contextlib2>=0.5.4', 'numpy', 'pandas', 'six', 'scikit-learn',],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: Other/Proprietary License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
