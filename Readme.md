#FFX: Fast Function Extraction

FFX is a technique for symbolic regression. It is:

- __Fast__ - runtime 5-60 seconds, depending on problem size (1GHz cpu)
- __Scalable__ - 1000 input variables, no problem!
- __Deterministic__ - no need to "hope and pray".

## Installation
To install from PyPI, simply run:

	pip install ffx

## Usage
FFX can either be run in stand-alone mode, or within your existing Python code. It installs both a binary `runffx` and the Python module `ffx`.

* Standalone: 

	runffx test TRAIN_IN.csv TRAIN_OUT.csv TEST_IN.csv TEST_OUT.csv


* The following snippet demonstrates how to use FFX within your existing Python code. Note that all arguments are expected to be of type `numpy.ndarray` or `pandas.DataFrame`.

	import ffx
	models = ffx.run(train_X, train_y, test_X, test_y, varnames)
	for model in models:
		yhat = model.simulate(X)
		print model

Presently, the FFX Python module only exposes a single API method, `ffx.run()`.


## More Information

#### Technical details

- Circuits-oriented description: [Slides](http://trent.st/content/2011-CICC-FFX-slides.ppt) [Paper](http://trent.st/content/2011-CICC-FFX-paper.pdf) (CICC 2011)
- AI-oriented description [Slides](http://trent.st/content/2011-GPTP-FFX-slides.pdf) [Paper](http://trent.st/content/2011-GPTP-FFX-paper.pdf) (GPTP 2011)

#### Code
* FFX.py (v1.3) - implements FFX algorithm
* runffx.py (v1.3) - toolkit for command-line testing of FFX


#### Dependencies
* python (tested on 2.5, 2.6, and 2.7)
* numpy (1.6.0+)
* scipy (0.9.0+) 
* scikits.learn (0.8+)

### Real-world test datasets:
Datasets are included as .tar.gz files under the `example-datasets/` folder.

* 6 Medium-dim. problems 
	- [All 36K] - `med-dimensional_benchmark_datasets.tar.gz`
* 12 High-dim. problems 
	- [Part1 17M] - `high-dimensional_benchmark_datasets_part1.tar.gz`
	- [Part2 14M] - `high-dimensional_benchmark_datasets_part2.tar.gz` 
	- [Part3 14M] - `high-dimensional_benchmark_datasets_part3.tar.gz`


#### References

1. McConaghy, FFX: Fast, Scalable, Deterministic Symbolic Regression Technology, _Genetic Programming Theory and Practice IX_, Edited by R. Riolo, E. Vladislavleva, and J. Moore, Springer, 2011.
2. McConaghy, High-Dimensional Statistical Modeling and Analysis of Custom Integrated Circuits, _Proc. Custom Integrated Circuits Conference_, Sept. 2011


## License
FFX Software Licence Agreement (like BSD, but adapted for non-commercial gain only)

Copyright (c) 2011, Solido Design Automation Inc.  Authored by Trent McConaghy.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Usage does not involve commercial gain. 
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of the associated institutions nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

For permissions beyond the scope of this license, please contact Trent McConaghy (trentmc@solidodesign.com).

THIS SOFTWARE IS PROVIDED BY THE DEVELOPERS ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE DEVELOPERS OR THEIR INSTITUTIONS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

Patent pending.
