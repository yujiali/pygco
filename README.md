## pygco package

This is a python wrapper for [gco-v3.0 package] (http://vision.csd.uwo.ca/code/), which implements a graph cuts based move-making algorithm for optimization in Markov Random Fields.

It contains a copy of the gco-v3.0 package.  Some of the design were borrowed from the [gco_python] (https://github.com/amueller/gco_python) package. However, compared to gco_python:
* This package does not depend on Cython. Instead it is implemented using the ctypes library and a C wrapper of the C++ code.
* This package is an almost complete wrapper for gco-v3.0, which supports more direct low level control over GCoptimization objects.
* This package supports graphs with edges weighted differently.

This wrapper is composed of two parts, a C wrapper and a python wrapper.

To use this package, you should first compile the C wrapper. First compile gco-v3.0 using "make". Then compile test_wrapper using "make test_wrapper". Then run the C test code "./test_wrapper". Now you have the C wrapper ready.

Next test the python wrapper using "python test.py", if it works fine you are ready to use pygco.

To include pygco in your code, simply import pygco module. See the documentation inside code for more details.

