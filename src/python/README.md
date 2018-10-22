# How to Create the Python Wrappers

The wrappers for python are generated using SWIG ([website](http://www.swig.org/)), and for ease of use, the wrappers
can also be generated with the help of the `distutils` python module. 
The interface for the wrappers is specified by the file `penguinV.i`. To generate the wrappers, 
you will need to make sure you have the `distutils` python module, which most likely came as a standard module with your version of `python`.

From this directory, run the following command from a terminal (in this directory):

```
python setup.py build_ext --inplace
``` 

This will generate several files in this directory, but most importantly it creates `penguinV.py` which contains the python wrappers, and also a file to
link in to a compiled version of the C++ code, the name of this file will depend on your operating system.

# Creating Documentation for the Python Wrappers

The python wrappers should also create python docstrings containing information as to the types of arguments to be passed into the functions. To create
the documentation, simply run the following from a terminal (in this directory):

``` 
pydoc -w penguinV
```

This will create `penguinV.html`, containing the documentation for the python wrappers.
