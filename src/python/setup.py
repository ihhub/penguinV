from distutils.core import Extension, setup
import os

module = Extension( '_penguinV', 
                    sources = ['penguinV.i',
                               os.path.join('..', 'FileOperation', 'bitmap.cpp'),
                               os.path.join('..', 'image_function.cpp'),
                               os.path.join('..', 'image_function_helper.cpp'),  # Need image_function_helper.cpp
                                                                                 .
                              ],
                    swig_opts = ['-c++'], language = 'c++',
                    extra_compile_args=['-std=c++11']
                  )
setup(name = 'penguinV',
      ext_modules = [module],
      py_modules = ['penguinV']
     )
