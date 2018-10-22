from distutils.core import Extension, setup

module = Extension( '_penguinV', 
                    sources = ['penguinV.i', 
                               '..\\FileOperation\\bitmap.cpp',
                               '..\\image_function.cpp',
                               '..\\image_function_helper.cpp' # Need image_function_helper.cpp 
                                                               # or we get linking errors. 
                              ],
                    swig_opts = ['-c++'], language = 'c++'
                  )
setup(name = 'penguinV',
      ext_modules = [module],
      py_modules = ['penguinV']
     )
