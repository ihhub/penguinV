from distutils.core import Extension, setup

module = Extension( '_penguinV', 
                    sources = ['penguinV.i', 
                               # Need to include all project (i.e. non-standard) .cpp files that will be required 
                               # to build the functions we will put in interface. Or else you will get
                               # linking errors. 
                               '..\\FileOperation\\bitmap.cpp',
                               '..\\image_function.cpp',
                               '..\\image_function_helper.cpp' 
                              ],
                    swig_opts = ['-c++'], language = 'c++'
                  )
setup(name = 'penguinV',
      ext_modules = [module],
      py_modules = ['penguinV']
     )
