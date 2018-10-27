#!/bin/bash

if ["${STATIC_ANALYSIS}" == 'ON']; then
	git clone https://github.com/myint/cppclean
else
	mkdir build && cd build/
	cmake ../ && cmake --build . && ctest --extra-verbose
	
	if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
		cd ../test/unit_tests/cuda
		make
	fi
fi