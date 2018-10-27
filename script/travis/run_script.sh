#!/bin/bash

if [[ -z "${STATIC_ANALYSIS+x}"]]; then
	mkdir build && cd build/
	cmake ../ && cmake --build . && ctest --extra-verbose
	
	if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
		cd ../test/unit_tests/cuda
		make
	fi
else
	git clone https://github.com/myint/cppclean
	./cppclean/cppclean .
	exit 0
fi
