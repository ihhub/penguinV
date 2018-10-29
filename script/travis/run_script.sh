#!/bin/bash
set -o pipefail

if [ -z "${STATIC_ANALYSIS+x}" ]; then
	mkdir build && cd build/
	cmake -DPENGUINV_BUILD_QT_EXAMPLE=ON ../ && cmake --build . && ctest --extra-verbose

	if [[ $TRAVIS_OS_NAME == 'linux' && -z "${QT_BASE}" ]]; then
		cd ../test/unit_tests/cuda
		make
	fi
else
	git clone https://github.com/myint/cppclean
	./cppclean/cppclean src test examples
	exit 0
fi
