#!/bin/bash

if [[ -z "${STATIC_ANALYSIS+x}"]]; then
	if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
		source script/travis/install_cuda.sh
	fi
fi
