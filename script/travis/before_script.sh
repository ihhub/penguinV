#!/bin/bash

if [ -z "${STATIC_ANALYSIS+x}" ]; then
	if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
		source script/travis/install_cuda.sh
	fi
fi

if [ "$QT_BASE" = "59" ]; then 
	sudo add-apt-repository ppa:beineri/opt-qt596-trusty -y;
fi
