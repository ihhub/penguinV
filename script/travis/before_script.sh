#!/bin/bash

if [ -z "${STATIC_ANALYSIS+x}" ]; then
	if [[ $TRAVIS_OS_NAME == 'linux' && -z "${QT_BASE}" ]]; then
		source script/travis/install_cuda.sh
	fi
fi

if [ "$QT_BASE" = "59" ]; then 
	sudo add-apt-repository ppa:beineri/opt-qt596-trusty -y;
	sudo apt-get update;
	sudo apt-get install -qq qt59base;
	source /opt/qt59/bin/qt59-env.sh;
fi

if [ "$QT_BASE" = "latest" ]; then
	brew install qt5;
	brew link --force qt5;
	export CMAKE_PREFIX_PATH=$(brew --prefix qt5);
fi