#!/bin/bash

if [[ $QT_BASE ]]; then
    if [ "$TRAVIS_OS_NAME" = "linux" ]; then
        sudo apt-get update -qq;
        sudo apt-get install qt59base -y;
        source /opt/qt59/bin/qt59-env.sh;
        export CMAKE_PREFIX_PATH=/opt/qt59/lib/cmake;
    else
        brew update;
        brew install qt5;
        brew link --force qt5;
        export CMAKE_PREFIX_PATH=$(brew --prefix qt5)/lib/cmake;
    fi
fi
