#!/bin/bash

if [[ $QT_BASE ]]; then
    if [ "$TRAVIS_OS_NAME" = "linux" ]; then
        sudo add-apt-repository ppa:beineri/opt-qt596-trusty -y;
        sudo apt-get update -qq;
        sudo apt-get install -qq qt59base;
        source /opt/qt59/bin/qt59-env.sh;
        export CMAKE_PREFIX_PATH=/opt/qt59/lib/cmake;
    else
        brew update;
        cd $( brew --prefix )
        git checkout 3b920b5 Library/Formula/qt.rb # Homebrew qt 5.9.3
        brew install qt5;
        brew link --force qt5;
        export CMAKE_PREFIX_PATH=$(brew --prefix qt5)/lib/cmake;
    fi
fi
