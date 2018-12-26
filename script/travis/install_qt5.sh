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
        cd $(brew --repository)/Library/Taps/homebrew/homebrew-core
        git fetch
        git checkout 3b920b5fed -- Formula/qt.rb # Homebrew qt 5.9.3
        HOMEBREW_NO_AUTO_UPDATE=1 brew install qt5;
        brew link --force qt5;
        export CMAKE_PREFIX_PATH=$(brew --prefix qt5)/lib/cmake;
    fi
fi
