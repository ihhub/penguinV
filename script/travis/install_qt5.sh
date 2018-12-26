#!/bin/bash

if [[ $QT_BASE ]]; then
    if [ "$TRAVIS_OS_NAME" = "linux" ]; then
        sudo add-apt-repository ppa:beineri/opt-qt596-trusty -y;
        sudo apt-get update -qq;
        sudo apt-get install -qq qt59base;
        source /opt/qt59/bin/qt59-env.sh;
        export CMAKE_PREFIX_PATH=/opt/qt59/lib/cmake;
    else
        brew update
        current_dir=$(pwd) # Save current directory
        cd $(brew --repository)/Library/Taps/homebrew/homebrew-core
        git fetch --unshallow
        git checkout 3b920b5 Formula/qt.rb # Homebrew qt 5.9.3
        HOMEBREW_NO_AUTO_UPDATE=1 brew install qt@5.9;
        brew link --force qt@5.9;
        export CMAKE_PREFIX_PATH=$(brew --prefix qt@5.9)/lib/cmake;
        cd ${current_dir}
    fi
fi
