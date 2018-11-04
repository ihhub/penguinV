#!/bin/bash

if [[ "${DYNAMIC_ANALYSIS}" == "ON" ]]; then
    cat "${TRAVIS_BUILD_DIR}/build/Testing/Temporary/MemoryChecker.*.log" > out && cat out
    test ! -s out
fi
