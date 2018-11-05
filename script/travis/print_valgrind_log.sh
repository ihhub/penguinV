#!/bin/bash

echo "print_valgrind_log.sh is being executed"
cat "${TRAVIS_BUILD_DIR}/build/Testing/Temporary/MemoryChecker.*.log" > out && cat out
test ! -s out
