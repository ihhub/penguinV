##
# Flags for C++ compiler
##
CXXFLAGS += -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -O2 -march=native
LDFLAGS += -pthread

example_thread_pool : ../../src/image_function_helper.cpp ../../src/image_function.cpp ../../src/thread_pool.cpp

.PHONY: clean
clean:
	$(RM) example_thread_pool
