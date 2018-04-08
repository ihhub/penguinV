##
# Flags for C++ compiler
##
CXXFLAGS += -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -O2 -march=native
LDFLAGS += -pthread

example_function_pool : ../../src/image_function_helper.cpp ../../src/image_function.cpp ../../src/image_function_avx.cpp ../../src/image_function_neon.cpp ../../src/image_function_sse.cpp ../../src/thread_pool.cpp ../../src/function_pool.cpp ../../src/penguinv/penguinv.cpp

.PHONY: clean
clean:
	$(RM) example_function_pool
