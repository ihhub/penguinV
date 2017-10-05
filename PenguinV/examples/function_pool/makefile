##
# Flags for C++ compiler
##
CXXFLAGS += -std=c++11 -Wall
LDFLAGS += -pthread

example_function_pool : ../../Library/image_function.cpp ../../Library/image_function_avx.cpp ../../Library/image_function_neon.cpp ../../Library/image_function_avx.cpp ../../Library/thread_pool.cpp ../../Library/function_pool.cpp ../../Library/penguinv/penguinv.cpp

.PHONY: clean
clean:
	$(RM) example_function_pool
