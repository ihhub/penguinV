##
# Flags for C++ compiler
##
CXXFLAGS += -std=c++11 -Wall
LDFLAGS += -pthread

example_thread_pool : ../../Library/image_function.cpp ../../Library/thread_pool.cpp

.PHONY: clean
clean:
	$(RM) example_thread_pool
