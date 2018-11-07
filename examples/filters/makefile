##
# Flags for C++ compiler
##
CXXFLAGS += -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -O2 -march=native

example_filters : ../../src/image_function_helper.cpp ../../src/image_function.cpp ../../src/filtering.cpp ../../src/FileOperation/bitmap.cpp

.PHONY: clean
clean:
	$(RM) example_filters
