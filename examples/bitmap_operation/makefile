##
# Flags for C++ compiler
##
CXXFLAGS += -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -O2 -march=native

example_bitmap_operation : ../../src/image_function_helper.cpp ../../src/image_function.cpp ../../src/FileOperation/bitmap.cpp ../../src/penguinv/penguinv.cpp

.PHONY: clean
clean:
	$(RM) example_bitmap_operation
