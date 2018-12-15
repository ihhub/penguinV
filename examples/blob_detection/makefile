##
# Flags for C++ compiler
##
CXXFLAGS += -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -O2 -march=native

example_blob_detection : ../../src/math/math_base.cpp ../../src/image_function_helper.cpp ../../src/image_function.cpp ../../src/blob_detection.cpp ../../src/file/bmp_image.cpp

.PHONY: clean
clean:
	$(RM) example_blob_detection
