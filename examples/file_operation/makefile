##
# Flags for C++ compiler
##
CXXFLAGS += -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -O2 -march=native
CXXFLAGS += -D PENGUINV_ENABLED_PNG_SUPPORT

LDFLAGS += -lpng
LDFLAGS += -ljpeg

SRC = *.cpp
SRC += ../../src/image_function_helper.cpp
SRC += ../../src/image_function.cpp
SRC += ../../src/file/file_image.cpp
SRC += ../../src/file/bmp_image.cpp
SRC += ../../src/file/png_image.cpp
SRC += ../../src/file/jpeg_image.cpp

example_bitmap_operation : $(SRC)
	g++ $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) example_bitmap_operation
