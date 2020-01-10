##
# Flags for C++ compiler
##
PNG_NOT_EXISTS = $(shell g++ -lpng 2>&1 > /dev/null | grep 'cannot find' | wc -c)
JPEG_NOT_EXISTS = $(shell g++ -ljpeg 2>&1 > /dev/null | grep 'cannot find' | wc -c)
CXXFLAGS += -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -O2 -march=native

SRC = *.cpp
SRC += ../../src/image_function_helper.cpp
SRC += ../../src/image_function.cpp
SRC += ../../src/file/file_image.cpp
SRC += ../../src/file/bmp_image.cpp
SRC += ../../src/file/jpeg_image.cpp
SRC += ../../src/file/png_image.cpp

.PHONY: all clean
all: example_file_operation

clean:
	$(RM) example_file_operation

PNG_MESSAGE := ""
JPEG_MESSAGE := ""

ifeq ($(PNG_NOT_EXISTS),0)
CXXFLAGS += -D PENGUINV_ENABLED_PNG_SUPPORT
LDFLAGS += -lpng
else
PNG_MESSAGE := "\nNote! libpng-dev is missing from your computer,\nso .png images are not supported.\n\nYou can install libpng-dev by:\n$ sudo apt install libpng-dev\n"
endif

ifeq ($(JPEG_NOT_EXISTS),0)
CXXFLAGS += -D PENGUINV_ENABLED_JPEG_SUPPORT
LIBS += -ljpeg
else
JPEG_MESSAGE := "\nNote! libjpeg-dev is missing from your computer,\nso .jpeg images are not supported.\n\nYou can install libjpeg-dev by:\n$ sudo apt install libjpeg-dev\n"
endif

example_file_operation: $(SRC)
	@echo $(PNG_MESSAGE)
	@echo $(JPEG_MESSAGE)
	g++ $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
