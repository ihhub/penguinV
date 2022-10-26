##
# Flags for C++ compiler
##
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

ifeq ($(OS),Windows_NT) 
    PNG_MESSAGE_OS := ""''""
else
    dOS := $(shell uname -s)
	ifeq ($(dOS),Linux)
		PNG_MESSAGE_OS := "\nYou can install libpng-dev by:\n$ sudo apt install libpng-dev\n"
		JPG_MESSAGE_OS := "\nYou can install libjpeg-dev by:\n$ sudo apt install libjpeg-dev\n"
		PNG_NOT_EXISTS = $(shell g++ -lpng 2>&1 > /dev/null | grep 'cannot find' | wc -c)
		JPEG_NOT_EXISTS = $(shell g++ -ljpeg 2>&1 > /dev/null | grep 'cannot find' | wc -c)
	endif
	ifeq ($(dOS),Darwin)
		PNG_MESSAGE_OS := "\nYou can install libpng-dev by:\n$ brew install libpng\n"
		JPG_MESSAGE_OS := "\nYou can install libjpeg-dev by:\n$ brew install libjpeg\n"
		PNG_NOT_EXISTS = $(shell brew ls --version libpng > /dev/null; echo $$?)
		JPEG_NOT_EXISTS = $(shell brew ls --version libjpeg > /dev/null; echo $$?)
	endif
endif

PNG_MESSAGE := ""
JPEG_MESSAGE := ""

ifeq ($(PNG_NOT_EXISTS),0)
	CXXFLAGS += -D PENGUINV_ENABLED_PNG_SUPPORT
	LDFLAGS += -lpng
else
	PNG_MESSAGE := "\nNote! libpng-dev is missing from your computer,\nso .png images are not supported.\n" $(PNG_MESSAGE_OS)
endif

ifeq ($(JPEG_NOT_EXISTS),0)
	CXXFLAGS += -D PENGUINV_ENABLED_JPEG_SUPPORT
	LDFLAGS += -ljpeg
else
	JPEG_MESSAGE := "Note! libjpeg-dev is missing from your computer,\nso .jpeg images are not supported.\n" $(JPG_MESSAGE_OS)
endif

example_file_operation: $(SRC)
	@echo $(PNG_MESSAGE)
	@echo $(JPEG_MESSAGE)
	g++ $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
