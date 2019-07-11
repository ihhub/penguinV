##
# Flags for C++ compiler
##
PNG_NOT_EXISTS = $(shell g++ -lpng 2>&1 > /dev/null | grep 'cannot find' | wc -c)
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
	@echo
	@echo done

clean:
	$(RM) example_file_operation
ifeq ($(PNG_NOT_EXISTS),0)
CXXFLAGS += -D PENGUINV_ENABLED_PNG_SUPPORT
LDFLAGS += -lpng -lz

dep_error:
	@echo
else
dep_error:
	@echo 
	@echo "Note! libpng-dev is missing from your computer,"
	@echo "so .png images are not supported."
	@echo
	@echo "You can install libpng-dev by:"
	@echo "$ sudo apt install libpng-dev"
	@echo
endif

example_file_operation: $(SRC) | dep_error
	g++ $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

