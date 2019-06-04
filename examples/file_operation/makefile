##
# Flags for C++ compiler
##
PNG_EXISTS = $(shell ldconfig -p | grep libpng)
CXXFLAGS += -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -O2 -march=native

SRC = *.cpp
SRC += ../../src/image_function_helper.cpp
SRC += ../../src/image_function.cpp
SRC += ../../src/file/file_image.cpp
SRC += ../../src/file/bmp_image.cpp
SRC += ../../src/file/jpeg_image.cpp
SRC += ../../src/file/png_image.cpp

ifneq ($(PNG_EXISTS),)
CXXFLAGS += -D PENGUINV_ENABLED_PNG_SUPPORT
LDFLAGS += -lpng

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

.PHONY: clean
clean:
	$(RM) example_file_operation
