##
# Flags for C++ compiler
##
PNG_EXISTS = $(shell ldconfig -p | grep libpng)
CXXFLAGS += -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -O2 -march=native
CXXFLAGS += -D PENGUINV_ENABLED_PNG_SUPPORT

LDFLAGS += -lpng

SRC = *.cpp
SRC += ../../src/image_function_helper.cpp
SRC += ../../src/image_function.cpp
SRC += ../../src/file/file_image.cpp
SRC += ../../src/file/bmp_image.cpp
SRC += ../../src/file/png_image.cpp
SRC += ../../src/file/jpeg_image.cpp

ifneq ($(PNG_EXISTS),)
example_file_operation: $(SRC)
	g++ $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
dep_error:
	@echo "You shouldn't see this message - you have necessary libraries."
else
example_file_operation: 
	@echo
	@echo "libpng-dev is missing on your computer. Please install it."
	@echo
	@echo "$ sudo apt install libpng-dev"
	@echo
endif

.PHONY: clean
clean:
	$(RM) example_file_operation