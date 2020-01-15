##
# Flags for C++ compiler
##
CXXFLAGS += -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -O2 -march=native

SRC = *.cpp
SRC += ../../src/image_function_helper.cpp
SRC += ../../src/image_function.cpp
SRC += ../../src/filtering.cpp
SRC += ../../src/fft.cpp
SRC += ../../src/thirdparty/kissfft/kiss_fftnd.c
SRC += ../../src/thirdparty/kissfft/kiss_fft.c
SRC += ../../src/file/file_image.cpp
SRC += ../../src/file/bmp_image.cpp
SRC += ../../src/file/jpeg_image.cpp
SRC += ../../src/file/png_image.cpp
SRC += ../../src/math/fft_base.cpp

example_fft:$(SRC)
	@echo $(PNG_MESSAGE)
	@echo $(JPEG_MESSAGE)
	g++ $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	
.PHONY: clean
clean:
	$(RM) example_fft
