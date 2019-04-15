##
# Flags for C++ compiler
##
CXXFLAGS += -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -O2 -march=native

edge_detection : ../../../src/image_function_helper.cpp ../../../src/image_function.cpp ../../../src/edge_detection.cpp ../../../src/file/bmp_image.cpp ../../../src/ui/ui.cpp ../../../src/ui/x11/x11_ui.cpp -lX11

.PHONY: clean
clean:
	$(RM) edge_detection
