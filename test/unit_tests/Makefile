
PWD := $(shell pwd)
LIB_DIR := $(PWD)/../../src
TEST_DIR := $(PWD)/..
SRCS :=  \
    $(LIB_DIR)/file/bmp_image.cpp \
    $(LIB_DIR)/file/png_image.cpp \
    $(LIB_DIR)/file/jpeg_image.cpp \
    $(LIB_DIR)/math/haar_transform.cpp \
    $(LIB_DIR)/math/hough_transform.cpp \
    $(LIB_DIR)/math/fft_base.cpp \
    $(LIB_DIR)/math/math_base.cpp \
    $(LIB_DIR)/penguinv/penguinv.cpp \
    $(LIB_DIR)/thirdparty/kissfft/kiss_fft.c \
    $(LIB_DIR)/thirdparty/kissfft/kiss_fftnd.c \
    $(LIB_DIR)/blob_detection.cpp \
    $(LIB_DIR)/edge_detection.cpp \
    $(LIB_DIR)/filtering.cpp \
    $(LIB_DIR)/fft.cpp \
    $(LIB_DIR)/function_pool.cpp \
    $(LIB_DIR)/function_pool_task.cpp \
    $(LIB_DIR)/image_function.cpp \
    $(LIB_DIR)/image_function_helper.cpp \
    $(LIB_DIR)/image_function_simd.cpp \
    $(LIB_DIR)/thread_pool.cpp \
    $(TEST_DIR)/test_helper.cpp \
    unit_tests.cpp \
    unit_test_blob_detection.cpp \
    unit_test_edge_detection.cpp \
    unit_test_fft.cpp \
    unit_test_file.cpp \
    unit_test_framework.cpp \
    unit_test_helper.cpp \
    unit_test_image_buffer.cpp \
    unit_test_image_function.cpp \
    unit_test_math.cpp

TARGET := unit_tests

CXX := g++
LINKER := g++
INCDIRS := -I$(PWD) -I$(LIB_DIR)
LIBS := -pthread
FLAGS := -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -Wsign-conversion -Werror -O2 -march=native
CFLAGS := -x c $(FLAGS)
CXXFLAGS := -x c++ -std=c++11 $(FLAGS)
BUILD_DIR=build
BIN := $(BUILD_DIR)/bin
OBJS := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SRCS)))
OBJFILES := $(addprefix $(BIN)/, $(foreach obj, $(OBJS), $(shell basename $(obj))))

all: $(BIN) $(TARGET)

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
    CFLAGS += -D PENGUINV_ENABLED_PNG_SUPPORT
    CXXFLAGS += -D PENGUINV_ENABLED_PNG_SUPPORT
    LIBS += -lpng
else
	PNG_MESSAGE := "\nNote! libpng-dev is missing from your computer,\nso .png images are not supported.\n" $(PNG_MESSAGE_OS)
endif

ifeq ($(JPEG_NOT_EXISTS),0)
    CXXFLAGS += -D PENGUINV_ENABLED_JPEG_SUPPORT
    LIBS += -ljpeg
else
	JPEG_MESSAGE := "Note! libjpeg-dev is missing from your computer,\nso .jpeg images are not supported.\n" $(JPG_MESSAGE_OS)
endif

$(BIN):
	@echo $(PNG_MESSAGE)
	@echo $(JPEG_MESSAGE)
	@mkdir -p $(BIN)

$(TARGET): $(OBJFILES)
	$(LINKER) $^ -o $@ $(LIBS)

$(BIN)/%.o: ./%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(LIB_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(LIB_DIR)/**/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(LIB_DIR)/**/**/%.c
	$(CXX) $(CFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

run: $(TARGET)
	@echo "Running unit tests ..."
	@./$(TARGET)
clean:
	@rm -rf $(BUILD_DIR)
