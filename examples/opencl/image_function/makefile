PWD := $(shell pwd)
MAIN_LIB_DIR := $(PWD)/../../../src
SRCS :=  \
	$(MAIN_LIB_DIR)/opencl/opencl_device.cpp \
	$(MAIN_LIB_DIR)/opencl/opencl_helper.cpp \
	$(MAIN_LIB_DIR)/image_function.cpp \
	$(MAIN_LIB_DIR)/image_function_helper.cpp \
	$(MAIN_LIB_DIR)/file/bmp_image.cpp \
	$(MAIN_LIB_DIR)/opencl/image_function_opencl.cpp \
	main.cpp
TARGET := example

CXX := g++
LINKER := g++
INCDIRS := -I$(PWD) -I$(MAIN_LIB_DIR)
CXXFLAGS := -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -O2
BUILD_DIR=build
BIN := $(BUILD_DIR)/bin
OBJS := $(patsubst %.cpp,%.o,$(SRCS))
OBJFILES := $(addprefix $(BIN)/, $(foreach obj, $(OBJS), $(shell basename $(obj))))

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LINKER += -lOpenCL
endif
ifeq ($(UNAME_S),Darwin)
    LINKER += -framework OpenCL
endif

all: $(BIN) $(TARGET)

$(BIN):
	@mkdir -p $(BIN)

$(TARGET): $(OBJFILES)
	$(LINKER) $^ -o $@

$(BIN)/%.o: ./%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: ./**/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(MAIN_LIB_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(MAIN_LIB_DIR)/**/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

run: $(TARGET)
	@echo "Running example ..."
	@./$(TARGET)
clean:
	@rm -rf $(BUILD_DIR)
