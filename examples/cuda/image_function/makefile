PWD := $(shell pwd)
LIB_DIR := $(PWD)/../../../src

SRCSCU := \
    $(LIB_DIR)/cuda/image_function_cuda.cu \
    $(LIB_DIR)/cuda/cuda_device.cu \
    $(LIB_DIR)/cuda/cuda_helper.cu

SRCSCPP := \
    $(LIB_DIR)/file/bmp_image.cpp \
    $(LIB_DIR)/image_function.cpp \
    $(LIB_DIR)/image_function_helper.cpp \
    example_image_function.cpp

TARGET := example_cuda_image_function

INCDIRS := \
    -I$(LIB_DIR) \
    -I$(LIB_DIR)/file \
    -I$(LIB_DIR)/cuda

CXX := nvcc
LINKER := nvcc
CXXFLAGS := -std=c++11 -O2 -Wno-deprecated-gpu-targets
BUILD_DIR=build
BIN := $(BUILD_DIR)/bin
OBJSCU := $(patsubst %.cu,%.o,$(SRCSCU))
OBJSCPP := $(patsubst %.cpp,%.o,$(SRCSCPP))
OBJFILES := $(addprefix $(BIN)/, $(foreach obj, $(OBJSCU), $(shell basename $(obj))))
OBJFILES += $(addprefix $(BIN)/, $(foreach obj, $(OBJSCPP), $(shell basename $(obj))))

all: $(BIN) $(TARGET)
	@mv $(TARGET) $(BUILD_DIR)

$(BIN):
	@mkdir -p $(BIN)

$(TARGET): $(OBJFILES)
	$(LINKER) $^ -o $@

$(BIN)/%.o: $(LIB_DIR)/cuda/%.cu
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(SRCSCPP)
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

clean:
	@rm -rf $(BUILD_DIR)
