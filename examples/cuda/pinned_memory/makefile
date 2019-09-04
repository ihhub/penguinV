PWD := $(shell pwd)
LIB_DIR := $(PWD)/../../../src

SRCSCU := \
	$(LIB_DIR)/cuda/cuda_device.cu \
	$(LIB_DIR)/cuda/cuda_helper.cu
SRCSCPP := pinned_memory.cpp
TARGET := example_cuda_pinned_memory

INCDIRS := \
    -I$(LIB_DIR) \
    -I$(LIB_DIR)/file \
    -I$(LIB_DIR)/cuda \
    -I$(LIB_DIR)/memory

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
