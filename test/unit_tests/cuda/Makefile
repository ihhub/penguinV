PWD := $(shell pwd)
LIB_DIR := $(PWD)/../../../src
UNIT_TEST_DIR := $(PWD)/..
TEST_DIR := $(PWD)/../..
SRCSCU :=  \
        $(LIB_DIR)/cuda_device.cu \
        $(LIB_DIR)/cuda_helper.cu \
	$(LIB_DIR)/cuda/image_function_cuda.cu \
	unit_test_helper_cuda.cu
SRCSCPP :=  \
	$(LIB_DIR)/image_function.cpp \
	$(LIB_DIR)/image_function_helper.cpp \
	$(UNIT_TEST_DIR)/unit_test_framework.cpp \
	$(UNIT_TEST_DIR)/unit_test_helper.cpp \
	$(TEST_DIR)/test_helper.cpp \
	unit_tests_cuda.cpp \
	unit_test_image_function_cuda.cpp
TARGET := unit_test_cuda

CXX := nvcc
LINKER := nvcc
INCDIRS := -I$(PWD) -I$(LIB_DIR) -I$(UNIT_TEST_DIR) -I$(TEST_DIR)
CXXFLAGS := -std=c++11 -O2 -Wno-deprecated-gpu-targets
BUILD_DIR=build
BIN := $(BUILD_DIR)/bin
OBJSCU := $(patsubst %.cu,%.o,$(SRCSCU))
OBJSCPP := $(patsubst %.cpp,%.o,$(SRCSCPP))
OBJFILES := $(addprefix $(BIN)/, $(foreach obj, $(OBJSCU), $(shell basename $(obj))))
OBJFILES += $(addprefix $(BIN)/, $(foreach obj, $(OBJSCPP), $(shell basename $(obj))))

all: $(BIN) $(TARGET)

$(BIN):
	@mkdir -p $(BIN)

$(TARGET): $(OBJFILES)
	$(LINKER) $^ -o $@

$(BIN)/%.o: $(LIB_DIR)/%.cu
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(LIB_DIR)/**/%.cu
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(UNIT_TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@
	
$(BIN)/%.o: $(UNIT_TEST_DIR)/**/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: ./%.cu
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: ./**/%.cu
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(LIB_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(LIB_DIR)/**/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: ./%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: ./**/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

run: $(TARGET)
	@echo "Running CUDA unit tests ..."
	@./$(TARGET)
clean:
	@rm -rf $(BUILD_DIR)
