PWD := $(shell pwd)
MAIN_LIB_DIR := $(PWD)/../../../src
UNIT_TEST_DIR := $(PWD)/..
TEST_DIR := $(PWD)/../..
SRCS :=  \
	$(MAIN_LIB_DIR)/opencl/opencl_device.cpp \
	$(MAIN_LIB_DIR)/opencl/opencl_helper.cpp \
	$(MAIN_LIB_DIR)/image_function.cpp \
	$(MAIN_LIB_DIR)/image_function_helper.cpp \
	$(MAIN_LIB_DIR)/opencl/image_function_opencl.cpp \
	$(TEST_DIR)/test_helper.cpp \
	$(UNIT_TEST_DIR)/unit_test_helper.cpp \
	$(UNIT_TEST_DIR)/unit_test_framework.cpp \
	unit_test_helper_opencl.cpp \
	unit_test_image_function_opencl.cpp \
	unit_tests_opencl.cpp
TARGET := unit_tests_opencl

CXX := g++
LINKER := g++
INCDIRS := -I$(PWD) -I$(UNIT_TEST_DIR) -I$(MAIN_LIB_DIR)
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

$(BIN)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(UNIT_TEST_DIR)/%.cpp
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
