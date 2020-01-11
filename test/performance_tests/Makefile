
PWD := $(shell pwd)
LIB_DIR := $(PWD)/../../src
TEST_DIR := $(PWD)/..
SRCS :=  \
	$(LIB_DIR)/math/math_base.cpp \
	$(LIB_DIR)/blob_detection.cpp \
	$(LIB_DIR)/edge_detection.cpp \
	$(LIB_DIR)/filtering.cpp \
	$(LIB_DIR)/function_pool.cpp \
	$(LIB_DIR)/function_pool_task.cpp \
	$(LIB_DIR)/thread_pool.cpp \
	$(LIB_DIR)/image_function.cpp \
	$(LIB_DIR)/image_function_helper.cpp \
	$(LIB_DIR)/image_function_simd.cpp \
	$(LIB_DIR)/penguinv/penguinv.cpp \
	$(TEST_DIR)/test_helper.cpp \
	performance_tests.cpp \
	performance_test_blob_detection.cpp \
	performance_test_edge_detection.cpp \
	performance_test_filtering.cpp \
	performance_test_framework.cpp \
	performance_test_helper.cpp \
	performance_test_image_function.cpp
TARGET := performance_tests

CXX := g++
LINKER := g++
INCDIRS := -I$(PWD) -I$(LIB_DIR)
LIBS := -pthread
CXXFLAGS := -std=c++11 -Wall -Wextra -Wstrict-aliasing -Wpedantic -Wconversion -Werror -O2  -march=native
BUILD_DIR=build
BIN := $(BUILD_DIR)/bin
OBJS := $(patsubst %.cpp,%.o,$(SRCS))
OBJFILES := $(addprefix $(BIN)/, $(foreach obj, $(OBJS), $(shell basename $(obj))))

all: $(BIN) $(TARGET)

$(BIN):
	@mkdir -p $(BIN)

$(TARGET): $(OBJFILES)
	$(LINKER) $^ -o $@ $(LIBS)

$(BIN)/%.o: ./%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(LIB_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(LIB_DIR)/**/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

$(BIN)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCDIRS) -c $< -o $@

run: $(TARGET)
	@echo "Running performance tests ..."
	@./$(TARGET)
clean:
	@rm -rf $(BUILD_DIR)
