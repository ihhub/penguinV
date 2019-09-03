##
# Directories containing makefiles
##
subsystem:
	-$(MAKE) -C blob_detection
	-$(MAKE) -C file_operation
	-$(MAKE) -C filters
	-$(MAKE) -C function_pool
	-$(MAKE) -C image_function
	-$(MAKE) -C opencl/image_function
	-$(MAKE) -C cuda/image_function
	-$(MAKE) -C cuda/pinned_memory

.PHONY: clean
clean:
	-$(MAKE) clean -C blob_detection
	-$(MAKE) clean -C file_operation
	-$(MAKE) clean -C filters
	-$(MAKE) clean -C function_pool
	-$(MAKE) clean -C image_function
	-$(MAKE) clean -C opencl/image_function
	-$(MAKE) clean -C cuda/image_function
	-$(MAKE) clean -C cuda/pinned_memory
