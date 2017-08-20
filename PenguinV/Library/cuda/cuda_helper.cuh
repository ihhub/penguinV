#pragma once

#include <stdint.h>

struct KernelParameters
{
    KernelParameters( uint32_t threadsPerBlock_, uint32_t blocksPerGrid_ );

    uint32_t threadsPerBlock;
    uint32_t blocksPerGrid;
};

// Helper function which should return proper arguments for CUDA device functions
KernelParameters getKernelParameters( uint32_t size );

// Validation of last occured error in functions on host side
void ValidateLastError();

// Validates cudaError value and throws an except if the value is not cudaSuccess
void cudaCheck( cudaError error );

// Validates cudaError and returns true if the error is cudaSuccess
bool cudaSafeCheck( cudaError error );

bool IsCudaSupported(); // returns true if there is any CUDA device in system
