#pragma once

#if defined( __APPLE__ ) || defined( __MACOSX )
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <string>

namespace multiCL
{
    class OpenCLContext;
    class OpenCLKernel;
    class OpenCLProgram;

    bool isOpenCLSupported();
    void enableDeviceSupport( bool enableGPUSupport = true, bool enableCPUSupport = false );
    void getDeviceSupportStatus( bool & isGPUSupportActive, bool & isCPUSupportActive );

    void openCLCheck( cl_int error ); // validates cl_int value and throws an exception if the value is not CL_SUCCESS
    bool openCLSafeCheck( cl_int error ); // validates cl_int and returns true if the error is CL_SUCCESS

    OpenCLProgram CreateProgramFromFile( const std::string & fileName );
    OpenCLProgram CreateProgramFromFile( const std::string & fileName, const OpenCLContext & context );

    // Structure which holds a list of parameters required for kernel launch
    struct KernelParameters
    {
        KernelParameters();
        KernelParameters( size_t sizeX, size_t threadsPerX ); // 1D
        KernelParameters( size_t sizeX, size_t sizeY, size_t threadsPerX, size_t threadsPerY ); // 2D
        KernelParameters( size_t sizeX, size_t sizeY, size_t sizeZ, size_t threadsPerX, size_t threadsPerY, size_t threadsPerZ ); // 3D

        cl_uint dimensionCount;
        size_t dimensionSize[3]; // Global work size
        size_t threadsPerBlock[3]; // Local work size
    };

    // Helper function which returns calculated KernelParameters structure for kernel to be executed on current OpenCL device
    KernelParameters getKernelParameters( const OpenCLKernel & kernel, size_t sizeX ); // 1D
    KernelParameters getKernelParameters( const OpenCLKernel & kernel, size_t sizeX, size_t sizeY ); // 2D
    KernelParameters getKernelParameters( const OpenCLKernel & kernel, size_t sizeX, size_t sizeY, size_t sizeZ ); // 3D

    void launchKernel1D( const OpenCLKernel & kernel, size_t sizeX );
    void launchKernel2D( const OpenCLKernel & kernel, size_t sizeX, size_t sizeY );
    void launchKernel3D( const OpenCLKernel & kernel, size_t sizeX, size_t sizeY, size_t sizeZ );

    void readBuffer( cl_mem memory, size_t size, void * data );
    void writeBuffer( cl_mem memory, size_t size, const void * data );
}
