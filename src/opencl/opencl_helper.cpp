#include <assert.h>
#include <fstream>
#include <vector>
#include "opencl_device.h"
#include "opencl_helper.h"
#include "../image_exception.h"

namespace
{
    struct dim3
    {
        size_t x, y, z;
        dim3( size_t vx = 1, size_t vy = 1, size_t vz = 1 ) : x( vx ), y( vy ), z( vz ) {}
    };

    // Helper functions for internal calculations
    dim3 get2DThreadsPerBlock( size_t threadTotalCount )
    {
        size_t threadCount = 1;
        size_t overallThreadCount = (1 << 2);
        while( overallThreadCount <= threadTotalCount ) {
            threadCount <<= 1;
            overallThreadCount <<= 2;
        }

        return dim3( threadTotalCount / threadCount, threadCount );
    }

    dim3 get3DThreadsPerBlock( size_t threadTotalCount )
    {
        size_t threadCount = 1;
        size_t overallThreadCount = (1 << 3);
        while( overallThreadCount <= threadTotalCount ) {
            threadCount <<= 1;
            overallThreadCount <<= 3;
        }

        dim3 threadsPerBlock = get2DThreadsPerBlock( threadTotalCount / threadCount );
        threadsPerBlock.z = threadCount;

        return threadsPerBlock;
    }

    void runKernel( const multiCL::OpenCLKernel & kernel, const multiCL::KernelParameters & parameters )
    {
        cl_event waitingEvent;

        multiCL::openCLCheck( clEnqueueNDRangeKernel( multiCL::OpenCLDeviceManager::instance().device().queue()(), kernel(),
                                                     parameters.dimensionCount, NULL, parameters.dimensionSize, parameters.threadsPerBlock,
                                                     0, NULL, &waitingEvent ) );

        multiCL::openCLCheck( clWaitForEvents( 1, &waitingEvent ) );
    }
}

namespace multiCL
{
    bool isOpenCLSupported()
    {
        cl_uint platformCount = 0u;

        if( !openCLSafeCheck( clGetPlatformIDs( 0, NULL, &platformCount ) ) )
            return false;

        if( platformCount == 0u )
            return false;

        std::vector <cl_platform_id> platform( platformCount );
        if( !openCLSafeCheck( clGetPlatformIDs( platformCount, platform.data(), NULL ) ) )
            return false;

        for( std::vector<cl_platform_id>::const_iterator singlePlatform = platform.begin(); singlePlatform != platform.end(); ++singlePlatform ) {
            cl_uint deviceCount = 0u;

            if( !openCLSafeCheck( clGetDeviceIDs( *singlePlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount ) ) )
                continue;

            if( deviceCount > 0u )
                return true;
        }

        return false;
    }

    void openCLCheck( cl_int error )
    {
        if( error != CL_SUCCESS )
            throw imageException( std::string( "Failed to run OpenCL function with error ") + std::to_string( error ) );
    }

    bool openCLSafeCheck( cl_int error )
    {
        return (error == CL_SUCCESS);
    }

    OpenCLProgram CreateProgramFromFile( const std::string & fileName )
    {
        return CreateProgramFromFile( fileName, OpenCLDeviceManager::instance().device().context() );
    }

    OpenCLProgram CreateProgramFromFile( const std::string & fileName, const OpenCLContext & context )
    {
        std::fstream file;
        file.open( fileName, std::fstream::in | std::fstream::binary );

        if( !file )
            return OpenCLProgram( context, "" );

        file.seekg( 0, file.end );
        const std::streamoff fileLength = file.tellg();

        if( fileLength == std::char_traits<char>::pos_type( -1 ) )
            return OpenCLProgram( context, "" );

        file.seekg( 0, file.beg );

        std::vector <char> fileContent( static_cast<size_t>(fileLength) );

        file.read( fileContent.data(), fileLength );

        file.close();

        return OpenCLProgram( context, fileContent.data() );
    }

    KernelParameters::KernelParameters()
        : dimensionCount( 1 )
    {
        dimensionSize[0]   = dimensionSize[1]   = dimensionSize[2]   = 1u;
        threadsPerBlock[0] = threadsPerBlock[1] = threadsPerBlock[2] = 1u;
    }

    KernelParameters::KernelParameters( size_t sizeX, size_t threadsPerX )
        : dimensionCount( 1 )
    {
        assert( (sizeX >= threadsPerX) && (threadsPerX > 0) && ((sizeX % threadsPerX) == 0) );

        dimensionSize[0]   = sizeX;
        dimensionSize[1]   = dimensionSize[2]   = 1u;
        threadsPerBlock[0] = threadsPerX;
        threadsPerBlock[1] = threadsPerBlock[2] = 1u;
    }

    KernelParameters::KernelParameters( size_t sizeX, size_t sizeY, size_t threadsPerX, size_t threadsPerY )
        : dimensionCount( 2 )
    {
        assert( (sizeX >= threadsPerX) && (sizeY >= threadsPerY) && (threadsPerX > 0) && (threadsPerY > 0) &&
            ((sizeX % threadsPerX) == 0) && ((sizeY % threadsPerY) == 0) );

        dimensionSize[0]   = sizeX;
        dimensionSize[1]   = sizeY;
        dimensionSize[2]   = 1u;
        threadsPerBlock[0] = threadsPerX;
        threadsPerBlock[1] = threadsPerY;
        threadsPerBlock[2] = 1u;
    }

    KernelParameters::KernelParameters( size_t sizeX, size_t sizeY, size_t sizeZ, size_t threadsPerX, size_t threadsPerY, size_t threadsPerZ )
        : dimensionCount( 3 )
    {
        assert( (sizeX >= threadsPerX) && (sizeY >= threadsPerY) && (sizeZ >= threadsPerZ) &&
            (threadsPerX > 0) && (threadsPerY > 0) && (threadsPerZ > 0) &&
                ((sizeX % threadsPerX) == 0) && ((sizeY % threadsPerY) == 0) && ((sizeZ % threadsPerZ) == 0) );

        dimensionSize[0]   = sizeX;
        dimensionSize[1]   = sizeY;
        dimensionSize[2]   = sizeZ;
        threadsPerBlock[0] = threadsPerX;
        threadsPerBlock[1] = threadsPerY;
        threadsPerBlock[2] = threadsPerZ;
    }

    KernelParameters getKernelParameters( const OpenCLKernel & kernel, size_t sizeX )
    {
        const size_t threadTotalCount = OpenCLDeviceManager::instance().device().threadsPerBlock( kernel );

        return KernelParameters( ((sizeX + threadTotalCount - 1) / threadTotalCount) * threadTotalCount, threadTotalCount );
    }

    KernelParameters getKernelParameters( const OpenCLKernel & kernel, size_t sizeX, size_t sizeY )
    {
        const size_t threadTotalCount = OpenCLDeviceManager::instance().device().threadsPerBlock( kernel );
        const dim3 & threadsPerBlock = get2DThreadsPerBlock( threadTotalCount );

        return KernelParameters( ((sizeX + threadsPerBlock.x - 1) / threadsPerBlock.x) * threadsPerBlock.x,
            ((sizeY + threadsPerBlock.y - 1) / threadsPerBlock.y) * threadsPerBlock.y,
                                 threadsPerBlock.x, threadsPerBlock.y );
    }

    KernelParameters getKernelParameters( const OpenCLKernel & kernel, size_t sizeX, size_t sizeY, size_t sizeZ )
    {
        const size_t threadTotalCount = OpenCLDeviceManager::instance().device().threadsPerBlock( kernel );
        const dim3 & threadsPerBlock = get3DThreadsPerBlock( threadTotalCount );

        return KernelParameters( ((sizeX + threadsPerBlock.x - 1) / threadsPerBlock.x) * threadsPerBlock.x,
            ((sizeY + threadsPerBlock.y - 1) / threadsPerBlock.y) * threadsPerBlock.y,
                                 ((sizeZ + threadsPerBlock.z - 1) / threadsPerBlock.z) * threadsPerBlock.z,
                                 threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z );
    }

    void launchKernel1D( const OpenCLKernel& kernel, size_t sizeX )
    {
        runKernel( kernel, getKernelParameters( kernel, sizeX ) );
    }

    void launchKernel2D( const OpenCLKernel & kernel, size_t sizeX, size_t sizeY )
    {
        runKernel( kernel, getKernelParameters( kernel, sizeX, sizeY ) );
    }

    void launchKernel3D( const OpenCLKernel & kernel, size_t sizeX, size_t sizeY, size_t sizeZ )
    {
        runKernel( kernel, getKernelParameters( kernel, sizeX, sizeY, sizeZ ) );
    }

    void readBuffer( cl_mem memory, size_t size, void * data )
    {
        openCLCheck( clEnqueueReadBuffer( OpenCLDeviceManager::instance().device().queue()(), memory, CL_TRUE, 0, size, data, 0, NULL, NULL ) );
    }

    void writeBuffer( cl_mem memory, size_t size, const void * data )
    {
        openCLCheck( clEnqueueWriteBuffer( OpenCLDeviceManager::instance().device().queue()(), memory, CL_TRUE, 0, size, data, 0, NULL, NULL ) );
    }
}
