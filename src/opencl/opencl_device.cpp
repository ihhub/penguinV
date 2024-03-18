/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2024                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include "opencl_device.h"
#include "../penguinv_exception.h"
#include "opencl_helper.h"
#include <algorithm>
#include <assert.h>
#include <map>
#include <memory>
#include <mutex>

namespace
{
#if ( _MSC_VER && _MSC_VER >= 1400 )
#ifndef thread_local
#define thread_local __declspec( thread )
#endif
#endif

    thread_local uint32_t defaultDeviceId = 0;

    void setDefaultDeviceId( uint32_t deviceId )
    {
        defaultDeviceId = deviceId;
    }

    uint32_t getDefaultDeviceId()
    {
        return defaultDeviceId;
    }

    const std::string memsetCode = R"(
        __kernel void memsetOpenCL( __global uchar * data, uint offset, uint size, uchar value )
        {
            const size_t x = get_global_id(0);

            if( x < size )
                data[offset + x] = value;
        }
        )";

    struct MemsetKernelHolder
    {
        MemsetKernelHolder() = default;

        ~MemsetKernelHolder()
        {
            kernel.reset();
            program.reset();
        }

        std::shared_ptr<multiCL::OpenCLProgram> program;
        std::shared_ptr<multiCL::OpenCLKernel> kernel;
    };

    multiCL::OpenCLKernel & getMemsetKernel()
    {
        static std::map<cl_device_id, MemsetKernelHolder> deviceProgram;
        static std::mutex mapGuard;
        std::lock_guard<std::mutex> lock( mapGuard );

        multiCL::OpenCLDevice & device = multiCL::OpenCLDeviceManager::instance().device();

        auto program = deviceProgram.find( device.deviceId() );
        if ( program != deviceProgram.cend() ) {
            return *( program->second.kernel );
        }

        MemsetKernelHolder holder;
        holder.program = std::shared_ptr<multiCL::OpenCLProgram>( new multiCL::OpenCLProgram( device.context(), memsetCode.data() ) );
        holder.kernel = std::shared_ptr<multiCL::OpenCLKernel>( new multiCL::OpenCLKernel( *( holder.program ), "memsetOpenCL" ) );

        deviceProgram[device.deviceId()] = holder;

        return *( deviceProgram[device.deviceId()].kernel );
    }
}

namespace multiCL
{
    namespace MemoryManager
    {
        MemoryAllocator & memory()
        {
            return OpenCLDeviceManager::instance().device( getDefaultDeviceId() ).allocator();
        }

        MemoryAllocator & memory( const uint32_t deviceId )
        {
            return OpenCLDeviceManager::instance().device( deviceId ).allocator();
        }

        void memorySet( cl_mem data, const void * pattern, size_t patternSize, size_t offset, size_t size )
        {
            if ( patternSize == 1u ) {
                multiCL::OpenCLKernel & kernel = getMemsetKernel();
                kernel.reset();
                const uint8_t value = *( static_cast<const uint8_t *>( pattern ) );
                kernel.setArgument( data, static_cast<uint32_t>( offset ), static_cast<uint32_t>( size ), value );
                multiCL::launchKernel1D( kernel, size );
            }
            else {
#if defined( CL_VERSION_1_2 )
                const cl_int error = clEnqueueFillBuffer( OpenCLDeviceManager::instance().device().queue()(), data, pattern, patternSize, offset, size, 0, NULL, NULL );
                if ( error != CL_SUCCESS )
                    throw penguinVException( "Cannot fill a memory for a device" );
#else
                throw penguinVException( "clEnqueueFillBuffer is not supported in OpenCL with version 1.1 and lower" );
#endif
            }
        }
    }

    OpenCLContext::OpenCLContext( cl_device_id deviceId )
    {
        cl_int error;
        _context = clCreateContext( NULL, 1u, &deviceId, NULL, NULL, &error );
        openCLCheck( error );
    }

    OpenCLContext::~OpenCLContext()
    {
        clReleaseContext( _context );
    }

    OpenCLQueue::OpenCLQueue( const OpenCLContext & context, cl_device_id deviceId )
    {
        cl_int error;

#if defined( __APPLE__ ) || defined( __MACOSX )
        _commandQueue = clCreateCommandQueue( context(), deviceId, 0, &error );
#else
        _commandQueue = clCreateCommandQueueWithProperties( context(), deviceId, 0, &error );
#endif

        openCLCheck( error );
    }

    OpenCLQueue::~OpenCLQueue()
    {
        clReleaseCommandQueue( _commandQueue );
    }

    void OpenCLQueue::synchronize()
    {
        openCLCheck( clFinish( _commandQueue ) );
    }

    OpenCLProgram::OpenCLProgram( const OpenCLContext & context, const char * program )
        : _program( NULL )
    {
        cl_int error;
        _program = clCreateProgramWithSource( context(), 1, &program, NULL, &error );
        openCLCheck( error );

        if ( !openCLSafeCheck( clBuildProgram( _program, 0, NULL, NULL, NULL, NULL ) ) ) {
            cl_uint deviceCount = 0u;
            openCLCheck( clGetContextInfo( context(), CL_CONTEXT_NUM_DEVICES, sizeof( cl_uint ), &deviceCount, NULL ) );
            assert( deviceCount > 0u );

            std::vector<cl_device_id> device( deviceCount );
            openCLCheck( clGetContextInfo( context(), CL_CONTEXT_DEVICES, sizeof( cl_device_id ) * deviceCount, device.data(), NULL ) );

            std::string fullLog;
            for ( auto deviceId = device.begin(); deviceId != device.end(); ++deviceId ) {
                size_t logSize = 0;
                clGetProgramBuildInfo( _program, *deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize );

                // Allocate memory for the log
                std::vector<char> log( logSize );

                // Get the log
                clGetProgramBuildInfo( _program, OpenCLDeviceManager::instance().device().deviceId(), CL_PROGRAM_BUILD_LOG, logSize, log.data(), NULL );

                fullLog += std::string( log.begin(), log.end() );
                fullLog += "\n";
            }

            clReleaseProgram( _program );
            throw penguinVException( ( std::string( "Failed to build a program for OpenCL device with following code :\n" ) + fullLog ).data() );
        }
    }

    OpenCLProgram::~OpenCLProgram()
    {
        clReleaseProgram( _program );
    }

    OpenCLKernel::OpenCLKernel( const OpenCLProgram & program, const std::string & name )
        : _parameterId( 0 )
    {
        cl_int error;
        _kernel = clCreateKernel( program(), name.data(), &error );
        openCLCheck( error );
    }

    OpenCLKernel::~OpenCLKernel()
    {
        clReleaseKernel( _kernel );
    }

    void OpenCLKernel::_setArgument( size_t size, const void * data )
    {
        openCLCheck( clSetKernelArg( _kernel, _parameterId, size, data ) );
        ++_parameterId;
    }

    OpenCLDevice::OpenCLDevice( cl_device_id deviceId )
        : _deviceId( deviceId )
        , _context( _deviceId )
        , _currentQueueId( 0u )
    {
        _allocator = new MemoryAllocator( _context(), _deviceId, static_cast<size_t>( totalMemorySize() ) );

        setQueueCount( 1u );
    }

    OpenCLDevice::~OpenCLDevice()
    {
        delete _allocator;

        for ( auto queueId = _queue.begin(); queueId != _queue.end(); ++queueId ) {
            delete ( *queueId );
        }
    }

    size_t OpenCLDevice::threadsPerBlock( const OpenCLKernel & kernel ) const
    {
        uint64_t threadsPerBlock = 0u;
        openCLCheck( clGetKernelWorkGroupInfo( kernel(), _deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof( threadsPerBlock ), &threadsPerBlock, NULL ) );

        assert( threadsPerBlock > 0 );

        return static_cast<size_t>( threadsPerBlock );
    }

    uint64_t OpenCLDevice::totalMemorySize() const
    {
        cl_ulong totalMemorySize = 0u;
        openCLCheck( clGetDeviceInfo( _deviceId, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( totalMemorySize ), &totalMemorySize, NULL ) );

        return totalMemorySize;
    }

    std::string OpenCLDevice::name() const
    {
        size_t nameLength = 0u;

        openCLCheck( clGetDeviceInfo( _deviceId, CL_DEVICE_NAME, 0, NULL, &nameLength ) );

        std::vector<char> deviceName;
        deviceName.resize( nameLength );

        openCLCheck( clGetDeviceInfo( _deviceId, CL_DEVICE_NAME, nameLength, deviceName.data(), NULL ) );

        return deviceName.data();
    }

    std::string OpenCLDevice::computeCapability() const
    {
        size_t capabilityLength = 0u;

        openCLCheck( clGetDeviceInfo( _deviceId, CL_DEVICE_VERSION, 0, NULL, &capabilityLength ) );

        std::vector<char> capability;
        capability.resize( capabilityLength );

        openCLCheck( clGetDeviceInfo( _deviceId, CL_DEVICE_VERSION, capabilityLength, capability.data(), NULL ) );

        return capability.data();
    }

    void OpenCLDevice::synchronize()
    {
        for ( auto queueId = _queue.begin(); queueId != _queue.end(); ++queueId ) {
            ( *queueId )->synchronize();
        }
    }

    void OpenCLDevice::setQueueCount( size_t queueCount )
    {
        // No real device needs more than 255 queues.
        if ( queueCount > 255u ) {
            queueCount = 255u;
        }

        if ( queueCount != _queue.size() ) {
            if ( queueCount > _queue.size() ) {
                while ( queueCount != _queue.size() ) {
                    _queue.push_back( new OpenCLQueue( _context, _deviceId ) );
                }
            }
            else {
                if ( _currentQueueId >= queueCount ) {
                    _currentQueueId = 0;
                }

                for ( auto queueId = _queue.begin() + static_cast<uint8_t>( queueCount ); queueId != _queue.end(); ++queueId ) {
                    delete ( *queueId );
                }

                _queue.resize( queueCount );
            }
        }
    }

    OpenCLDeviceManager & OpenCLDeviceManager::instance()
    {
        static OpenCLDeviceManager manager;
        return manager;
    }

    void OpenCLDeviceManager::initializeDevices()
    {
        for ( uint32_t deviceId = 0; deviceId < _supportedDeviceId.size(); ++deviceId ) {
            initializeDevice( deviceId );
        }
    }

    void OpenCLDeviceManager::initializeDevice( uint32_t deviceId )
    {
        if ( deviceId >= _supportedDeviceId.size() ) {
            throw penguinVException( "System does not contain a device with such ID" );
        }

        auto foundDevice
            = std::find_if( _device.begin(), _device.end(), [&]( const OpenCLDevice * device ) { return device->deviceId() == _supportedDeviceId[deviceId]; } );

        if ( foundDevice == _device.end() ) {
            _device.push_back( new OpenCLDevice( _supportedDeviceId[deviceId] ) );
        }
    }

    void OpenCLDeviceManager::closeDevice( uint32_t deviceId )
    {
        if ( deviceId >= _supportedDeviceId.size() ) {
            throw penguinVException( "System does not contain a device with such ID" );
        }

        auto foundDevice
            = std::find_if( _device.begin(), _device.end(), [&]( const OpenCLDevice * device ) { return device->deviceId() == _supportedDeviceId[deviceId]; } );

        if ( foundDevice != _device.end() ) {
            delete ( *foundDevice );
            _device.erase( foundDevice );
        }
    }

    void OpenCLDeviceManager::closeDevices()
    {
        for ( auto device = _device.begin(); device != _device.end(); ++device ) {
            delete ( *device );
        }

        _device.clear();
    }

    OpenCLDevice & OpenCLDeviceManager::device()
    {
        return device( getDefaultDeviceId() );
    }

    const OpenCLDevice & OpenCLDeviceManager::device() const
    {
        return device( getDefaultDeviceId() );
    }

    OpenCLDevice & OpenCLDeviceManager::device( uint32_t deviceId )
    {
        if ( _device.empty() ) {
            throw penguinVException( "Device manager does not contain any devices" );
        }

        auto foundDevice
            = std::find_if( _device.begin(), _device.end(), [&]( const OpenCLDevice * device ) { return device->deviceId() == _supportedDeviceId[deviceId]; } );

        if ( foundDevice == _device.end() ) {
            throw penguinVException( "Device ID is invalid. Please check that you initialize devices!" );
        }

        return *( *foundDevice );
    }

    const OpenCLDevice & OpenCLDeviceManager::device( uint32_t deviceId ) const
    {
        if ( _device.empty() ) {
            throw penguinVException( "Device manager does not contain any devices" );
        }

        auto foundDevice
            = std::find_if( _device.begin(), _device.end(), [&]( const OpenCLDevice * device ) { return device->deviceId() == _supportedDeviceId[deviceId]; } );

        if ( foundDevice == _device.end() ) {
            throw penguinVException( "Device ID is invalid. Please check that you initialize devices!" );
        }

        return *( *foundDevice );
    }

    void OpenCLDeviceManager::setActiveDevice( uint32_t deviceId )
    {
        auto foundDevice
            = std::find_if( _device.begin(), _device.end(), [&]( const OpenCLDevice * device ) { return device->deviceId() == _supportedDeviceId[deviceId]; } );

        if ( foundDevice == _device.end() ) {
            throw penguinVException( "Device ID is invalid. Please check that you initialize devices!" );
        }

        setDefaultDeviceId( deviceId );
    }

    void OpenCLDeviceManager::resetSupportedDevice()
    {
        closeDevices();
        _supportedDeviceId.clear();

        cl_uint platformCount = 0u;
        std::vector<cl_platform_id> platformId;
        if ( openCLSafeCheck( clGetPlatformIDs( 0, NULL, &platformCount ) ) && ( platformCount > 0u ) ) {
            platformId.resize( platformCount );
            if ( !openCLSafeCheck( clGetPlatformIDs( platformCount, platformId.data(), NULL ) ) ) {
                platformId.clear();
            }
        }

        bool isGPUSupportEnabled = false;
        bool isCPUSupportEnabled = false;
        getDeviceSupportStatus( isGPUSupportEnabled, isCPUSupportEnabled );

        const cl_device_type deviceType = ( isGPUSupportEnabled ? CL_DEVICE_TYPE_GPU : 0u ) + ( isCPUSupportEnabled ? CL_DEVICE_TYPE_CPU : 0u );

        uint32_t supportedDeviceCount = 0u;
        for ( auto platform = platformId.begin(); platform != platformId.end(); ++platform ) {
            uint32_t deviceCount = 0u;
            if ( openCLSafeCheck( clGetDeviceIDs( *platform, deviceType, 0, NULL, &deviceCount ) ) ) {
                _supportedDeviceId.resize( supportedDeviceCount + deviceCount );

                if ( !openCLSafeCheck( clGetDeviceIDs( *platform, deviceType, deviceCount, _supportedDeviceId.data() + supportedDeviceCount, NULL ) ) ) {
                    _supportedDeviceId.resize( supportedDeviceCount );
                    continue;
                }

                for ( uint32_t deviceId = 0u; deviceId < deviceCount; ) {
                    cl_int error;
                    cl_context _context = clCreateContext( NULL, 1u, &_supportedDeviceId[supportedDeviceCount + deviceId], NULL, NULL, &error );
                    if ( error != CL_SUCCESS ) {
                        --deviceCount;
                    }
                    else {
                        clReleaseContext( _context );
                        ++deviceId;
                    }
                }

                supportedDeviceCount += deviceCount;
            }
        }
    }
}
