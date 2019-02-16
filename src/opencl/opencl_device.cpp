#include <algorithm>
#include <assert.h>
#include <map>
#include <memory>
#include <mutex>
#include "opencl_device.h"
#include "opencl_helper.h"
#include "../image_exception.h"

namespace
{
#if (_MSC_VER && _MSC_VER >= 1400)
#ifndef thread_local
#define thread_local __declspec(thread)
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
        MemsetKernelHolder() {}
        ~MemsetKernelHolder()
        {
            kernel.reset();
            program.reset();
        }

        std::shared_ptr< multiCL::OpenCLProgram > program;
        std::shared_ptr< multiCL::OpenCLKernel  > kernel;
    };

    multiCL::OpenCLKernel & getMemsetKernel()
    {
        static std::map< cl_device_id, MemsetKernelHolder > deviceProgram;
        static std::mutex mapGuard;

        multiCL::OpenCLDevice & device = multiCL::OpenCLDeviceManager::instance().device();

        std::map< cl_device_id, MemsetKernelHolder >::const_iterator program = deviceProgram.find( device.deviceId() );
        if ( program != deviceProgram.cend() )
            return *(program->second.kernel);

        mapGuard.lock();

        MemsetKernelHolder holder;
        holder.program = std::shared_ptr< multiCL::OpenCLProgram >( new multiCL::OpenCLProgram( device.context(), memsetCode.data() ) );
        holder.kernel = std::shared_ptr< multiCL::OpenCLKernel >( new multiCL::OpenCLKernel( *(holder.program), "memsetOpenCL" ) );

        deviceProgram[device.deviceId()] = holder;
        mapGuard.unlock();

        return *(deviceProgram[device.deviceId()].kernel);
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

        MemoryAllocator & memory( uint32_t deviceId )
        {
            return OpenCLDeviceManager::instance().device( deviceId ).allocator();
        }

        void memorySet( cl_mem data, const void * pattern, size_t patternSize, size_t offset, size_t size )
        {
            if ( patternSize == 1u ) {
                multiCL::OpenCLKernel & kernel = getMemsetKernel();
                kernel.reset();
                const uint8_t value = *(static_cast<const uint8_t*>(pattern));
                kernel.setArgument( data, static_cast<uint32_t>(offset), static_cast<uint32_t>(size), value );
                multiCL::launchKernel1D( kernel, size );
            }
            else
            {
#if defined(CL_VERSION_1_2)
                const cl_int error = clEnqueueFillBuffer( OpenCLDeviceManager::instance().device().queue()(), data, pattern, patternSize, offset, size, 0, NULL, NULL );
                if( error != CL_SUCCESS )
                    throw imageException( "Cannot fill a memory for a device" );
#else
                throw imageException( "clEnqueueFillBuffer is not supported in OpenCL with version 1.1 and lower" );
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

    OpenCLContext::OpenCLContext( cl_context context )
        : _context( context )
    {
    }

    OpenCLContext::~OpenCLContext()
    {
        clReleaseContext( _context );
    }

    OpenCLContext::OpenCLContext( const OpenCLContext& )
    {
    }

    OpenCLContext& OpenCLContext::operator= ( const OpenCLContext& )
    {
        return (*this);
    }

    cl_context OpenCLContext::operator()() const
    {
        return _context;
    }

    OpenCLQueue::OpenCLQueue( const OpenCLContext& context, cl_device_id deviceId )
    {
        cl_int error;
        _commandQueue = clCreateCommandQueue( context(), deviceId, 0, &error );
        openCLCheck( error );
    }

    OpenCLQueue::OpenCLQueue( cl_command_queue queue )
        : _commandQueue( queue )
    {
    }

    OpenCLQueue::~OpenCLQueue()
    {
        clReleaseCommandQueue( _commandQueue );
    }

    OpenCLQueue::OpenCLQueue( const OpenCLQueue& )
    {
    }

    OpenCLQueue& OpenCLQueue::operator= ( const OpenCLQueue& )
    {
        return (*this);
    }

    cl_command_queue OpenCLQueue::operator()() const
    {
        return _commandQueue;
    }

    void OpenCLQueue::synchronize()
    {
        openCLCheck( clFinish( _commandQueue ) );
    }

    OpenCLProgram::OpenCLProgram(const OpenCLContext& context, const char * program )
        : _program( NULL )
    {
        cl_int error;
        _program = clCreateProgramWithSource( context(), 1, &program, NULL, &error );
        openCLCheck( error );

        if( !openCLSafeCheck( clBuildProgram( _program, 0, NULL, NULL, NULL, NULL ) ) ) {
            cl_uint deviceCount = 0u;
            openCLCheck( clGetContextInfo( context(), CL_CONTEXT_NUM_DEVICES, sizeof( cl_uint ), &deviceCount, NULL ) );
            assert( deviceCount > 0u );

            std::vector <cl_device_id> device( deviceCount );
            openCLCheck( clGetContextInfo( context(), CL_CONTEXT_DEVICES, sizeof( cl_device_id ) * deviceCount, device.data(), NULL ) );

            std::string fullLog;
            for( std::vector <cl_device_id>::iterator deviceId = device.begin(); deviceId != device.end(); ++deviceId ) {
                size_t logSize = 0;
                clGetProgramBuildInfo( _program, *deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize );

                // Allocate memory for the log
                std::vector <char> log( logSize );

                // Get the log
                clGetProgramBuildInfo( _program, OpenCLDeviceManager::instance().device().deviceId(), CL_PROGRAM_BUILD_LOG, logSize, log.data(), NULL );

                fullLog += std::string( log.begin(), log.end() );
                fullLog += "\n";
            }

            clReleaseProgram( _program );
            throw imageException( (std::string( "Failed to build a program for OpenCL device with following code :\n" ) + fullLog).data() );
        }
    }

    OpenCLProgram::OpenCLProgram( cl_program program )
        : _program( program )
    {
    }

    OpenCLProgram::OpenCLProgram( OpenCLProgram && program )
        : _program( NULL )
    {
        std::swap( _program, program._program );
    }

    OpenCLProgram::~OpenCLProgram()
    {
        clReleaseProgram( _program );
    }

    OpenCLProgram::OpenCLProgram( const OpenCLProgram& )
    {
    }

    OpenCLProgram& OpenCLProgram::operator=( const OpenCLProgram& )
    {
        return (*this);
    }

    cl_program OpenCLProgram::operator()() const
    {
        return _program;
    }

    OpenCLKernel::OpenCLKernel( const OpenCLProgram& program, const std::string& name )
        : _parameterId( 0 )
    {
        cl_int error;
        _kernel = clCreateKernel( program(), name.data(), &error );
        openCLCheck( error );
    }

    OpenCLKernel::OpenCLKernel( cl_kernel kernel )
        : _kernel( kernel )
        , _parameterId( 0 )
    {
    }

    OpenCLKernel::~OpenCLKernel()
    {
        clReleaseKernel( _kernel );
    }

    OpenCLKernel::OpenCLKernel( const OpenCLKernel& )
    {
    }

    OpenCLKernel& OpenCLKernel::operator=( const OpenCLKernel& )
    {
        return (*this);
    }

    cl_kernel OpenCLKernel::operator()() const
    {
        return _kernel;
    }

    void OpenCLKernel::reset()
    {
        _parameterId = 0;
    }

    void OpenCLKernel::_setArgument( size_t size, const void * data )
    {
        openCLCheck( clSetKernelArg( _kernel, _parameterId, size, data ) );
        ++_parameterId;
    }

    OpenCLDevice::OpenCLDevice( cl_device_id deviceId )
        : _deviceId      ( deviceId )
        , _context       ( _deviceId )
        , _currentQueueId( 0u )
    {
        _allocator = new MemoryAllocator( _context(), static_cast<size_t>(totalMemorySize()) );

        setQueueCount( 1u );
    }

    OpenCLDevice::~OpenCLDevice()
    {
        delete _allocator;

        for( std::vector< OpenCLQueue * >::iterator queueId = _queue.begin(); queueId != _queue.end(); ++queueId )
            delete (*queueId);
    }

    OpenCLDevice::OpenCLDevice( const OpenCLDevice& )
        : _deviceId      ( NULL )
        , _context       ( _deviceId )
        , _currentQueueId( 0u )
    {
    }

    OpenCLDevice& OpenCLDevice::operator= ( const OpenCLDevice& )
    {
        return (*this);
    }

    cl_device_id OpenCLDevice::deviceId() const
    {
        return _deviceId;
    }

    OpenCLContext & OpenCLDevice::context()
    {
        return _context;
    }

    const OpenCLContext & OpenCLDevice::context() const
    {
        return _context;
    }

    size_t OpenCLDevice::threadsPerBlock( const OpenCLKernel& kernel ) const
    {
        uint64_t threadsPerBlock = 0u;
        openCLCheck( clGetKernelWorkGroupInfo( kernel(), _deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof( threadsPerBlock ), &threadsPerBlock, NULL ) );

        assert( threadsPerBlock > 0 );

        return static_cast<size_t>(threadsPerBlock);
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
        std::vector<char> deviceName;

        openCLCheck( clGetDeviceInfo( _deviceId, CL_DEVICE_NAME, 0, NULL, &nameLength ) );

        deviceName.resize( nameLength );

        openCLCheck( clGetDeviceInfo( _deviceId, CL_DEVICE_NAME, nameLength, deviceName.data(), NULL ) );

        return deviceName.data();
    }

    std::string OpenCLDevice::computeCapability() const
    {
        size_t capabilityLength = 0u;
        std::vector<char> capability;

        openCLCheck( clGetDeviceInfo( _deviceId, CL_DEVICE_VERSION, 0, NULL, &capabilityLength ) );

        capability.resize( capabilityLength );

        openCLCheck( clGetDeviceInfo( _deviceId, CL_DEVICE_VERSION, capabilityLength, capability.data(), NULL ) );

        return capability.data();
    }

    void OpenCLDevice::synchronize()
    {
        for( std::vector< OpenCLQueue * >::iterator queueId = _queue.begin(); queueId != _queue.end(); ++queueId )
            (*queueId)->synchronize();
    }

    size_t OpenCLDevice::currentQueueId() const
    {
        return _currentQueueId;
    }

    void OpenCLDevice::setCurrentQueueId( size_t queueId )
    {
        if( _currentQueueId != queueId && queueId < _queue.size() )
            _currentQueueId = queueId;
    }

    OpenCLQueue & OpenCLDevice::queue()
    {
        return *(_queue[_currentQueueId]);
    }

    const OpenCLQueue & OpenCLDevice::queue() const
    {
        return *(_queue[_currentQueueId]);
    }

    OpenCLQueue & OpenCLDevice::queue( size_t queueId )
    {
        return *(_queue[queueId]);
    }

    const OpenCLQueue & OpenCLDevice::queue( size_t queueId ) const
    {
        return *(_queue[queueId]);
    }

    size_t OpenCLDevice::queueCount() const
    {
        return _queue.size();
    }

    void OpenCLDevice::setQueueCount( size_t queueCount )
    {
        if( queueCount != _queue.size() ) {
            if( queueCount > _queue.size() ) {
                while( queueCount != _queue.size() )
                    _queue.push_back( new OpenCLQueue( _context, _deviceId ) );
            }
            else {
                if( _currentQueueId >= queueCount )
                    _currentQueueId = 0;

                for( std::vector< OpenCLQueue * >::iterator queueId = _queue.begin() + queueCount; queueId != _queue.end(); ++queueId )
                    delete (*queueId);

                _queue.resize( queueCount );
            }
        }
    }

    MemoryAllocator & OpenCLDevice::allocator()
    {
        return *_allocator;
    }

    const MemoryAllocator & OpenCLDevice::allocator() const
    {
        return *_allocator;
    }


    OpenCLDeviceManager::OpenCLDeviceManager()
    {
        resetSupportedDevice( false, true );
    }

    OpenCLDeviceManager::~OpenCLDeviceManager()
    {
        closeDevices();
    }

    OpenCLDeviceManager & OpenCLDeviceManager::instance()
    {
        static OpenCLDeviceManager manager;
        return manager;
    }

    void OpenCLDeviceManager::initializeDevices()
    {
        for( uint32_t deviceId = 0; deviceId < _supportedDeviceId.size(); ++deviceId )
            initializeDevice( deviceId );
    }

    void OpenCLDeviceManager::initializeDevice( uint32_t deviceId )
    {
        if( deviceId >= _supportedDeviceId.size() )
            throw imageException( "System does not contain a device with such ID" );

        std::list<OpenCLDevice *>::const_iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                              [&]( const OpenCLDevice * device ) { return device->deviceId() == _supportedDeviceId[deviceId]; } );
        if( foundDevice == _device.end() )
            _device.push_back( new OpenCLDevice( _supportedDeviceId[deviceId] ) );
    }

    void OpenCLDeviceManager::closeDevice( uint32_t deviceId )
    {
        if( deviceId >= _supportedDeviceId.size() )
            throw imageException( "System does not contain a device with such ID" );

        std::list<OpenCLDevice *>::iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                        [&]( const OpenCLDevice * device ) { return device->deviceId() == _supportedDeviceId[deviceId]; } );
        if( foundDevice != _device.end() ) {
            delete (*foundDevice);
            _device.erase( foundDevice );
        }
    }

    void OpenCLDeviceManager::closeDevices()
    {
        for( std::list<OpenCLDevice *>::iterator device = _device.begin(); device != _device.end(); ++device )
            delete (*device);

        _device.clear();
    }

    uint32_t OpenCLDeviceManager::deviceCount() const
    {
        return static_cast<uint32_t>(_device.size());
    }

    uint32_t OpenCLDeviceManager::supportedDeviceCount() const
    {
        return static_cast<uint32_t>(_supportedDeviceId.size());
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
        if( _device.empty() )
            throw imageException( "Device manager does not contain any devices" );

        std::list<OpenCLDevice *>::iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                        [&]( const OpenCLDevice * device ) { return device->deviceId() == _supportedDeviceId[deviceId]; } );
        if( foundDevice == _device.end() )
            throw imageException( "Device ID is invalid. Please check that you initialize devices!" );

        return *(*foundDevice);
    }

    const OpenCLDevice & OpenCLDeviceManager::device( uint32_t deviceId ) const
    {
        if( _device.empty() )
            throw imageException( "Device manager does not contain any devices" );

        std::list<OpenCLDevice *>::const_iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                              [&]( const OpenCLDevice * device ) { return device->deviceId() == _supportedDeviceId[deviceId]; } );
        if( foundDevice == _device.end() )
            throw imageException( "Device ID is invalid. Please check that you initialize devices!" );

        return *(*foundDevice);
    }

    void OpenCLDeviceManager::setActiveDevice( uint32_t deviceId )
    {
        std::list<OpenCLDevice *>::iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                        [&]( const OpenCLDevice * device ) { return device->deviceId() == _supportedDeviceId[deviceId]; } );
        if( foundDevice == _device.end() )
            throw imageException( "Device ID is invalid. Please check that you initialize devices!" );

        setDefaultDeviceId( deviceId );
    }

    void OpenCLDeviceManager::resetSupportedDevice( bool enableCpuSupport, bool enableGpuSupport )
    {
        closeDevices();
        _supportedDeviceId.clear();

        cl_uint platformCount = 0u;
        std::vector <cl_platform_id> platformId;
        if( openCLSafeCheck( clGetPlatformIDs( 0, NULL, &platformCount ) ) && (platformCount > 0u) ) {
            platformId.resize( platformCount );
            if( !openCLSafeCheck( clGetPlatformIDs( platformCount, platformId.data(), NULL ) ) ) {
                platformId.clear();
            }
        }

        const cl_device_type deviceType =  (enableCpuSupport ? CL_DEVICE_TYPE_CPU : 0) | (enableGpuSupport ? CL_DEVICE_TYPE_GPU : 0);

        uint32_t supportedDeviceCount = 0u;
        for( std::vector <cl_platform_id>::iterator platform = platformId.begin(); platform != platformId.end(); ++platform ) {
            uint32_t deviceCount = 0u;
            if( openCLSafeCheck( clGetDeviceIDs( *platform, deviceType, 0, NULL, &deviceCount ) ) ) {
                _supportedDeviceId.resize( supportedDeviceCount + deviceCount );

                if( !openCLSafeCheck( clGetDeviceIDs( *platform, deviceType, deviceCount, _supportedDeviceId.data() + supportedDeviceCount, NULL ) ) ) {
                    _supportedDeviceId.resize( supportedDeviceCount );
                    continue;
                }

                supportedDeviceCount += deviceCount;
            }
        }
    }
}
