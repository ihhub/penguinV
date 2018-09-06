#pragma once

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <list>
#include <string>
#include <vector>
#include "opencl_memory.h"

namespace multiCL
{
    class OpenCLDeviceManager;

    // This is a shortcut (facade) namespace to access memory allocator for GPU devices
    namespace MemoryManager
    {
        // Returns memory allocator for current thread
        // By default it will return an allocator for device with ID = 0 otherwise an ID what is actual for current thread
        MemoryAllocator & memory();

        // Returns memory allocator for specified device ID
        MemoryAllocator & memory( uint32_t deviceId );
    }

    class OpenCLContext
    {
    public:
        explicit OpenCLContext( cl_device_id deviceId );
        explicit OpenCLContext( cl_context context );
        ~OpenCLContext();

        cl_context operator()() const;

    private:
        cl_context _context;

        OpenCLContext( const OpenCLContext& );
        OpenCLContext& operator= ( const OpenCLContext& );
    };

    class OpenCLProgram
    {
    public:
        OpenCLProgram( const OpenCLContext& context, const char * program );
        explicit OpenCLProgram( cl_program program );
        OpenCLProgram( OpenCLProgram && program );
        ~OpenCLProgram();

        cl_program operator()() const;

    private:
        cl_program _program;

        OpenCLProgram( const OpenCLProgram& );
        OpenCLProgram& operator=( const OpenCLProgram& );
    };

    class OpenCLKernel
    {
    public:
        OpenCLKernel( const OpenCLProgram& program, const std::string& name );
        explicit OpenCLKernel( cl_kernel kernel );
        ~OpenCLKernel();

        cl_kernel operator()() const;

        template <typename T>
        void setArgument( T value )
        {
            _setArgument( sizeof( T ), &value );
        }

        template <typename T, typename... Args>
        void setArgument( T value, Args... args )
        {
            setArgument( value );
            setArgument( args... );
        }

        void reset();

    private:
        cl_kernel _kernel;
        cl_uint _parameterId;

        OpenCLKernel( const OpenCLKernel& );
        OpenCLKernel& operator=( const OpenCLKernel& );

        void _setArgument( size_t size, const void * data );
    };

    class OpenCLQueue
    {
    public:
        OpenCLQueue( const OpenCLContext& context, cl_device_id deviceId );
        explicit OpenCLQueue( cl_command_queue queue );
        ~OpenCLQueue();

        cl_command_queue operator()() const;

        void synchronize();

    private:
        cl_command_queue _commandQueue;

        OpenCLQueue( const OpenCLQueue& );
        OpenCLQueue& operator=( const OpenCLQueue& );
    };

    class OpenCLDevice
    {
    public:
        friend class OpenCLDeviceManager;

        ~OpenCLDevice();

        // Device information
        cl_device_id deviceId() const;

        size_t threadsPerBlock( const OpenCLKernel& kernel ) const; // maximum available number of threads per block

        uint64_t totalMemorySize() const; // total available memory in bytes
        std::string name() const;
        std::string computeCapability() const;

        // Device manipulation
        void synchronize(); // synchronize all operations on device with CPU

        OpenCLContext & context();
        const OpenCLContext & context() const;

        size_t currentQueueId() const; // current queue ID which is used as a default value in queue() function
        void setCurrentQueueId( size_t queueId );

        OpenCLQueue & queue(); // a reference to current queue
        const OpenCLQueue & queue() const;

        OpenCLQueue & queue( size_t queueId ); // a reference to queue with specified ID
        const OpenCLQueue & queue( size_t queueId ) const;

        size_t queueCount() const; // total number of queues
        void setQueueCount( size_t queueCount );

        MemoryAllocator & allocator(); // memory allocator associated with device
        const MemoryAllocator & allocator() const;

    private:
        cl_device_id _deviceId;
        OpenCLContext _context;

        size_t _currentQueueId;
        std::vector< OpenCLQueue * > _queue; // array of queues within the device

        MemoryAllocator * _allocator; // memory allocator on current device

        explicit OpenCLDevice( cl_device_id deviceId );
        OpenCLDevice( const OpenCLDevice& );
        OpenCLDevice& operator= ( const OpenCLDevice& );
    };

    class OpenCLDeviceManager
    {
    public:
        static OpenCLDeviceManager & instance();

        void initializeDevices(); // initializes all GPU devices available in system
        void initializeDevice( uint32_t deviceId ); // initializes a GPU device with specified ID
        void closeDevice( uint32_t deviceId ); // closes initialized GPU device with specified ID
        void closeDevices(); // closes all GPU devices initialized by manager

        uint32_t deviceCount() const; // initialized GPU devices via manager
        uint32_t supportedDeviceCount() const; // maximum available GPU devices in the system

        OpenCLDevice & device(); // returns GPU device within current thread
        const OpenCLDevice & device() const; // returns GPU device within current thread

        OpenCLDevice & device( uint32_t deviceId ); // returns GPU device with specified ID
        const OpenCLDevice & device( uint32_t deviceId ) const; // returns GPU device with specified ID

        void setActiveDevice( uint32_t deviceId ); // set GPU device with specified ID as a active device in current thread

    private:
        OpenCLDeviceManager();
        ~OpenCLDeviceManager();

        std::vector < cl_device_id > _supportedDeviceId;
        std::list<OpenCLDevice *> _device; // a list of initialized GPU devices
    };
}
