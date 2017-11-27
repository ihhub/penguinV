#pragma once

#include <cuda_runtime.h>
#include <list>
#include <string>
#include <vector>
#include "cuda_memory.cuh"

namespace multiCuda
{
    class CudaDeviceManager;
    class CudaStream;

    // This is a shortcut (facade) namespace to access memory allocator for CUDA devices
    namespace MemoryManager
    {
        // Returns memory allocator for current thread
        // By default it will return an allocator for device with ID = 0 otherwise an ID what is actual for current thread
        MemoryAllocator & memory();

        // Returns memory allocator for specified device ID
        MemoryAllocator & memory( int deviceId );
    }

    // This class represents a single CUDA device (videocard) in a system
    // An object of the class can only be contructed by CudaDeviceManager
    class CudaDevice
    {
    public:
        friend class CudaDeviceManager;

        ~CudaDevice();

        // Device information
        int deviceId() const;
        std::string name() const;
        size_t totalMemorySize() const; // total available memory in bytes
        std::string computeCapability() const; // compute capability or version of supported computations
        size_t sharedMemoryPerBlock() const; // maximum supported shared memory in bytes per block
        uint32_t threadsPerBlock() const; // current number of threads per block
        uint32_t maximumThreadsPerBlock() const; // maximum available number of threads per block
        dim3 blockDimension() const; // maximum number of threads per block
        dim3 dimensionSize() const; // maximum supported dimension size of block's grid
        int dmaEngineCount() const; // supported DMA engines count to execute copy operations concurrently with kernel execution

        // User defined settings
        void setThreadsPerBlock( uint32_t threadCount ); // limit number of threads per block. The value cannot be bigger than device's supported and it must be a miltiplier of 32

        // Device manipulation
        void synchronize(); // synchronize all operations on device with CPU

        size_t currentStreamId() const; // current stream ID which is used as a default value in stream() function
        void setCurrentStreamId( size_t streamId );

        CudaStream & stream(); // a reference to current stream
        const CudaStream & stream() const;

        CudaStream & stream( size_t streamId ); // a reference to stream with specified ID
        const CudaStream & stream( size_t streamId ) const;

        size_t streamCount() const; // total number of streams
        void setStreamCount( size_t streamCount );

        MemoryAllocator & allocator(); // memory allocator associated with device
        const MemoryAllocator & allocator() const;
    private:
        int _deviceId; // associated device ID
        cudaDeviceProp _deviceProperty; // full device properties taken from CUDA
        cudaDeviceProp _backupDeviceProperty; // backup version of device properties

        size_t _currentStreamId;
        std::vector< CudaStream * > _stream; // array of streams within the device

        MemoryAllocator * _allocator; // memory allocator on current device

        CudaDevice( int deviceId_ );
        CudaDevice( const CudaDevice & );
        CudaDevice & operator=( const CudaDevice & );

        void setActive(); // set device active for currect thread
    };

    // Manager class for all CUDA devices which are in the system
    class CudaDeviceManager
    {
    public:
        static CudaDeviceManager & instance();

        void initializeDevices(); // initializes all CUDA devices available in system
        void initializeDevice( int deviceId ); // initializes a CUDA device with specified ID
        void closeDevice( int deviceId ); // closes initialized CUDA device with specified ID
        void closeDevices(); // closes all CUDA devices initialized by manager

        int deviceCount() const; // initialized CUDA devices via manager
        int supportedDeviceCount() const; // maximum available CUDA devices in the system

        CudaDevice & device(); // returns CUDA device within current thread
        const CudaDevice & device() const; // returns CUDA device within current thread

        CudaDevice & device( int deviceId ); // returns CUDA device with specified ID
        const CudaDevice & device( int deviceId ) const; // returns CUDA device with specified ID

        void setActiveDevice( int deviceId ); // set CUDA device with specified ID as a active device in current thread

    private:
        CudaDeviceManager();
        ~CudaDeviceManager();

        int _supportedDeviceCount; // maximum available CUDA devices in the system
        std::list<CudaDevice *> _device; // a list of initialized CUDA devices
    };

    // A wrapper class for CUDA stream
    class CudaStream
    {
    public:
        CudaStream();
        ~CudaStream();

        void synchronize(); // synchronize CPU and this stream
        bool isFree(); // returns true if no operation is currently executed within the stream

        cudaStream_t id() const; // returns an ID used in kernel executions or memory operations

        void setCallback( cudaStreamCallback_t callbackFunction, void * data ); // sets callback function for stream

    private:
        cudaStream_t _id;
    };
}
