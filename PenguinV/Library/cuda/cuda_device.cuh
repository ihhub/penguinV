#pragma once

#include <cuda_runtime.h>
#include <string>
#include <list>
#include "cuda_memory.cuh"

namespace Cuda
{
    class CudaDeviceManager;

    // This is a shortcut (facade) namespace to access memory allocator for CUDA devices
    namespace MemoryManager
    {
        // Returns memory allocator for current thread
        // By default it will return an allocator for device with ID = 0
        Cuda_Memory::MemoryAllocator & memory();

        // Returns memory allocator for specified device ID
        Cuda_Memory::MemoryAllocator & memory( int deviceId );
    };

    // This class represents a single CUDA device (videocard) in a system
    // An object of the class can only be contructed by CudaDeviceManager
    class CudaDevice
    {
    public:
        friend class CudaDeviceManager;

        ~CudaDevice();

        int deviceId() const; // assigned device ID
        std::string name() const; // name of the device
        size_t totalMemorySize() const; // total available memory in bytes
        std::string computeCapability() const; // compute capability or version of supported computations
        size_t sharedMemoryPerBlock() const; // maximum supported shared memory in bytes per block

        Cuda_Memory::MemoryAllocator & allocator(); // memory allocator associated with device
    private:
        int _deviceId; // associated device ID
        cudaDeviceProp _deviceProperty; // full device properties taken from CUDA

        Cuda_Memory::MemoryAllocator * _allocator; // memory allocator on current device

        CudaDevice( int deviceId_ );
        CudaDevice( const CudaDevice & device );
        CudaDevice & operator=( const CudaDevice & device );

        void setActive(); // set device active for currect thread
    };

    // Manager class for all CUDA devices which are in the system
    class CudaDeviceManager
    {
    public:
        static CudaDeviceManager & instance();

        void initializeDevice( int deviceId ); // initializes a CUDA device with specified ID
        void closeDevice( int deviceId ); // closes initialized CUDA device with specified ID
        void closeDevices(); // closes all CUDA devices initialized by manager

        size_t deviceCount() const; // initialized CUDA devices via manager
        size_t supportedDeviceCount() const; // maximum available CUDA devices in the system

        CudaDevice & device(); // returns CUDA device within current thread
        const CudaDevice & device() const; // returns CUDA device within current thread

        CudaDevice & device( int deviceId ); // returns CUDA device with specified ID
        const CudaDevice & device( int deviceId ) const; // returns CUDA device with specified ID

        void setActiveDevice( int deviceId ); // set CUDA device with specified ID as a active device in current thread

    private:
        CudaDeviceManager();
        ~CudaDeviceManager();

        size_t _supportedDeviceCount; // maximum available CUDA devices in the system
        std::list<CudaDevice *> _device; // a list of initialized CUDA devices
    };
}
