#include <algorithm>
#include <assert.h>
#include "../image_exception.h"
#include "cuda_helper.cuh"
#include "cuda_device.cuh"

namespace
{
    thread_local int defaultDeviceId = 0;

    void setDefaultDeviceId( int deviceId )
    {
        defaultDeviceId = deviceId;
    }

    int getDefaultDeviceId()
    {
        return defaultDeviceId;
    }

    // This class is a helper to remember and restore back
    // previous device ID for current thread
    class DeviceAutoRestorer
    {
    public:
        DeviceAutoRestorer( int currentDeviceId )
            : _currentDeviceId ( currentDeviceId )
            , _previousDeviceId( getDefaultDeviceId() )
        {
            if( _currentDeviceId != _previousDeviceId )
                Cuda::cudaCheck( cudaSetDevice( _currentDeviceId ) );
        }

        ~DeviceAutoRestorer()
        {
            if( _currentDeviceId != _previousDeviceId ) {
                Cuda::cudaCheck( cudaSetDevice( _previousDeviceId ) );
                setDefaultDeviceId( _previousDeviceId );
            }
        }

    private:
        int _currentDeviceId;
        int _previousDeviceId;
    };
}

namespace Cuda
{
    namespace MemoryManager
    {
        Cuda_Memory::MemoryAllocator & memory()
        {
            return CudaDeviceManager::instance().device( getDefaultDeviceId() ).allocator();
        }

        Cuda_Memory::MemoryAllocator & memory( int deviceId )
        {
            return CudaDeviceManager::instance().device( deviceId ).allocator();
        }
    }


    CudaDevice::CudaDevice( int deviceId_ )
    {
        if( deviceId_ < 0 )
            imageException( "Invalid CUDA device ID" );

        _deviceId = deviceId_;

        Cuda::cudaCheck( cudaGetDeviceProperties( &_deviceProperty, _deviceId ) );

        DeviceAutoRestorer restorer( _deviceId );

        size_t freeSpace  = 0;
        size_t totalSpace = 0;

        Cuda::cudaCheck( cudaMemGetInfo( &freeSpace, &totalSpace ) );

        assert( totalSpace == _deviceProperty.totalGlobalMem );

        _allocator = new Cuda_Memory::MemoryAllocator( freeSpace );
    }

    CudaDevice::CudaDevice( const CudaDevice & )
    {
    }

    CudaDevice & CudaDevice::operator=( const CudaDevice & )
    {
        return (*this);
    }

    CudaDevice::~CudaDevice()
    {
        delete _allocator;
    }

    int CudaDevice::deviceId() const
    {
        return _deviceId;
    }

    std::string CudaDevice::name() const
    {
        return _deviceProperty.name;
    }

    size_t CudaDevice::totalMemorySize() const
    {
        return _deviceProperty.totalGlobalMem;
    }

    std::string CudaDevice::computeCapability() const
    {
        char capability[32];
        sprintf( capability, "%d.%d", _deviceProperty.major, _deviceProperty.minor );

        return capability;
    }

    size_t CudaDevice::sharedMemoryPerBlock() const
    {
        return _deviceProperty.sharedMemPerBlock;
    }

    Cuda_Memory::MemoryAllocator & CudaDevice::allocator()
    {
        return *_allocator;
    }

    void CudaDevice::setActive()
    {
        Cuda::cudaCheck( cudaSetDevice( _deviceId ) );
        setDefaultDeviceId( _deviceId );
    }


    CudaDeviceManager::CudaDeviceManager()
        : _supportedDeviceCount( 0 )
    {
        int deviceCount = 0;
        if( cudaSafeCheck( cudaGetDeviceCount( &deviceCount ) ) ) {
            _supportedDeviceCount = static_cast<size_t>(deviceCount);

            if( _supportedDeviceCount > 0 )
                initializeDevice( 0 );
        }
    }

    CudaDeviceManager::~CudaDeviceManager()
    {
        closeDevices();
    }

    CudaDeviceManager & CudaDeviceManager::instance()
    {
        static CudaDeviceManager manager;

        return manager;
    }

    void CudaDeviceManager::initializeDevice( int deviceId )
    {
        if( deviceId < 0 || static_cast<size_t>(deviceId) >= _supportedDeviceCount )
            throw imageException( "System does not contain a device with such ID" );

        std::list<CudaDevice *>::const_iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                            [&deviceId]( const CudaDevice * device ) { return device->deviceId() == deviceId; } );
        if( foundDevice == _device.end() )
            _device.push_back( new CudaDevice( deviceId ) );
    }

    void CudaDeviceManager::closeDevice( int deviceId )
    {
        if( deviceId < 0 || static_cast<size_t>(deviceId) >= _supportedDeviceCount )
            throw imageException( "System does not contain a device with such ID" );

        std::list<CudaDevice *>::iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                      [&deviceId]( const CudaDevice * device ) { return device->deviceId() == deviceId; } );
        if( foundDevice != _device.end() ) {
            delete (*foundDevice);
            _device.erase( foundDevice );
        }
    }

    void CudaDeviceManager::closeDevices()
    {
        for( std::list<CudaDevice *>::iterator device = _device.begin(); device != _device.end(); ++device )
            delete (*device);

        _device.clear();
    }

    size_t CudaDeviceManager::deviceCount() const
    {
        return _device.size();
    }

    size_t CudaDeviceManager::supportedDeviceCount() const
    {
        return _supportedDeviceCount;
    }

    CudaDevice & CudaDeviceManager::device()
    {
        return device( getDefaultDeviceId() );
    }

    const CudaDevice & CudaDeviceManager::device() const
    {
        return device( getDefaultDeviceId() );
    }

    CudaDevice & CudaDeviceManager::device( int deviceId )
    {
        if( _device.empty() )
            throw imageException( "Device manager does not contain any devices" );

        std::list<CudaDevice *>::iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                      [&deviceId]( const CudaDevice * cudaDevice ) { return cudaDevice->deviceId() == deviceId; } );
        if( foundDevice == _device.end() )
            throw imageException( "Device ID is invalid. Please check that you initialize devices!" );

        return *(*foundDevice);
    }

    const CudaDevice & CudaDeviceManager::device( int deviceId ) const
    {
        if( _device.empty() )
            throw imageException( "Device manager does not contain any devices" );

        std::list<CudaDevice *>::const_iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                            [&deviceId]( const CudaDevice * cudaDevice ) { return cudaDevice->deviceId() == deviceId; } );
        if( foundDevice == _device.end() )
            throw imageException( "Device ID is invalid. Please check that you initialize devices!" );

        return *(*foundDevice);
    }

    void CudaDeviceManager::setActiveDevice( int deviceId )
    {
        std::list<CudaDevice *>::iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                      [&deviceId]( const CudaDevice * cudaDevice ) { return cudaDevice->deviceId() == deviceId; } );
        if( foundDevice == _device.end() )
            throw imageException( "Device ID is invalid. Please check that you initialize devices!" );

        (*foundDevice)->setActive();
    }
}