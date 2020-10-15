#include <algorithm>
#include <assert.h>
#include "cuda_device.cuh"
#include "cuda_helper.cuh"
#include "../penguinv_exception.h"

namespace
{
#if (_MSC_VER && _MSC_VER >= 1400)
    #ifndef thread_local
        #define thread_local __declspec(thread)
    #endif
#endif

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
                multiCuda::cudaCheck( cudaSetDevice( _currentDeviceId ) );
        }

        ~DeviceAutoRestorer()
        {
            if( _currentDeviceId != _previousDeviceId ) {
                multiCuda::cudaCheck( cudaSetDevice( _previousDeviceId ) );
                setDefaultDeviceId( _previousDeviceId );
            }
        }

    private:
        int _currentDeviceId;
        int _previousDeviceId;
    };
}

namespace multiCuda
{
    namespace MemoryManager
    {
        MemoryAllocator & memory()
        {
            return CudaDeviceManager::instance().device( getDefaultDeviceId() ).allocator();
        }

        MemoryAllocator & memory( int deviceId )
        {
            return CudaDeviceManager::instance().device( deviceId ).allocator();
        }
    }


    CudaDevice::CudaDevice( int deviceId_ )
        : _currentStreamId( 0u )
    {
        if( deviceId_ < 0 )
            penguinVException( "Invalid CUDA device ID" );

        _deviceId = deviceId_;

        cudaCheck( cudaGetDeviceProperties( &_deviceProperty, _deviceId ) );

        DeviceAutoRestorer restorer( _deviceId );

        size_t freeSpace  = 0;
        size_t totalSpace = 0;

        cudaCheck( cudaMemGetInfo( &freeSpace, &totalSpace ) );

        assert( totalSpace == _deviceProperty.totalGlobalMem );

        _allocator = new MemoryAllocator( freeSpace );

        _backupDeviceProperty = _deviceProperty;
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
        setActive();
        delete _allocator;

        for( std::vector< CudaStream * >::iterator streamId = _stream.begin(); streamId != _stream.end(); ++streamId )
            delete (*streamId);
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

    uint32_t CudaDevice::threadsPerBlock() const
    {
        return static_cast<uint32_t>(_deviceProperty.maxThreadsPerBlock);
    }

    uint32_t CudaDevice::maximumThreadsPerBlock() const
    {
        return static_cast<uint32_t>(_backupDeviceProperty.maxThreadsPerBlock);
    }

    dim3 CudaDevice::blockDimension() const
    {
        return dim3( static_cast<uint32_t>(_deviceProperty.maxThreadsDim[0]),
                     static_cast<uint32_t>(_deviceProperty.maxThreadsDim[1]),
                     static_cast<uint32_t>(_deviceProperty.maxThreadsDim[2]) );
    }

    dim3 CudaDevice::dimensionSize() const
    {
        return dim3( static_cast<uint32_t>(_deviceProperty.maxGridSize[0]),
                     static_cast<uint32_t>(_deviceProperty.maxGridSize[1]),
                     static_cast<uint32_t>(_deviceProperty.maxGridSize[2]) );
    }

    int CudaDevice::dmaEngineCount() const
    {
        return _deviceProperty.asyncEngineCount;
    }

    void CudaDevice::setThreadsPerBlock( uint32_t threadCount )
    {
        if( (threadCount == 0) || (threadCount % 32) != 0 )
            throw penguinVException( "Invalid thread count per block" );

        const int threads = static_cast<int>(threadCount);

        if( threads <= _backupDeviceProperty.maxThreadsPerBlock )
            _deviceProperty.maxThreadsPerBlock = threads;
    }

    void CudaDevice::synchronize()
    {
        cudaCheck( cudaDeviceSynchronize() );
    }

    size_t CudaDevice::currentStreamId() const
    {
        return _currentStreamId;
    }

    void CudaDevice::setCurrentStreamId( size_t streamId )
    {
        if( _currentStreamId != streamId && streamId < _stream.size() )
            _currentStreamId = streamId;
    }

    CudaStream & CudaDevice::stream()
    {
        return *(_stream[_currentStreamId]);
    }

    const CudaStream & CudaDevice::stream() const
    {
        return *(_stream[_currentStreamId]);
    }

    CudaStream & CudaDevice::stream( size_t streamId )
    {
        return *(_stream[streamId]);
    }

    const CudaStream & CudaDevice::stream( size_t streamId ) const
    {
        return *(_stream[streamId]);
    }

    size_t CudaDevice::streamCount() const
    {
        return _stream.size();
    }

    void CudaDevice::setStreamCount( size_t streamCount )
    {
        if( streamCount != _stream.size() ) {
            if( streamCount > _stream.size() ) {
                while( streamCount != _stream.size() )
                    _stream.push_back( new CudaStream() );
            }
            else {
                if( _currentStreamId >= streamCount )
                    _currentStreamId = 0;

                for( std::vector< CudaStream * >::iterator streamId = _stream.begin() + streamCount; streamId != _stream.end(); ++streamId )
                    delete (*streamId);

                _stream.resize( streamCount );
            }
        }
    }

    MemoryAllocator & CudaDevice::allocator()
    {
        return *_allocator;
    }

    const MemoryAllocator & CudaDevice::allocator() const
    {
        return *_allocator;
    }

    void CudaDevice::setActive()
    {
        cudaCheck( cudaSetDevice( _deviceId ) );
        setDefaultDeviceId( _deviceId );
    }


    CudaDeviceManager::CudaDeviceManager()
        : _supportedDeviceCount( 0 )
    {
        int deviceCount = 0;
        if( cudaSafeCheck( cudaGetDeviceCount( &deviceCount ) ) )
            _supportedDeviceCount = deviceCount;
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

    void CudaDeviceManager::initializeDevices()
    {
        for( int deviceId = 0; deviceId < _supportedDeviceCount; ++deviceId )
            initializeDevice( deviceId );
    }

    void CudaDeviceManager::initializeDevice( int deviceId )
    {
        if( deviceId < 0 || deviceId >= _supportedDeviceCount )
            throw penguinVException( "System does not contain a device with such ID" );

        std::list<CudaDevice *>::const_iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                            [&deviceId]( const CudaDevice * device ) { return device->deviceId() == deviceId; } );
        if( foundDevice == _device.end() )
            _device.push_back( new CudaDevice( deviceId ) );
    }

    void CudaDeviceManager::closeDevice( int deviceId )
    {
        if( deviceId < 0 || deviceId >= _supportedDeviceCount )
            throw penguinVException( "System does not contain a device with such ID" );

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

    int CudaDeviceManager::deviceCount() const
    {
        return static_cast<int>(_device.size()); // CUDA works with signed int rathen than unsigned int :(
    }

    int CudaDeviceManager::supportedDeviceCount() const
    {
        return _supportedDeviceCount; // CUDA works with signed int rathen than unsigned int :(
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
            throw penguinVException( "Device manager does not contain any devices" );

        std::list<CudaDevice *>::iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                      [&deviceId]( const CudaDevice * cudaDevice ) { return cudaDevice->deviceId() == deviceId; } );
        if( foundDevice == _device.end() )
            throw penguinVException( "Device ID is invalid. Please check that you initialize devices!" );

        return *(*foundDevice);
    }

    const CudaDevice & CudaDeviceManager::device( int deviceId ) const
    {
        if( _device.empty() )
            throw penguinVException( "Device manager does not contain any devices" );

        std::list<CudaDevice *>::const_iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                            [&deviceId]( const CudaDevice * cudaDevice ) { return cudaDevice->deviceId() == deviceId; } );
        if( foundDevice == _device.end() )
            throw penguinVException( "Device ID is invalid. Please check that you initialize devices!" );

        return *(*foundDevice);
    }

    void CudaDeviceManager::setActiveDevice( int deviceId )
    {
        std::list<CudaDevice *>::iterator foundDevice = std::find_if( _device.begin(), _device.end(),
                                                                      [&deviceId]( const CudaDevice * cudaDevice ) { return cudaDevice->deviceId() == deviceId; } );
        if( foundDevice == _device.end() )
            throw penguinVException( "Device ID is invalid. Please check that you initialize devices!" );

        (*foundDevice)->setActive();
    }

    CudaStream::CudaStream()
        : _id( 0 )
    {
        cudaCheck( cudaStreamCreateWithFlags( &_id, cudaStreamNonBlocking ) );
    }

    CudaStream::~CudaStream()
    {
        cudaSafeCheck( cudaStreamDestroy( _id ) );
    }

    void CudaStream::synchronize()
    {
        cudaCheck( cudaStreamSynchronize( _id ) );
    }

    bool CudaStream::isFree()
    {
        cudaError error = cudaStreamQuery( _id );

        if( error == cudaSuccess )
            return true;
        if( error == cudaErrorNotReady )
            return false;

        cudaCheck( error );

        return false;
    }

    cudaStream_t CudaStream::id() const
    {
        return _id;
    }

    void CudaStream::setCallback( cudaStreamCallback_t callbackFunction, void * data )
    {
        cudaCheck( cudaStreamAddCallback( _id, callbackFunction, data, 0 ) );
    }
}
