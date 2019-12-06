#include "performance_test_helper_cuda.cuh"
#include "../../../src/cuda/cuda_helper.cuh"

namespace
{
    void setCudaThreadCount( uint32_t threadCount )
    {
        multiCuda::CudaDeviceManager::instance().device().setThreadsPerBlock( threadCount );
    }

    uint32_t getMaximumCudaThreadCount()
    {
        return multiCuda::CudaDeviceManager::instance().device().maximumThreadsPerBlock();
    }
}

namespace Performance_Test
{
    namespace Cuda_Helper
    {
        TimerContainerCuda::TimerContainerCuda()
        {
            multiCuda::cudaCheck( cudaEventCreate( &_startEvent ) );
            multiCuda::cudaCheck( cudaEventCreate( &_stopEvent  ) );
        }

        TimerContainerCuda::~TimerContainerCuda()
        {
            multiCuda::cudaCheck( cudaEventDestroy( _startEvent ) );
            multiCuda::cudaCheck( cudaEventDestroy( _stopEvent  ) );
        }

        void TimerContainerCuda::start()
        {
            multiCuda::cudaCheck( cudaEventRecord( _startEvent, multiCuda::getCudaStream() ) );
        }

        void TimerContainerCuda::stop()
        {
            multiCuda::cudaCheck( cudaEventRecord( _stopEvent, multiCuda::getCudaStream() ) );
            multiCuda::cudaCheck( cudaEventSynchronize( _stopEvent ) );

            float time = 0.0f;

            multiCuda::cudaCheck( cudaEventElapsedTime( &time, _startEvent, _stopEvent ) );

            push( time );
        }

        std::pair < double, double > runPerformanceTestCuda( performanceFunctionCuda function, uint32_t size, uint32_t threadCountDivider )
        {
            setCudaThreadCount( getMaximumCudaThreadCount() / threadCountDivider );
            TimerContainerCuda timer;
            function(timer, size);
            return timer.mean();
        }

        PenguinV_Image::Image uniformImage( uint32_t width, uint32_t height )
        {
            return uniformImage( width, height, randomValue<uint8_t>( 256 ) );
        }

        PenguinV_Image::Image uniformImage( uint32_t width, uint32_t height, uint8_t value )
        {
            PenguinV_Image::ImageCuda image( width, height );

            image.fill( value );

            PenguinV_Image::Image imageOut;
            imageOut.swap( image );

            return imageOut;
        }

        std::vector< PenguinV_Image::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height )
        {
            std::vector < PenguinV_Image::Image > image( count );

            for( std::vector< PenguinV_Image::Image >::iterator im = image.begin(); im != image.end(); ++im )
                *im = uniformImage( width, height );

            return image;
        }
    }
}
