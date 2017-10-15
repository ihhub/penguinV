#include "performance_test_helper_cuda.cuh"
#include "../../Library/thirdparty/multicuda/src/cuda_helper.cuh"

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
            multiCuda::cudaCheck( cudaEventRecord( _startEvent, 0 ) );
        }

        void TimerContainerCuda::stop()
        {
            multiCuda::cudaCheck( cudaEventRecord( _stopEvent, 0 ) );
            multiCuda::cudaCheck( cudaEventSynchronize( _stopEvent ) );

            float time = 0.0f;

            multiCuda::cudaCheck( cudaEventElapsedTime( &time, _startEvent, _stopEvent ) );

            push( time );
        }

        Bitmap_Image_Cuda::Image uniformImage( uint32_t width, uint32_t height )
        {
            return uniformImage( width, height, randomValue<uint8_t>( 256 ) );
        }

        Bitmap_Image_Cuda::Image uniformImage( uint32_t width, uint32_t height, uint8_t value )
        {
            Bitmap_Image_Cuda::Image image( width, height );

            image.fill( value );

            return image;
        }

        std::vector< Bitmap_Image_Cuda::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height )
        {
            std::vector < Bitmap_Image_Cuda::Image > image( count );

            for( std::vector< Bitmap_Image_Cuda::Image >::iterator im = image.begin(); im != image.end(); ++im )
                *im = uniformImage( width, height );

            return image;
        }
    }
}
