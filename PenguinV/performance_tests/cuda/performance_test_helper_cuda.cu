#include "performance_test_helper_cuda.cuh"
#include "../../Library/cuda/cuda_helper.cuh"

namespace Performance_Test
{
    namespace Cuda_Helper
    {
        void TimerContainerCuda::stop()
        {
            Cuda::cudaCheck( cudaDeviceSynchronize() );

            TimerContainer::stop();
        }

        Bitmap_Image_Cuda::Image uniformImage( size_t width, size_t height )
        {
            return uniformImage( width, height, randomValue<uint8_t>( 256 ) );
        }

        Bitmap_Image_Cuda::Image uniformImage( size_t width, size_t height, uint8_t value )
        {
            Bitmap_Image_Cuda::Image image( width, height );

            image.fill( value );

            return image;
        }

        std::vector< Bitmap_Image_Cuda::Image > uniformImages( size_t count, size_t width, size_t height )
        {
            std::vector < Bitmap_Image_Cuda::Image > image( count );

            for( std::vector< Bitmap_Image_Cuda::Image >::iterator im = image.begin(); im != image.end(); ++im )
                *im = uniformImage( width, height );

            return image;
        }
    };
};
