/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2022                                             *
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

#include <cmath>
#include <cuda_runtime.h>
#include "filtering_cuda.cuh"
#include "../filtering.h"
#include "../image_buffer.h"
#include "../penguinv_exception.h"
#include "../parameter_validation.h"

namespace Image_Function_Cuda
{
    Image Gaussian( const Image & in, uint32_t kernelSize, float sigma )
    {
        Image_Function::ValidateImageParameters( in );

        ImageCuda out( in.width(), in.height() );

        Gaussian( in, out, kernelSize, sigma );

        return out;
    }

    void Gaussian( const Image & in, Image & out, uint32_t kernelSize,  float sigma )
    {
        Image_Function::ValidateImageParameters( in, out );

        if( sigma < 0 )
            throw penguinVException( "Sigma value cannot be negative" );

        FFT_Cuda::ComplexData image( in );
        FFT_Cuda::ComplexData filter = GetGaussianKernel( in.width(), in.height(), kernelSize, sigma );

        FFT_Cuda::FFTExecutor executor( in.width(), in.height() );

        executor.directTransform( image );
        executor.directTransform( filter );

        executor.complexMultiplication( image, filter, image );

        executor.inverseTransform( image );

        out = image.get();
    }

    FFT_Cuda::ComplexData GetGaussianKernel( uint32_t width, uint32_t height, uint32_t kernelSize, float sigma )
    {
        std::vector<float> data;
        Image_Function::GetGaussianKernel( data, width, height, kernelSize, sigma );

        multiCuda::Array<float> cudaData( data );

        FFT_Cuda::ComplexData complexData;
        complexData.resize( width, height );
        complexData.set( cudaData );

        return complexData;
    }
}
