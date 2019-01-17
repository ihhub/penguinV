#pragma once
#include <cstddef>
#include <vector>

namespace Image_Function
{
    void HaarDirectTransform ( const std::vector< double > & input, std::vector< double > & output, size_t width, size_t height );
    void HaarInverseTransform( const std::vector< double > & input, std::vector< double > & output, size_t width, size_t height );

    void HaarDirectTransform ( const std::vector< float > & input, std::vector< float > & output, size_t width, size_t height );
    void HaarInverseTransform( const std::vector< float > & input, std::vector< float > & output, size_t width, size_t height );
}
