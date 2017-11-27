#pragma once

#include <exception>
#include <string>

// Thanks to Visual Studio noexcept is no supported so we have to do like this :(
#ifndef _MSC_VER
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

class multiCudaException : public std::exception
{
public:
    multiCudaException()
        : _name( "unknown multiCUDA exception" )
    {
    }

    explicit multiCudaException( const char * message )
        : _name( message )
    {
    }

    multiCudaException( const multiCudaException & ex )
        : std::exception( ex )
        , _name( ex._name )
    {
    }

    virtual ~multiCudaException()
    {
    }

    multiCudaException & operator=( const multiCudaException & ex )
    {
        std::exception::operator=( ex );

        _name = ex._name;

        return (*this);
    }

    virtual const char * what() const NOEXCEPT
    {
        return _name.c_str();
    }

private:
    std::string _name;
};
