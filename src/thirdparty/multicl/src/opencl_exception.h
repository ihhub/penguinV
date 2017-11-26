#pragma once

#include <exception>
#include <string>

// Thanks to Visual Studio noexcept is no supported so we have to do like this :(
#ifndef _MSC_VER
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

class openCLException : public std::exception
{
public:
    openCLException()
        : _name( "unknown openCL exception" )
    {
    }

    explicit openCLException( const char * message )
        : _name( message )
    {
    }

    openCLException( const openCLException & ex )
        : std::exception( ex )
        , _name( ex._name )
    {
    }

    virtual ~openCLException()
    {
    }

    openCLException & operator=( const openCLException & ex )
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
