#pragma once

#include <exception>
#include <string>

// Thanks to Visual Studio noexcept is no supported so we have to do like this :(
#ifndef _MSC_VER
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

class imageException : public std::exception
{
public:

    imageException()
        :  _name( "unknown image library exception" )
    {
    }

    explicit imageException( const char * message )
        : _name( message )
    {  
    }

    imageException( const imageException & ex )
        : std::exception( ex )
        , _name         ( ex._name )
    {
        
    }

    virtual ~imageException()
    {
    }

    imageException & operator=( const imageException & ex )
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
