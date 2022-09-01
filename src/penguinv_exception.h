#pragma once

#include <exception>
#include <string>

// Thanks to Visual Studio noexcept is no supported so we have to do like this :(
#ifndef _MSC_VER
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

class penguinVException : public std::exception
{
public:
    penguinVException()
        : _name( "unknown image library exception" )
    {}

    explicit penguinVException( const char * message )
        : _name( message )
    {}

    explicit penguinVException( const std::string & message )
        : _name( message.data() )
    {}

    penguinVException( const penguinVException & ex )
        : std::exception( ex )
        , _name( ex._name )
    {}

    virtual ~penguinVException() {}

    penguinVException & operator=( const penguinVException & ex )
    {
        std::exception::operator=( ex );

        _name = ex._name;

        return ( *this );
    }

    virtual const char * what() const NOEXCEPT
    {
        return _name.c_str();
    }

private:
    std::string _name;
};
