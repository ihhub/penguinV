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
