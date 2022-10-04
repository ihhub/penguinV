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

#include <intrin.h>

struct CpuInformation
{
    static bool isSseSupported()
    {
        int info[4];
        __cpuidex( info, 0, 0 );
        const int id = info[0];

        if ( id >= 0x00000001 ) {
            __cpuidex( info, 0x00000001, 0 );
            return ( info[3] & ( (int)1 << 26 ) ) != 0;
        }

        return false;
    }

    static bool isAvxSupported()
    {
        int info[4];
        __cpuidex( info, 0, 0 );
        const int id = info[0];

        if ( id >= 0x00000007 ) {
            __cpuidex( info, 0x00000007, 0 );
            return ( info[1] & ( (int)1 << 5 ) ) != 0;
        }

        return false;
    }

    static bool isAvx512SKLSupported()
    {
        int info[4];
        __cpuidex( info, 0, 0 );
        const int id = info[0];

        if ( id >= 0x00000007 ) {
            __cpuidex( info, 0x00000007, 0 );
            return ( info[1] & ( ( (int)1 << 31 ) | ( (int)1 << 30 ) | ( (int)1 << 28 ) | ( (int)1 << 17 ) | ( (int)1 << 16 ) ) ) != 0;
        }

        return false;
    }

    static bool isNeonSupported()
    {
#ifdef _M_ARM
        return true;
#else
        return false;
#endif
    }
};
