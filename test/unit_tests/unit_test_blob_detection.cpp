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

#include "unit_test_blob_detection.h"
#include "../../src/blob_detection.h"
#include "../../src/image_function.h"
#include "unit_test_framework.h"
#include "unit_test_helper.h"

namespace blob_detection
{
    bool Detect1Blob()
    {
        for ( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            penguinV::Image image = Unit_Test::blackImage();

            uint32_t roiX, roiY;
            uint32_t roiWidth, roiHeight;
            Unit_Test::generateRoi( image, roiX, roiY, roiWidth, roiHeight );

            Unit_Test::fillImage( image, roiX, roiY, roiWidth, roiHeight, Unit_Test::randomValue<uint8_t>( 1, 256 ) );

            Blob_Detection::BlobDetection detection;
            detection.find( image );

            const uint32_t contour = ( ( roiWidth > 1 ) && ( roiHeight > 2 ) ) ? ( 2 * roiWidth + 2 * ( roiHeight - 2 ) ) : ( roiWidth * roiHeight );

            if ( detection().size() != 1 || detection()[0].width() != roiWidth || detection()[0].height() != roiHeight || detection()[0].size() != roiWidth * roiHeight
                 || detection()[0].contourX().size() != contour || detection()[0].edgeX().size() != contour )
                return false;
        }

        return true;
    }
}

void addTests_Blob_Detection( UnitTestFramework & framework )
{
    framework.add( blob_detection::Detect1Blob, "blob_detection::Detect one blob" );
}
