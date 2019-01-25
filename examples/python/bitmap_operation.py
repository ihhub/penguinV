''' 
Example of how to use python wrappers of penguinV to save a gray scale version of a bitmap image.
'''

# Module files for penguinV are in separate folder, so we use the sys module to add that location
# to the places that python will look for the module.

import sys
import os
sys.path.insert(0, os.path.join('..', '..', 'src', 'python'))

# Now get the module.

import penguinV

def saveGrayScaleCopy(filenameToOpen, filenameToSave):
    ''' Saves a Gray Scale Thresholded version of a bitmap file.
    
    Parameters
    ----------
    filenameToOpen : String
        Name of the bitmap to open.

    filenameToSave : String
        Name of the resulting bitmap.
    '''

    image = penguinV.Load(filenameToOpen)
    
    if image.empty():
        raise penguinV.ImageException('penguinV.Load()', 'Problem loading file ' + filenameToOpen)

    if image.colorCount() != penguinV.GRAY_SCALE:
        image = penguinV.ConvertToGrayScale(image)

    threshold_value = penguinV.GetThreshold( penguinV.Histogram(image) ) 
    image = penguinV.Threshold(image, threshold_value)

    penguinV.Save(filenameToSave, image)

########################
### main execution
########################

try:

    saveGrayScaleCopy( filenameToOpen = os.path.join('..', '_image', 'mercury.bmp'),
                       filenameToSave = 'result.bmp'
                     )
    print('Output image was saved into result.bmp file')

except penguinV.ImageException as err:

        # This error occurs when there was trouble opening the file.

        print(err)

except BaseException as err: 

        # Something more serious has gone wrong.

        print('Unknown Error ', err) 
