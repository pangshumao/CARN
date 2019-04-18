import numpy as np
from skimage import transform

def imResize(inIm, out_height=512, out_width=256, in_pixelSizes=[0.4688, 0.4688]):
    '''
    @author: Shumao Pang, E-mail: pangshumao@126.com
    resize an image to the shape of [out_height, out_width]
    :param inIm: a numpy array with shape of [height, width]
    :param out_height:
    :param out_width:
    :param in_pixelSizes: the input pixel size of inIm, the format is [in_h_pixelSize, in_w_pixelSize]
    :return: outIm, a numpy array with the shape of [out_height, out_width],
                out_pixelSize, a numpy array with the format of [out_h_pixelSize, out_w_pixelSize]
    '''
    in_height, in_width = inIm.shape
    outIm = transform.resize(inIm, (out_height, out_width), mode='constant')
    out_pixelSizes = np.zeros((2))
    out_pixelSizes[0] = in_height * in_pixelSizes[0] / out_height
    out_pixelSizes[1] = in_width * in_pixelSizes[1] / out_width
    return outIm, out_pixelSizes

def actual2scale(im, actual_label, pixelSizes, mode='height'):
    '''
    @author: Shumao Pang, E-mail: pangshumao@126.com
    transfer the actual label to scale label
    :param im: the intensity image, with the shape of [height, width]
    :param actual_label: a constant
    :param pixelSizes: the pixel size of im, with the format of [h_pixelSize, w_pixelSize]
    :param mode: 'height' or 'width'
    :return: the scale label
    '''
    height, width = im.shape
    if mode == 'height':
        return actual_label / (height * pixelSizes[0])
    elif mode == 'width':
        return actual_label / (width * pixelSizes[1])

def scale2actual(scale_label, pixelSizes, heights, widths, mode='height'):
    xSize, ySize = scale_label.shape
    if mode == 'height':
        return scale_label * heights * np.transpose(np.tile(pixelSizes[:, 0], (ySize, 1)))
    elif mode == 'width':
        return scale_label * widths * np.transpose(np.tile(pixelSizes[:, 1], (ySize, 1)))