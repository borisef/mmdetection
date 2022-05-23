import os, glob
import numpy as np

EXTS = ['jpg', 'tif', 'png']


def redefine_images(imgs):
    if (isinstance(imgs, list)):
        return imgs

    if (isinstance(imgs, str)):
        pa = imgs
        if (os.path.exists(pa)):
            # get all images from path
            for ext in EXTS:
                temp = glob.glob(os.path.join(pa, "*." + ext))
                if (len(temp) > 0):
                    imgs = temp
                    return imgs
    else:
        return imgs
