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


def last_chkp(work_dir):
    if(os.path.exists(os.path.join(work_dir,'last_checkpoint'))):
        #TODO
        pass

    files = glob.glob(os.path.join(work_dir,'epoch_*.pth'))
    mx = None
    mnn = -1
    for f in files:
        nn = int(os.path.basename(f).split("epoch_")[1].split(".pth")[0])
        if(nn > mnn):
            mx=f
            mnn=nn

    return mx



if __name__=="__main__":
    a = last_chkp("/home/borisef/projects/mm/mmdetection/boris3/work_dirs/faster-rcnn_r18_fpn_1x_vehicles/")
    print(a)

