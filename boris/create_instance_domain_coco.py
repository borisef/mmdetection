import json
import os, random
import shutil, random
import cv2
import numpy as np


def change_crop(cropIn, newDomain):
    cropOut = cropIn.copy()
    if(newDomain == "GRAYBLUR"):
        temp = cv2.cvtColor(cropIn, cv2.COLOR_BGR2GRAY)
        cropOut[:, :, 0] = temp
        cropOut[:, :, 1] = temp
        cropOut[:, :, 2] = temp
        # Apply blurring kernel
        kernel2 = np.ones((5, 5), np.float32) / 25
        cropOut = cv2.filter2D(src=cropOut, ddepth=-1, kernel=kernel2)

    if (newDomain == "GRAY"):
        temp = cv2.cvtColor(cropIn, cv2.COLOR_BGR2GRAY)
        cropOut[:, :, 0] = temp
        cropOut[:, :, 1] = temp
        cropOut[:, :, 2] = temp


    if(newDomain == "NEGATIVE"):
        temp = cv2.cvtColor(cropIn, cv2.COLOR_BGR2GRAY)
        cropOut[:, :, 0] = 255 - temp
        cropOut[:, :, 1] = 255 - temp
        cropOut[:, :, 2] = 255 - temp

    if(newDomain == "RGB"):
        pass



    return cropOut

def get_image_crop(imgmat, bbox):
    cropped_image = imgmat[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2]),:]

    #cv2.imshow("cropped", cropped_image)
    # Save the cropped image
    cv2.imwrite("Cropped Image.jpg", cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return cropped_image

def set_image_crop(imgmat, bbox, crop):
    imgmat[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2]), :] = crop
    return imgmat


def synthesize_multidomain_coco(imgPath,  json_name, out_json, imgExt = '.jpg', newImgExt = '_md.jpg', domains = ['RGB', 'GRAYBLUR', 'NEGATIVE'], img_domain = 0):
    # load json
    with open(json_name, 'r') as f:
        dat = json.load(f)
    num_domains = len(domains)
    for im in dat['images']:
        im_name = im["file_name"]
        image_id = im["id"]
        imgmat = cv2.imread(os.path.join(imgPath,im_name))

        for ann in dat['annotations']:
            if(ann["image_id"] != image_id):
                continue
            bbox = ann["bbox"]
            crop = get_image_crop(imgmat,bbox)

            #select random id in [0,num_domains-1]
            rd = random.randrange(0, num_domains)
            #change crop
            newcrop = change_crop(crop, domains[rd])
            imgmat = set_image_crop(imgmat, bbox, newcrop)

            #update json
            ann["domain_id"] = rd

        #image domain
        im["domain_id"] = img_domain
        im_name_new = im_name.replace(imgExt, newImgExt)
        full_im_name_new = os.path.join(imgPath, im_name_new)
        im["file_name"] = im_name_new
        cv2.imwrite(full_im_name_new, imgmat)
    #save json
    with open(out_json, 'w') as outfile:
        json.dump(dat, outfile, indent=4)


if __name__=="__main__":
    if(0):
        imgPath = '/home/borisef/datasets/traffic/set_100_domain_mixed_3instances/'
        json_name = '/home/borisef/datasets/traffic/set_100_domain_mixed_3instances/data.json'
        out_json = '/home/borisef/datasets/traffic/set_100_domain_mixed_3instances/data_da.json'
        domains = ['RGB', 'GRAYBLUR', 'NEGATIVE']
        synthesize_multidomain_coco(imgPath = imgPath,  json_name= json_name,
                                    out_json=out_json, imgExt = '.jpg', newImgExt = '_md.jpg',
                                    domains = domains)

    if (0):
        imgPath = '/home/borisef/datasets/traffic/set_100_domain_mixed_2instances/'
        json_name = '/home/borisef/datasets/traffic/set_100_domain_mixed_2instances/data.json'
        out_json = '/home/borisef/datasets/traffic/set_100_domain_mixed_2instances/data_da.json'
        domains = ['RGB', 'GRAYBLUR']
        synthesize_multidomain_coco(imgPath=imgPath, json_name=json_name, out_json=out_json, imgExt='.jpg',
                                    newImgExt='_md.jpg', domains=domains)

    if (1):
        imgPath = '/home/borisef/datasets/traffic/set_500_domain_mixed_3instances/'
        json_name = '/home/borisef/datasets/traffic/set_500_domain_mixed_3instances/data.json'
        out_json = '/home/borisef/datasets/traffic/set_500_domain_mixed_3instances/data_da.json'
        domains = ['RGB', 'GRAYBLUR', 'NEGATIVE']
        synthesize_multidomain_coco(imgPath=imgPath, json_name=json_name,
                                    out_json=out_json, imgExt='.jpg', newImgExt='_md.jpg',
                                    domains=domains)