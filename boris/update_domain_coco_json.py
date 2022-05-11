import json
import os, random
import shutil

def replace_images_in_json(imgPath, json_name,out_json, max_gray = 250):
    # load json
    with open(json_name, 'r') as f:
        dat = json.load(f)

    dat_new = dat  # TODO: deep copy
    cg = 0
    for im in dat_new["images"]:
        im_name = im["file_name"]
        im_name_new = im_name[:-4] + '_gray.jpg'
        if(os.path.exists(os.path.join(imgPath,im_name_new))):
            im["file_name"] = im_name_new
            cg = cg + 1
            if(cg  == max_gray):
                break

    with open(out_json, 'w') as outfile:
        json.dump(dat_new, outfile, indent=4)


def add_domain_coco(imgPath, json_name, out_json, image_partition_by_domain = None,
                    image_partition_by_string = None, default_domain = 0):
    #load json
    with open(json_name, 'r') as f:
        dat = json.load(f)

    dat_new = dat #TODO: deep copy
    # go through images and add "domain_id"

    for im in dat["images"]:
        domain_id = default_domain
        im_name = im["file_name"]
        if(image_partition_by_domain is None):
            domain_id = default_domain
        else:
            ND = len(image_partition_by_domain)
            for d in range(ND):
                images = image_partition_by_domain[d]
                if(im_name in images):
                    domain_id = d
                    break
        if (image_partition_by_string is None):
            domain_id = default_domain
        else:
            NS = len(image_partition_by_string)
            for d,s in enumerate(image_partition_by_string):
                if(len(s)<3):
                    continue
                if(s in im_name):
                    domain_id = d
                    break

        im["domain_id"] = domain_id


    with open(out_json, 'w') as outfile:
        json.dump(dat_new, outfile,indent=4)

if __name__=="__main__":
    if(0):
        imgPath="/home/borisef/datasets/traffic/set_500_domain_RGB/"
        jsonName="/home/borisef/datasets/traffic/set_500_domain_RGB/data.json"
        out_jsonName = "/home/borisef/datasets/traffic/set_500_domain_RGB/data_da.json"

        add_domain_coco(imgPath = imgPath, json_name = jsonName,
                        out_json = out_jsonName,
                        image_partition_by_domain=None,
                        default_domain=0)

        imgPath = "/home/borisef/datasets/traffic/set_500_domain_Gray/"
        jsonName = "/home/borisef/datasets/traffic/set_500_domain_Gray/data.json"
        out_jsonName = "/home/borisef/datasets/traffic/set_500_domain_Gray/data_da.json"

        add_domain_coco(imgPath=imgPath, json_name=jsonName,
                        out_json=out_jsonName,
                        image_partition_by_domain=None,
                        default_domain=1)

    if(0):
        imgPath = "/home/borisef/datasets/traffic/set_500_mixed_domains"
        jsonName = "/home/borisef/datasets/traffic/set_500_mixed_domains/data.json"
        out_jsonName = "/home/borisef/datasets/traffic/set_500_mixed_domains/data_1.json"
        replace_images_in_json(imgPath=imgPath, json_name=jsonName,
                        out_json=out_jsonName, max_gray = 250)

    if(1):
        imgPath = "/home/borisef/datasets/traffic/set_500_mixed_domains"
        jsonName = "/home/borisef/datasets/traffic/set_500_mixed_domains/data_1.json"
        out_jsonName = "/home/borisef/datasets/traffic/set_500_mixed_domains/data_da.json"

        add_domain_coco(imgPath=imgPath, json_name=jsonName,
                        out_json=out_jsonName,
                        image_partition_by_domain=None,
                        image_partition_by_string=['', '_gray'],
                        default_domain=0)