import json
import os, random
import shutil

def split_coco(imgPath, json_name,outPath1, outPath2, small_size=100):
    #load json
    with open(json_name, 'r') as f:
        dat = json.load(f)

    if(os.path.exists(outPath1)):
        shutil.rmtree(outPath1)
    os.mkdir(outPath1)
    if (os.path.exists(outPath2)):
        shutil.rmtree(outPath2)
    os.mkdir(outPath2)

    #select N random
    dat_small = dat.copy()

    dat_small['images']=[]
    dat_small['annotations']=[]
    dat_big = dat.copy()
    dat_big['images'] = []
    dat_big['annotations'] = []

    N = len(dat['images'])
    s = random.sample(range(N), small_size)
    ids = []

    for i in range(N):
        im = dat['images'][i]
        if(i in s):
            dat_small['images'].append(im)
            outPath = outPath2
            ids.append(im['id'])
        else:
            dat_big['images'].append(im)
            outPath = outPath1

        #add to small
        shutil.copy(os.path.join(imgPath,im['file_name']), outPath)
        #add to json


    # add annotations
    for a in dat['annotations']:
        if (a['image_id'] in ids):
            dat_small['annotations'].append(a)
        else:
            dat_big['annotations'].append(a)

    with open(os.path.join(outPath2,'data.json'), 'w') as outfile:
        json.dump(dat_small, outfile,indent=4)
    with open(os.path.join(outPath1, 'data.json'), 'w') as outfile:
        json.dump(dat_big, outfile,indent=4)

if __name__=="__main__":
    imgPath="/home/borisef/datasets/SelfDrivingCar.v3-fixed-small.coco/export/"
    jsonName="/home/borisef/datasets/SelfDrivingCar.v3-fixed-small.coco/export/_annotations.coco.json"
    outPath1 = "/home/borisef/datasets/SelfDrivingCar.v3-fixed-small.coco/temp"
    outPath2 = "/home/borisef/datasets/SelfDrivingCar.v3-fixed-small.coco/set_50"
    split_coco(imgPath=imgPath, json_name=jsonName, small_size=50,outPath1=outPath1,outPath2=outPath2)

    imgPath = outPath1
    jsonName = os.path.join(imgPath,"data.json")
    outPath1 = "/home/borisef/datasets/SelfDrivingCar.v3-fixed-small.coco/temp1"
    outPath2 = "/home/borisef/datasets/SelfDrivingCar.v3-fixed-small.coco/set_100"
    split_coco(imgPath=imgPath, json_name=jsonName, small_size=100, outPath1=outPath1, outPath2=outPath2)

    imgPath = outPath1
    jsonName = os.path.join(imgPath,"data.json")
    outPath1 = "/home/borisef/datasets/SelfDrivingCar.v3-fixed-small.coco/temp2"
    outPath2 = "/home/borisef/datasets/SelfDrivingCar.v3-fixed-small.coco/set_300"
    split_coco(imgPath=imgPath, json_name=jsonName, small_size=300, outPath1=outPath1, outPath2=outPath2)

    imgPath = outPath1
    jsonName = os.path.join(imgPath, "data.json")
    outPath1 = "/home/borisef/datasets/SelfDrivingCar.v3-fixed-small.coco/temp3"
    outPath2 = "/home/borisef/datasets/SelfDrivingCar.v3-fixed-small.coco/set_500"
    split_coco(imgPath=imgPath, json_name=jsonName, small_size=500, outPath1=outPath1, outPath2=outPath2)
