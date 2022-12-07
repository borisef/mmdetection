import json
import os, random
import shutil

class Glob:
    count = 0

def _get_img_pref(folder, index, img_prefix_type):
    if(img_prefix_type is None):
        return ''
    if (img_prefix_type=='auto'):
        return str(10000 + index)[1:] + '_'

    if (img_prefix_type == 'folder_name'):
        temp = os.path.basename(folder)
        if(len(temp)==0):
            temp = os.path.basename(folder[0:-1])# remove filesep at the end of string
        return temp + '_'
    if (img_prefix_type == 'count'):
        Glob.count = Glob.count + 1
        return str(10000 + Glob.count)[1:] + '_'


def combine_multi_folders(image_folders, json_names, out_folder, out_json = 'data.json', img_prefix_type = 'auto'):
    #combine all images and jsons into one folder
    # warning : suppose same categories for all
    #image_folders = list of folders, full path
    #json_names = list of json names, like 'data.json'
    #out_folder = outfolder (if exist will be removed)
    #out_json = list of json names, like 'data.json'
    #img_prefix_type = string one of 'folder_name', 'auto', None
    N = len(image_folders)
    if(isinstance(json_names,str)):
        json_names = [json_names]*N

    if(os.path.exists(out_folder)):
        shutil.rmtree(out_folder)
    os.mkdir(out_folder)


    # load json
    with open(os.path.join(image_folders[0],json_names[0]), 'r') as f:
        dat = json.load(f)

    out_coco = dat.copy()
    # make empty json struct
    out_coco['images'].clear()
    out_coco['annotations'].clear()

    last_image_id = 0
    last_annotation_id = 0
    for ind,pa in enumerate(image_folders):


        #load json
        with open(os.path.join(image_folders[ind], json_names[ind]), 'r') as f:
            dat = json.load(f)

        #copy images and files
        for im in dat['images']:
            # generate img_prefix
            img_pref = _get_img_pref(pa, ind, img_prefix_type)

            old_file_name = im['file_name']
            new_file_name = img_pref + im['file_name']
            old_im_id = im['id']
            new_im_id = last_image_id

            im['file_name']  =new_file_name
            im['id'] = new_im_id
            out_coco['images'].append(im)
            last_image_id = last_image_id + 1

            shutil.copyfile(os.path.join(pa,old_file_name),os.path.join(out_folder,new_file_name))
        #copy annotations
            for ann in dat['annotations']:
                if(ann['image_id'] == old_im_id):
                    old_ann_id = ann['id']
                    new_ann_id = last_annotation_id
                    last_annotation_id = last_annotation_id + 1
                    ann['id'] = new_ann_id
                    ann['image_id'] = new_im_id
                    out_coco['annotations'].append(ann)


    #save json
    with open(os.path.join(out_folder, out_json), 'w') as outfile:
        json.dump(out_coco, outfile,indent=4)



if __name__ == '__main__':

    image_folders = ['/home/borisef/datasets/racoon/valid', '/home/borisef/datasets/racoon/test']
    json_names = ['data.json']*2
    out_folder = '/home/borisef/datasets/try/combine'

    image_folders = ['/home/borisef/datasets/traffic/set_50_domain_Gray',
                     '/home/borisef/datasets/traffic/set_50_domain_RGB']
    json_names = ['data_da.json'] * 2

    combine_multi_folders(image_folders = image_folders,
                          json_names = json_names,
                          out_folder = out_folder,
                          out_json='data.json',
                          img_prefix_type='folder_name')