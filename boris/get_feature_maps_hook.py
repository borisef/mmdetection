
from mmcv.runner import HOOKS, Hook
import torch, torchvision
import numpy as np
import os, sys
import cv2
import torch.nn.functional as F
import torch.nn as nn



def save_lines(nz, output_file):
    pass

def squeeze_fm(feature_map):
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale / feature_map.shape[0]
    res = gray_scale.data.cpu().numpy()
    return res

def save_im_fm(feature_map, outim):
    fig = plt.figure(figsize=(30, 50))
    plt.imshow(squeeze_fm(feature_map))
    plt.savefig(outim, bbox_inches='tight')




def get_multilevel_attribute(obj, ff):
    re = obj
    for fi in ff.split('.'):
        re = getattr(re,fi)
    return re

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    # plt.figure(figsize=(nrow, rows))
    # plt.imshow(grid.numpy().transpose((1, 2, 0)))
    return grid.numpy().transpose((1, 2, 0))


def save_filters_in_image(t, outName):
    arr = visTensor(t.cpu(), ch=0, allkernels=False)
    plt.imsave(outName, arr)
    plt.close()

from matplotlib import pyplot as plt
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook

@HOOKS.register_module()
class FeatureMapHook(Hook):

    def __init__(self, conv_filters, relu_filters, outDir, imName = None):
        self.conv_filters=conv_filters
        self.relu_filters=relu_filters
        self.outDir = outDir
        self.imName = imName
        self.keep_nz = []

        if(not os.path.exists(outDir)):
            os.mkdir(outDir)
        pass




    def before_run(self, runner):
        #register hooks
        for ff in self.conv_filters:
            t = get_multilevel_attribute(runner.model.module, ff)
            t.register_forward_hook(get_activation(ff))

        for ff in self.relu_filters:
            t = get_multilevel_attribute(runner.model.module, ff)
            t.register_forward_hook(get_activation(ff))
        pass

    def after_train_epoch(self, runner):
        #TODO: only for 1 image

        for ff in self.conv_filters:
            conv_activation = activation[ff]
            out_im_name = os.path.join(self.outDir, str(runner.epoch) + "_" + ff + ".jpg")
            save_im_fm(conv_activation, out_im_name)



        for ff in self.relu_filters:
            relu_activation = activation[ff]
            out_im_name = os.path.join(self.outDir, str(runner.epoch) + "_" + ff + ".jpg")
            non_zeros = float(torch.count_nonzero(relu_activation))
            ratio_nz = non_zeros/np.array(relu_activation.shape).prod()
            save_im_fm(relu_activation, out_im_name)
            self.keep_nz.append([runner.epoch,ratio_nz, ff])
            print(self.keep_nz)

        # TODO: save as im
        save_lines(self.keep_nz, os.path.join(self.outDir, str(runner.epoch) + "_non_zeros" +  ".jpg"))


        pass


