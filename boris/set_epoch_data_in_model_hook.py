
from mmcv.runner import HOOKS, Hook
import torch, torchvision
import numpy as np
import os, sys
import cv2

from matplotlib import pyplot as plt

@HOOKS.register_module()
class SetEpochDataInModelHook(Hook):
    #change StandardRoIHeadWithExtraBBoxHead

    def __init__(self, submodule = None, jump = 1):
        #TODO: find model and keep reference
        self.submodule = submodule
        self.jump = jump
        self.params = None

        pass

    def before_run(self, runner):
        if(self.submodule == "roi_head.extra_head_lambda_params"):
            self.params = runner.model.module.roi_head.extra_head_lambda_params
        else:
            raise NotImplementedError
        self.params['internal_epoch_counter'] = False
        self.params['curr_epoch'] = 0


    def before_epoch(self, runner):
       #set epoch
       if (self.submodule == "roi_head.extra_head_lambda_params"):
           self.params['curr_epoch'] = (runner.epoch + 1) * self.jump
       else:
           raise NotImplementedError



