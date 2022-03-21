import shutil

from mmcv.runner import HOOKS, Hook
import torch, torchvision
import numpy as np
import os, sys
import cv2
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter


@HOOKS.register_module()
class VizDebugFeaturesHook(Hook):

    def __init__(self, num_classes, log_folder , max_per_class = 100, model_type = 'StandardRoIHead', epochs = None):
        self.num_classes=num_classes
        self.max_per_class=max_per_class
        self.log_folder = log_folder
        self.model_type = model_type
        self.count_labels = np.zeros(num_classes+1)
        self.epochs = epochs



        if(os.path.exists(log_folder)):
            shutil.rmtree(log_folder)
        os.mkdir(log_folder)

        self.writer = SummaryWriter(log_folder)





    def before_run(self, runner):
        # set flags make empty arrays
        self.features = None
        self.labels = np.empty(0)
        self.active = False
        if(self.model_type == 'StandardRoIHead'):
            s = str(type(runner.model.module.roi_head))
            if ('.StandardRoIHead' in s):
                runner.model.module.roi_head.keep_debug_results = True
                self.active = True
        pass

    def after_train_epoch(self, runner):
        if(self.active == False):
            return
        if(self.epochs is not None):
            if(runner.epoch not in self.epochs):
                return
        #TODO: update tensorboard with self.features and self.labels

        # get the class labels for each image
        #class_labels = [classes[lab] for lab in labels]

        # log embeddings
        #features = images.view(-1, 28 * 28)
        # writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))
        self.writer.add_embedding(self.features, metadata=self.labels, global_step= runner.epoch)


        pass

    def after_train_iter(self, runner):
        if (self.active == False):
            return
        #TODO: coppy arrays
        if (self.model_type == 'StandardRoIHead'):
            debug_results = runner.model.module.roi_head.debug_results
            N=debug_results['features'].shape[0]
            debug_results['features'] = np.reshape(debug_results['features'],(N,-1))
            M = debug_results['features'].shape[1]
            if(self.features is None):
                self.features=np.empty((0,M))
                self.labels = np.empty((0, 1))
            for i in range(N):
                lab = debug_results['labels'][i]
                if(self.count_labels[lab]<self.max_per_class):
                    self.count_labels[lab] = self.count_labels[lab]+1
                    #append lab, debug_results['features'][i,:]
                    self.features=np.append(self.features, np.array([debug_results['features'][i,:]]), axis=0)
                    self.labels = np.append(self.labels, [[lab]], axis=0)



    def after_run(self, runner):
        self.writer.close()  # ???