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

    def __init__(self, num_classes, log_folder , max_per_class = 100, class_names = [], model_type = 'StandardRoIHead', epochs = None):
        self.num_classes=num_classes
        self.max_per_class=max_per_class
        self.log_folder = log_folder
        self.model_type = model_type
        self.epochs = epochs
        self.class_names = class_names

        #only StandardRoIHeadWithExtraBBoxHead
        if(model_type == 'StandardRoIHeadWithExtraBBoxHead'):
            if(isinstance(class_names,list) and len(class_names)>1):
                self.extra_class_names = class_names[1]
                self.class_names = class_names[0]
            if (isinstance(num_classes, list) and len(num_classes) > 1):
                self.extra_num_classes = num_classes[1]
                self.num_classes = num_classes[0]
            if (isinstance(log_folder, list) and len(log_folder) > 1):
                self.extra_log_folder = log_folder[1]
                self.log_folder = log_folder[0]
                if (os.path.exists(self.extra_log_folder)):
                    shutil.rmtree(self.extra_log_folder)
                os.mkdir(self.extra_log_folder)
                self.extra_writer = SummaryWriter(self.extra_log_folder)



        if(os.path.exists(self.log_folder)):
            shutil.rmtree(self.log_folder)
        os.mkdir(self.log_folder)

        self.writer = SummaryWriter(self.log_folder)
        self.count_labels = np.zeros(self.num_classes + 1)


    def before_run(self, runner):
        # set flags make empty arrays
        self.features = None
        self.labels = np.empty(0)
        self.extra_labels = np.empty(0)
        self.active = False
        if(self.model_type == 'StandardRoIHead' or self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):
            s = str(type(runner.model.module.roi_head))
            if (self.model_type in s):
                runner.model.module.roi_head.keep_debug_results = True
                self.active = True




    def after_train_epoch(self, runner):
        if(self.active == False):
            return
        if(self.epochs is not None):
            if(runner.epoch not in self.epochs):
                return

        #update tensorboard with self.features and self.labels

        # get the class labels for each image
        #class_labels = [classes[lab] for lab in labels]

        # log embeddings
        #features = images.view(-1, 28 * 28)
        # writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))
        self.writer.add_embedding(self.features, metadata=self.labels, global_step= runner.epoch)

        if (self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):
            self.extra_writer.add_embedding(self.features, metadata=self.extra_labels, global_step= runner.epoch)


    def after_train_iter(self, runner):
        if (self.active == False):
            return
        #TODO: coppy arrays
        if (self.model_type == 'StandardRoIHead'
        or self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):
            debug_results = runner.model.module.roi_head.debug_results
            N=debug_results['features'].shape[0]
            debug_results['features'] = np.reshape(debug_results['features'],(N,-1))
            M = debug_results['features'].shape[1]
            if(self.features is None):
                self.features=np.empty((0,M))
                self.labels = np.empty((0, 1))
                if (self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):
                    self.extra_labels = np.empty((0, 1))
            for i in range(N):
                lab = debug_results['labels'][i]
                if (self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):
                    extra_lab = debug_results['extra_labels'][i]

                if(self.count_labels[lab]<self.max_per_class):
                    self.count_labels[lab] = self.count_labels[lab]+1
                    #append lab, debug_results['features'][i,:]
                    self.features=np.append(self.features, np.array([debug_results['features'][i,:]]), axis=0)
                    self.labels = np.append(self.labels, [[lab]], axis=0)

                    if(self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):
                        self.extra_labels = np.append(self.extra_labels, [[extra_lab]], axis=0)



    def after_run(self, runner):
        self.writer.close()
        if hasattr(self, "extra_writer"):
            self.extra_writer.close()