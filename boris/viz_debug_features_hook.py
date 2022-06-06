import shutil

from mmcv.runner import HOOKS, Hook
import torch, torchvision
import numpy as np
import os, sys
import cv2


from torch.utils.tensorboard import SummaryWriter

cross = np.array([[0,0,1,0,0],
         [0,0,1,0,0],
         [1,1,1,1,1],
         [0,0,1,0,0],
         [0,0,1,0,0]], dtype=np.uint8)
iks = np.array([[1,0,0,0,1],
         [0,1,0,1,0],
         [0,0,1,0,0],
         [0,1,0,1,0],
         [1,0,0,0,1]], dtype=np.uint8)
fram = np.array([[0,0,0,0,0],
         [0,1,1,1,0],
         [0,1,1,1,0],
         [0,1,1,1,0],
         [0,0,0,0,0]], dtype=np.uint8)
ribua = np.ones((5,5), dtype=np.uint8)
diag = np.array([[0,0,0,0,0],
                 [1,0,0,0,0],
                 [1,1,0,0,0],
                 [1,1,1,0,0],
                 [1,1,1,1,0]], dtype=np.uint8)
cross1 = 1.0 - cross
eye = np.eye(5)
set_shapes = (cross, ribua, iks,eye, cross1, fram, diag)

def f1(arr):
    arr = arr*255
    arr = cv2.resize(arr,dsize = (10,10), interpolation=cv2.INTER_NEAREST)
    x1 = np.expand_dims(arr, axis= 0)
    x2 = np.repeat(x1, repeats=1, axis=0)
    return np.expand_dims(x2, axis = 0)

# create images by shape + color label_img: (N, 3, 20,20)
def create_label_images(labels, extra_labels, num_classes, extra_num_classes):
    aa = []
    for el in extra_labels:
        ii = np.mod(el[0], len(set_shapes))
        temp = f1(set_shapes[int(ii)]) #shape
        aa.append(temp.copy())
    label_images = torch.tensor(np.concatenate(aa, axis = 0), dtype=torch.uint8)

    return label_images

# prototype class
class VizDebugFeaturesHookAbstract(Hook):
    def __init__(self, model_type='Unknown', log_folder = None, with_writer = False):
        self.log_folder = log_folder
        self.model_type = model_type
        self.active = False
        self.with_writer = with_writer
        self.init_log_files()

    def get_model_by_type(self, runner):
        raise NotImplementedError("set_debug_results_in_model is abstract")
        return None

    def init_debug_results_in_model_before_run(self, runner):
        mo = self.get_model_by_type(runner)
        if(mo is not None):
            mo.debug_results = []
            mo.keep_debug_results = True
            self.active = True
            self.model = mo

    def init_log_files(self):
        if(self.log_folder is None):
            return
        if (os.path.exists(self.log_folder)):
            shutil.rmtree(self.log_folder)
        os.mkdir(self.log_folder)

        if(self.with_writer):
            self.writer = SummaryWriter(self.log_folder)

    def before_run(self, runner):
       self.init_debug_results_in_model_before_run(runner)

    def after_run(self, runner):
        if(hasattr(self, 'writer')):
            self.writer.close()


@HOOKS.register_module()
class VizDebugFeaturesHookStandardRoIHead(VizDebugFeaturesHookAbstract):
    def __init__(self, num_classes, log_folder, max_per_class=100, class_names=[],
                 epochs=None):
        super(VizDebugFeaturesHookStandardRoIHead, self).__init__(model_type='StandardRoIHead',
                                                                  log_folder = log_folder,
                                                                  with_writer = True)
        self.num_classes = num_classes
        self.max_per_class = max_per_class
        self.log_folder = log_folder
        self.epochs = epochs
        self.class_names = class_names
        self.count_labels = np.zeros(self.num_classes + 1)


    def get_model_by_type(self, runner):#have to reimplement
        s = str(type(runner.model.module.roi_head))
        if (self.model_type in s):
            if('Extra' not in s):
                return runner.model.module.roi_head

        return None

    def before_run(self, runner):
        super(VizDebugFeaturesHookStandardRoIHead, self).before_run(runner)
        # set flags make empty arrays
        self.features = None
        self.labels = np.empty(0)


    def after_train_iter(self, runner):
        if (self.active == False):
            return
        if (self.epochs is not None):
            if (runner.epoch not in self.epochs):
                return

        debug_results = self.model.debug_results
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


    def after_train_epoch(self, runner):
        if (self.active == False):
            return
        if (self.epochs is not None):
            if (runner.epoch not in self.epochs):
                return
        # log embeddings, update tensorboard with self.features and self.labels
        self.writer.add_embedding(self.features, metadata=self.labels, global_step=runner.epoch)



@HOOKS.register_module()
class VizDebugFeaturesHookStandardRoIHeadWithExtraBBoxHead(VizDebugFeaturesHookAbstract):
    def __init__(self, num_classes, log_folder, max_per_class=100, class_names=[[],[]],
                 epochs=None, log_lambda_to_runner = True):
        super(VizDebugFeaturesHookStandardRoIHeadWithExtraBBoxHead, self).__init__(model_type='StandardRoIHeadWithExtraBBoxHead',
                                                                  log_folder=log_folder,
                                                                  with_writer=True)
        self.num_classes = num_classes[0]
        self.num_extra_classes = num_classes[1]
        self.max_per_class = max_per_class
        self.log_folder = log_folder
        self.epochs = epochs
        self.class_names = class_names[0]
        self.extra_class_names = class_names[1]
        self.count_labels = np.zeros(self.num_classes + 1)
        self.log_lambda_to_runner = log_lambda_to_runner

    def get_model_by_type(self, runner):
        s = str(type(runner.model.module.roi_head))
        if (self.model_type in s):
            return runner.model.module.roi_head

    def before_run(self, runner):
        super(VizDebugFeaturesHookStandardRoIHeadWithExtraBBoxHead, self).before_run(runner)
        # set flags make empty arrays
        self.features = None
        self.labels = np.empty(0)
        self.extra_labels = np.empty(0)


    def after_train_iter(self, runner):
        if (self.active == False):
            return

        if(self.log_lambda_to_runner):
            temp = self.model.last_grl_lambda
            runner.log_buffer.update({'da/lambda': temp})

        if (self.epochs is not None):
            if (runner.epoch not in self.epochs):
                return

        debug_results = self.model.debug_results
        N=debug_results['features'].shape[0]
        debug_results['features'] = np.reshape(debug_results['features'],(N,-1))
        M = debug_results['features'].shape[1]
        if(self.features is None):
            self.features=np.empty((0,M))
            self.labels = np.empty((0, 1))
            self.extra_labels = np.empty((0, 1))
        for i in range(N):
            lab = debug_results['labels'][i]
            extra_lab = debug_results['extra_labels'][i]
            if(self.count_labels[lab]<self.max_per_class):
                self.count_labels[lab] = self.count_labels[lab]+1
                #append lab, debug_results['features'][i,:]
                self.features=np.append(self.features, np.array([debug_results['features'][i,:]]), axis=0)
                self.labels = np.append(self.labels, [[lab]], axis=0)
                self.extra_labels = np.append(self.extra_labels, [[extra_lab]], axis=0)

    def after_train_epoch(self, runner):
        if (self.active == False):
            return
        if (self.epochs is not None):
            if (runner.epoch not in self.epochs):
                return
        # log embeddings, update tensorboard with self.features and self.labels
        label_images = create_label_images(self.labels, self.extra_labels, self.num_classes, self.num_extra_classes)
        self.writer.add_embedding(self.features, metadata=self.labels, label_img=label_images,
                                        global_step=runner.epoch)




# @HOOKS.register_module()
# class VizDebugFeaturesHook(Hook):
#
#     def __init__(self, num_classes, log_folder , max_per_class = 100, class_names = [], model_type = 'StandardRoIHead', epochs = None):
#         self.num_classes=num_classes
#         self.max_per_class=max_per_class
#         self.log_folder = log_folder
#         self.model_type = model_type
#         self.epochs = epochs
#         self.class_names = class_names
#
#         #only StandardRoIHeadWithExtraBBoxHead
#         if(model_type == 'StandardRoIHeadWithExtraBBoxHead'):
#             if(isinstance(class_names,list) and len(class_names)>1):
#                 self.extra_class_names = class_names[1]
#                 self.class_names = class_names[0]
#             if (isinstance(num_classes, list) and len(num_classes) > 1):
#                 self.extra_num_classes = num_classes[1]
#                 self.num_classes = num_classes[0]
#             if (isinstance(log_folder, list) and len(log_folder) > 1):
#                 self.extra_log_folder = log_folder[1]
#                 self.log_folder = log_folder[0]
#                 if (os.path.exists(self.extra_log_folder)):
#                     shutil.rmtree(self.extra_log_folder)
#                 os.mkdir(self.extra_log_folder)
#                 self.extra_writer = SummaryWriter(self.extra_log_folder)
#
#
#
#         if(os.path.exists(self.log_folder)):
#             shutil.rmtree(self.log_folder)
#         os.mkdir(self.log_folder)
#
#         self.writer = SummaryWriter(self.log_folder)
#         self.count_labels = np.zeros(self.num_classes + 1)
#
#
#     def before_run(self, runner):
#         # set flags make empty arrays
#         self.features = None
#         self.labels = np.empty(0)
#         self.extra_labels = np.empty(0)
#         self.active = False
#         if(self.model_type == 'StandardRoIHead' or self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):
#             s = str(type(runner.model.module.roi_head))
#             if (self.model_type in s):
#                 runner.model.module.roi_head.keep_debug_results = True
#                 self.active = True
#
#
#
#
#     def after_train_epoch(self, runner):
#         if(self.active == False):
#             return
#         if(self.epochs is not None):
#             if(runner.epoch not in self.epochs):
#                 return
#
#         #update tensorboard with self.features and self.labels
#
#         # get the class labels for each image
#         #class_labels = [classes[lab] for lab in labels]
#
#         # log embeddings
#         #features = images.view(-1, 28 * 28)
#         # writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))
#         if (self.model_type == 'StandardRoIHead'):
#             self.writer.add_embedding(self.features, metadata=self.labels, global_step= runner.epoch)
#
#         #create images by shape + color label_img: (N, 3, 20,20)
#         if (self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):
#             label_images = create_label_images(self.labels, self.extra_labels, self.num_classes, self.extra_num_classes)
#             self.extra_writer.add_embedding(self.features, metadata=self.labels, label_img = label_images,
#                                             global_step= runner.epoch)
#
#
#     def after_train_iter(self, runner):
#         #runner.outputs['log_vars']['aaa'] = 42 #TEMP
#
#         if (self.active == False):
#             return
#         if (self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):#TODO: should it be in this hook ?
#             temp = runner.model.module.roi_head.last_grl_lambda
#             runner.log_buffer.update({'da/lambda': temp})
#         if (self.model_type == 'StandardRoIHead'
#         or self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):
#             debug_results = runner.model.module.roi_head.debug_results
#             N=debug_results['features'].shape[0]
#             debug_results['features'] = np.reshape(debug_results['features'],(N,-1))
#             M = debug_results['features'].shape[1]
#             if(self.features is None):
#                 self.features=np.empty((0,M))
#                 self.labels = np.empty((0, 1))
#                 if (self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):
#                     self.extra_labels = np.empty((0, 1))
#             for i in range(N):
#                 lab = debug_results['labels'][i]
#                 if (self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):
#                     extra_lab = debug_results['extra_labels'][i]
#
#                 if(self.count_labels[lab]<self.max_per_class):
#                     self.count_labels[lab] = self.count_labels[lab]+1
#                     #append lab, debug_results['features'][i,:]
#                     self.features=np.append(self.features, np.array([debug_results['features'][i,:]]), axis=0)
#                     self.labels = np.append(self.labels, [[lab]], axis=0)
#
#                     if(self.model_type == 'StandardRoIHeadWithExtraBBoxHead'):
#                         self.extra_labels = np.append(self.extra_labels, [[extra_lab]], axis=0)
#
#
#
#     def after_run(self, runner):
#         self.writer.close()
#         if hasattr(self, "extra_writer"):
#             self.extra_writer.close()