
from mmcv.runner import HOOKS, Hook
import torch, torchvision
import numpy as np
import os, sys
import cv2

from matplotlib import pyplot as plt



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

def VizObjectness(objectness, img, outpath):
    numScales = len(objectness)
    H, W = objectness[0].shape[2], objectness[0].shape[3]
    outArr = np.zeros((H,W))
    imgnp = img.data.cpu().numpy().copy()
    imgnp = np.transpose(imgnp, (1, 2, 0))

    #numScales = 1
    for sc in range(numScales):

        numAnchors = objectness[sc].shape[1]
        for a in range(numAnchors):
            t = objectness[sc][0,a,:,:]
            npt = torch.sigmoid(t.data.cpu()).numpy().copy()
            #resize to H,W
            nptBig = cv2.resize(npt,(W,H),interpolation=cv2.INTER_CUBIC)
            outArr = np.maximum(nptBig, outArr)

    imgnp1 = cv2.resize(imgnp,(W,H),interpolation=cv2.INTER_CUBIC)
    imgnp1 = imgnp1 - imgnp1.min()
    imgnp1 = imgnp1 / imgnp1.max()

    imgnp1[:, :, 2] = np.minimum(0.5*imgnp1[:, :, 2] + 0.8*outArr, 1.0)  # TODO overlay map

    imgnp1 = (imgnp1*255).astype(np.uint8)
    outArr = (outArr * 255.0).astype(np.uint8)

    cv2.imwrite(outpath, imgnp1)
    pass

def VizGraph(model, dict_params):
    from torchviz import make_dot
    import hiddenlayer as hl

    from torch.utils.tensorboard import SummaryWriter

    x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False)
    transforms = [hl.transforms.Prune('Constant')]  # Removes Constant nodes from graph.
    graph = hl.build_graph(model, x, transforms=transforms)
    graph.save('rnn_hiddenlayer', format='png')

    graph1 = hl.build_graph(model, x)
    graph1.theme = hl.graph.THEMES['blue'].copy()
    graph1.save('rnn_hiddenlayer1', format='png')

    out = model(x)
    make_dot(out).render("model", format="png")  # plot graph of variable, not of a nn.Module
    make_dot(out, params=dict(list(resnet.named_parameters()))).render("model_1", format="png")  # plot graph of variable, not of a nn.Module

    #TODO: show graph in tensorboard
    if('tb_writer' in dict_params):
        writer = dict_params['tb_writer']
        writer.add_graph(model)
        writer.close()

    #TODO: save graph in png

    #TODO: show any submodule
    pass


@HOOKS.register_module()
class ExperimentalHook(Hook):

    def __init__(self, a, b, outDir):
        self.a = a
        self.b = b
        self.keepFilters=[]
        self.keepObjectness=[]
        self.outDir = outDir
        if(not os.path.exists(outDir)):
            os.mkdir(outDir)
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):

        t = runner.model.module.backbone.conv1.weight.clone()
        t1 = t.cpu().data.numpy().copy()
        self.keepFilters.append(t1)


        arr = visTensor(t.cpu(), ch=0, allkernels=False)
        outName = os.path.join(self.outDir , str(runner.epoch) + ".jpg")
        plt.imsave(outName, arr)
        # plt.axis('off')
        # plt.ioff()
        # plt.show()
        #
        # plt.close('all')
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        #TODO: grab objectness (1st tensor from outs)  and output in plot
        pass

    def after_iter(self, runner):
        if(hasattr(runner.model.module,'last_objectness')):
            #generate plot
            objectness = runner.model.module.last_objectness
            img = runner.model.module.last_image
            #outName = os.path.join(self.outDir, 'objectness_' + str(runner.epoch) +'_'+ str(runner.iter) + ".jpg")
            outName = os.path.join(self.outDir, 'objectness_' + str(runner.epoch) + ".jpg")
            VizObjectness(objectness, img, outName)
            pass
        pass


