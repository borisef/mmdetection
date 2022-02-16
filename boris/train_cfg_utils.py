import argparse
import os
from mmcv import Config


def replace_val_in_config(cfg, nam, val):
    #nam is like aaa.bbb.ccc find it in the cfg and replace by val
    nams = nam.split('.')
    entry_point = cfg
    if(len(nams) > 1):
        gotcha = True
        for i, na in enumerate(nams):
            if(na in entry_point):
                if(i < len(nams)-1):
                    entry_point = entry_point[na]
                else:
                    entry_point[na] = val
                    print('Changed:' + nam)
            else:
                gotcha = False
                break
        # if(gotcha):
        #     entry_point=val
    else:#user defined
        pass

    return cfg



def print_param(args):
    kwargs = args._get_kwargs()
    print("---")
    for k in kwargs:
        print(k)


class ArgConfigParams():
    def arguments_default(self):
        self.parser.add_argument('--batch-size', type=int, default=None, metavar='N',
                            help='input batch size for training (default: None)')

        self.parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: None)')

        self.parser.add_argument('--some_str', type=str, default=None,
                            help='str')

        self.parser.add_argument('--isbool', type=bool, default=False,
                            help='boolean only')

        self.parser.add_argument('--mytry.aa', type=int, default=[0, 0, 0], nargs='+',
                            help='boolean')
    def arguments_cfg(self):
        self.parser.add_argument('--config_file', type=str, default=None, metavar='N',
                                 help='config_file (default: None)')

        self.parser.add_argument('--model.backbone.frozen_stages', type=int, default=None, metavar='N',
                            help='frozen stages (default: None)')

        self.parser.add_argument('--model.rpn_head.anchor_generator.scales', type=int, default=None, metavar='N', nargs='+',
                            help='number of epochs to train (default: None)')

        self.parser.add_argument('--model.neck.type', type=str, default=None, metavar='N',
                                 help='FPN, PAFPN')

        self.parser.add_argument('--model.train_cfg.rpn_proposal.max_per_img', type=int, default=None, metavar='N',
                                 help='max ')
        self.parser.add_argument('--model.train_cfg.rpn.pos_weight', type=float, default=None, metavar='N',
                                 help='pos weight rpn ')
        self.parser.add_argument('--model.train_cfg.rcnn.pos_weight', type=float, default=None, metavar='N',
                                 help='pos weight rpn ')
        self.parser.add_argument('--optimizer.weight_decay', type=float, default=None, metavar='N',
                                 help='optimizer.weight_decay')




    def __init__(self):
        self.parser = argparse.ArgumentParser(description='cnvrg-io arguments')




    def get_known_args(self, exclude_default = True):
        args_all, temp = self.parser.parse_known_args()
        kwargs = args_all._get_kwargs()
        out = []
        if(exclude_default):
            for k in kwargs:
                nm = k[0]
                val = k[1]
                if(self.parser.get_default(nm) != val):
                    out.append(k)
        else:
            out = kwargs.copy()
        return out

def update_config_from_args(cfg,cp):
    new_params = cp.get_known_args()
    for name_value in new_params:
        cfg=replace_val_in_config(cfg,name_value[0], name_value[1])

    return cfg

#tests
def check1():
    cp = ArgConfigParams()
    cp.arguments_cfg()
    aa = cp.get_known_args()
    print(aa)
    print("OK1")

def check2():
    cfg_file = 'configs/car_damage_config.py'
    cfg = Config.fromfile(cfg_file)
    a = ('model.backbone.frozen_stages', 1)
    cfg = replace_val_in_config(cfg, a[0],a[1])
    b = ('model.rpn_head.anchor_generator.scales', [4,8,32])
    cfg = replace_val_in_config(cfg, b[0], b[1])
    print("OK")

    pass
def check3():
    cp = ArgConfigParams()
    cp.arguments_cfg()
    new_params = cp.get_known_args()
    print('New params:')
    print(new_params)

    args = cp.parser.parse_args()
    cfg_file = 'configs/car_damage_config.py'
    if(args.config_file is not None):
        cfg_file = args.config_file
    cfg = Config.fromfile(cfg_file)
    cfg = update_config_from_args(cfg, cp)

if __name__ == "__main__":
    #check1()
    check3()