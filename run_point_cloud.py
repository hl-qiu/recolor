#%%
import os
import sys
# sys.path.insert(0, '../')
import torch
import numpy as np
import imageio

from matplotlib import pyplot as plt

# try:
#     import piplite
#     await piplite.install(['ipywidgets'])
# except ImportError:
#     pass
# import ipywidgets as widgets
#%%
from engine.trainer import Trainer
from engine.eval import evaluation_path
from processing_point.get_point_cloud import write_point_cloud,read_point_cloud,write_point_cloud_with_color_decomposition,write_opaque_with_color_decomposition
from data import dataset_dict
from utils.opt import config_parser

#%%
path_redirect = [
    # option name, path in the config, redirected path
    ('palette_path', '../data_palette', './data_palette')
]
#%%
run_dir = './logs/materials'
ckpt_path = None
out_dir = os.path.join(run_dir,'demo_out')
#%%
#读取数据
def read_data(dataset_type='train'):
    parser = config_parser()
    config_path = os.path.join(run_dir,'args.txt')

    if os.path.exists(config_path):
        with open(config_path,'r') as f:
            args,remainings = parser.parse_known_args(args=[],config_file_contents=f.read())

            if ckpt_path is not None:
                setattr(args,'ckpt',ckpt_path)

            for entry in path_redirect:
                setattr(args,entry[0],getattr(args,entry[0]).replace(entry[1],entry[2]))

            print("Args loaded:", args)
    else:
        print(f"ERROR : cannot read args in {run_dir}.")
    print()

    dataset = dataset_dict[args.dataset_name]
    # train_dataset
    # test_dataset
    if dataset_type =='train':
        train_dataset = dataset(args.datadir,split='train',downsample=args.downsample_train,is_stack=True)
        return args,train_dataset
    else:
        test_dataset = dataset(args.datadir,split='test',downsample=args.downsample_test, is_stack=True)
        return args,test_dataset

#%%
from processing_point.color_decomposition import color_decomposition



def write_pointcloud(dataset_type='train',true_choice=[0]):

    #读取数据
    args,dataset = read_data(dataset_type=dataset_type,)

    print("Initializing trainer and model...")
    ckpt_dir = os.path.join(run_dir,"checkpoints")
    tb_dir = os.path.join(run_dir,"tensorboard")

    trainer = Trainer(args,run_dir,ckpt_dir, tb_dir)

    model = trainer.build_network()
    model.eval()
    print()

    #调色板提取
    palette_prior = trainer.palette_prior.detach().cpu().numpy()
    palette = model.renderModule.palette.get_palette_array().detach()

    print("==============*****************==================")
    "write point cloud"
    # write_point_cloud_with_color_decomposition(dataset, model, args, trainer.renderer,eps=0.2, savePath=None, N_vis=2, N_samples=-1, white_bg=False,
    #            ndc_ray=False, palette=palette, new_palette=None,device='cuda',filename=None)

    "write opaque true idx"
    write_opaque_with_color_decomposition(dataset, model, args, true_choice=true_choice,renderer=trainer.renderer, eps=0.2, savePath=None, N_vis=-1,
                                               N_samples=-1, white_bg=True,
                                               ndc_ray=False, palette=palette, new_palette=None, device='cuda',
                                               filename=None)

#%%
#写点云
write_pointcloud('train',[1,2])
#%%

#读点云
# read_point_cloud("./logs/point_cloud","point_clouds.txt")


#write point in line with color decomposition

