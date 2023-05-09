from .data_io import read_pfm
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import cv2

class depth_dataset(Dataset):
    def __init__(self, datadir, depth_num,poses,split='train', downsample=4, is_stack=False, hold_every=8):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.depth_num = depth_num
        self.poses = poses
        self.root_dir = datadir
        self.split = split
        self.hold_every = hold_every
        self.is_stack = is_stack
        self.downsample = downsample
        self.read_meta()
        self.white_bg = False


    def read_meta(self):
        depth_folder_name = f'depth_{int(self.downsample)}' if self.downsample > 1 else 'depth'
        depth_path = os.path.join(self.root_dir, depth_folder_name)
        mask_path = os.path.join(self.root_dir,"mask")
        if not os.path.exists(depth_path):
            print('warning: images with the specified downsample scale do not exist.')
            image_folder_name = 'depth'

        i_test = np.arange(0, self.depth_num, self.hold_every)  # [np.argmin(dists)]
        # 测试集 训练集分开
        img_list = i_test if self.split != 'train' else list(set(np.arange(self.depth_num)) - set(i_test))

        # use first N_images-1 to train, the LAST is val
        self.all_depth = []

        self.final_mask = []
        for i in img_list:
            filename = os.path.join(depth_path,"%08d.pfm"%(i))
            depth,_ = read_pfm(filename)
            poses_tmp = np.array([[self.poses[i, 0, 2], -self.poses[i, 1, 2], -self.poses[i, 2, 2]]])  # 1,3
            tt = np.array([[self.poses[i, 2, 3]]])  # 1,1
            depth_tmp = np.reshape(depth,(-1,1))
            depth = depth_tmp * poses_tmp
            depth = np.sum(depth,-1).reshape(-1,1)
            depth = depth + tt

            filename2 = os.path.join(mask_path,"%08d_final.png"%(i))
            img = cv2.imread(filename2)
            img_ar = np.array(img,dtype=np.float)
            img_ar = np.sum(img_ar,-1).reshape(-1,)
            img_ar[img_ar==765] = 1.
            img_ar = img_ar.astype(bool)
            self.final_mask +=[torch.tensor(img_ar)] #(h*w,1)
            self.all_depth += [torch.tensor(depth)]  # (h*w, 1)

        if not self.is_stack:
            self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 1)
            self.final_mask = torch.cat(self.final_mask,0)
        else:
            self.all_depth = torch.stack(self.all_depth, 0)  # (len(self.meta['frames]),h*w, 1)
            self.final_mask = torch.stack(self.final_mask,0)


    def __len__(self):
        return len(self.all_depth)

    def __getitem__(self, idx):

        sample = {'depth': self.all_depth[idx]}

        return sample
