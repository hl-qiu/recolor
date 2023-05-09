import os.path

import numpy as np

from data.llff import LLFFDataset
from data.blender import BlenderDataset

filedir = "/home/ubuntu/Rencq/nerf_data/nerf_synthetic/materials"
outdir = "../logs/materials/out_image"
# flag = [0, 1, 10, 11, 13, 14, 15, 16, 17, 18, 19, 2, 21, 22, 24, 27, 28, 29, 3, 30, 31, 33, 34, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 46, 47, 48, 49, 5, 50, 51, 52, 54, 55, 57, 58, 59, 6, 60, 61, 62, 63, 64, 65, 66, 68, 69, 7, 70, 71, 72, 73, 74, 76, 77, 78, 79, 8, 80, 81, 82, 83, 84, 85, 87, 88, 89, 91, 92, 93, 94, 95, 97, 98, 99]
flag = [11, 13, 15, 19, 2, 28, 31, 33, 34, 36, 46, 47, 53, 58, 61, 64, 68, 69, 72, 73, 79, 8, 80, 83, 84, 85, 95, 97, 99]
blenderdataset = BlenderDataset(filedir,split='train',downsample=1,spheric_poses=True)
index = 0
for i in flag:
    outpath_pose = os.path.join(outdir,"%08d_cam.txt"%(index))
    tmp = np.array([[0.,0.,0.,1.]])
    poses_tmp = np.reshape(np.concatenate([blenderdataset.poses[i,:3,0],blenderdataset.poses[i,:3,1],blenderdataset.poses[i,:3,2]],-1),(3,3))
    tt = -poses_tmp @ np.reshape(blenderdataset.poses[i,:3,3].numpy(),(3,1))
    poses = np.concatenate([poses_tmp,tt], -1)
    pose_tmp = np.concatenate((poses,tmp),axis=0)
    np.savetxt(outpath_pose,pose_tmp,header='extrinsic',comments='')
    index+=1
index=0
for i in flag:
    outpath_pose = os.path.join(outdir, "%08d_cam.txt"%(index))
    K = [[blenderdataset.focal,0.0,blenderdataset.img_wh[0]/2],[0.,blenderdataset.focal,blenderdataset.img_wh[1]/2],[0.,0.,1.]]
    with open(outpath_pose,"a") as file:
        file.write("\n")
        file.write('intrinsic\n')
        for i in range(len(K)):
            for j in range(len(K[i])):
                file.write(str(K[i][j]))
                if j != len(K[i])-1:
                    file.write(" ")
            file.write("\n")
        file.write("\n")
        d = (blenderdataset.near_far[1]-blenderdataset.near_far[0])/512/1.06
        file.write(str(blenderdataset.near_far[0]))
        file.write(" ")
        file.write(str(d))
    index+=1

# """save nearfar"""
# outpath_nearfar = os.path.join(outdir,f"nearfar.txt")
# np.savetxt(outpath_nearfar,blenderdataset.near_far)
# """save focal"""
# outpath_focal = os.path.join(outdir,f"focal.txt")
# np.savetxt(outpath_focal,blenderdataset.focal)
