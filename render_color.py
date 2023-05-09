# %%
import os
import sys
import cv2
sys.path.insert(0, './')
import torch
import numpy as np
import imageio
import glob
from einops import rearrange
from matplotlib import pyplot as plt


# %%
from engine.trainer import Trainer
from engine.eval import evaluation_path
from data import dataset_dict
from utils.opt import config_parser
from utils.vis import plot_palette_colors, visualize_depth_numpy, visualize_palette_components_numpy
from utils.color import rgb2hex, hex2rgb
from utils.ray import get_rays, ndc_rays_blender
from point_Model.point import point_cloud,point_cloud_classical,point_cloud_classical_num,point_empty


# %% md
## Utils
# %%
def print_divider():
    print()



def render_one_view(test_dataset, tensorf, c2w, renderer, N_samples=-1,
                    white_bg=False, ndc_ray=False, new_palette=None, palette=None, device='cuda',is_choose=False,net1=None,net2=None):
    torch.cuda.empty_cache()

    near_far = test_dataset.near_far

    if palette is None and hasattr(tensorf, 'get_palette_array'):
        palette = tensorf.get_palette_array().cpu()

    # 测试数据图像的宽高
    W, H = test_dataset.img_wh

    # 旋转矩阵
    c2w = torch.FloatTensor(c2w)
    # 根据方向和旋转矩阵 得到光线位置和方向
    rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)

    if ndc_ray:
        rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
    rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
    # render_one_view(ds_test_dataset, model, c2w, trainer.renderer, palette=torch.from_numpy(new_palette),
    # N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=trainer.device)
    # trainer.render 等于 chunkify_render
    res = renderer(rays, tensorf, chunk=2048, N_samples=N_samples, new_palette=new_palette, palette=palette,
                   ndc_ray=ndc_ray, white_bg=white_bg, device=device, ret_opaque_map=True,is_choose=is_choose,net1=net1,net2=net2)

    rgb_map = res['rgb_map']
    depth_map = res['depth_map']

    rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

    rgb_map = (rgb_map.numpy() * 255).astype('uint8')

    depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

    is_vis_plt = (palette is not None) and ('opaque_map' in res)
    plt_decomp = None
    # 获得调色板分解图
    if is_vis_plt:
        opaque = rearrange(res['opaque_map'], '(h w) c-> h w c', h=H, w=W).cpu()
        plt_decomp = visualize_palette_components_numpy(opaque.numpy(), palette.numpy())
        plt_decomp = (plt_decomp * 255).astype('uint8')

    return rgb_map, depth_map, plt_decomp


# %%

# %% md
## Config
# %%
# Make paths accessible by this notebook
path_redirect = [
    # option name, path in the config, redirected path
    ('palette_path', '../data_palette', './data_palette')
]
# %%
run_dir = './logs/materials/'
ckpt_path = None
out_dir = os.path.join(run_dir, 'demo_out')
# Setup trainer
print('Initializing trainer and model...')
ckpt_dir = os.path.join(run_dir, 'checkpoints')
tb_dir = os.path.join(run_dir, 'tensorboard')

'''Modify this to name this editing'''
edit_name = 'test_materials'

print('Run dir:', run_dir)
print('Demo output dir:', out_dir)
# %%
## Load and Setup
# %%
# Read args
parser = config_parser()
# 对args.txt里的参数进行获取
config_path = os.path.join(run_dir, 'args.txt')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        args, remainings = parser.parse_known_args(args=[], config_file_contents=f.read())

        # override ckpt path
        if ckpt_path is not None:
            setattr(args, 'ckpt', ckpt_path)

        # redirect path
        for entry in path_redirect:
            setattr(args, entry[0], getattr(args, entry[0]).replace(entry[1], entry[2]))

        print('Args loaded:', args)
else:
    print(f'ERROR: cannot read args in {run_dir}.')
print_divider()
"""
修改参数
"""

# 训练器
trainer = Trainer(args, run_dir, ckpt_dir, tb_dir)
# 模型
model = trainer.build_network()
model.eval()
print_divider()

# Create downsampled dataset
# dataset = dataset_dict[args.dataset_name]
# ds_test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train * 2., is_stack=True)
# print('Downsampled dataset loaded')

# %% md
## Palette Editing
# %%
def palette_editing():
    palette_prior = np.load('palette_rgb_11.npy')
    palette = model.renderModule.palette.get_palette_array().detach().cpu().numpy()
    palette = palette.clip(0. ,1.)
    palette_prior = palette_prior.clip(0.,1.)
    # %%
    print('Initial palette prior:')
    plot_palette_colors(palette_prior,'palette_prior_image.jpg')
    print(palette)
    plot_palette_colors(palette,'palette_image.jpg')
    print(palette_prior)
    # %%

    palette_num = palette.shape[0]

    palette_max = 50

    palette_color = np.load('./data_palette/fern/rgb_palette.npy')
    print("palette_color = \n",palette_color)
    plot_palette_colors(palette_color,'palette_color.jpg')

    # new_palette = torch.ones((palette_num,palette_max,palette_num,3))
    # new_palette[...,:,:] = torch.tensor(palette)
    # print("=====> new palette shape")
    # print(new_palette.shape)
    # print("====> new palette\n")
    # print(new_palette)
    #
    # # %%
    # print('Optimized palette:')
    # new_palette = palette.clip(0., 1.)
    #
    # # new_palette = new_palette.clip(0.5, 0.7)
    # print(new_palette)
    # plot_palette_colors(new_palette,'new_palette_image.jpg')
    # %%



print('Palette for rendering:')


def render_one(palette,new_palette,is_choose=False,net1=None,net2=None):
    render_cam_idx = 1

    c2w = trainer.test_dataset.poses[render_cam_idx]
    white_bg = trainer.test_dataset.white_bg
    ndc_ray = args.ndc_ray

    with torch.no_grad():
        rgb, depth, plt_decomps = render_one_view(trainer.test_dataset, model, c2w, trainer.renderer,
                                                  palette=palette,
                                                  new_palette=new_palette,
                                                  N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=trainer.device,is_choose=is_choose,net1=net1,net2=net2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 16))
    axes[0].set_axis_off()
    axes[0].imshow(rgb)
    axes[1].set_axis_off()
    axes[1].imshow(depth)

    fig, axes = plt.subplots(1, 1, figsize=(16, 8))
    axes.set_axis_off()
    axes.imshow(plt_decomps)

# Run the cells below to save this editing


def save_palette(new_palette):

    assert edit_name

    out_fn = f'rgb_palette{"_" + edit_name if edit_name else ""}'
    out_path = os.path.join(out_dir, f'{out_fn}.npy')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if os.path.exists(out_path):
        print('Error: file exists. Please specify another `edit_name`.')
    else:
        np.save(out_path, new_palette)
        print('Save palette to', out_path)


# %%
    '''Choose between 'test' / 'path' '''
def save(palette,new_palette,N_samples=-1,is_choose=False,net1=None,net2=None,probability=0.,**kwargs):
    cam_poses = 'train'

    save_dir = os.path.join(out_dir, f'render_{cam_poses}{"_" + edit_name if edit_name else ""}')

    if os.path.exists(save_dir):
        print('Error: directory exists. Please specify another `edit_name`.')
    else:
        c2ws = trainer.test_dataset.poses if cam_poses == 'test' else trainer.test_dataset.render_path
        if cam_poses=='train':
            c2ws = trainer.train_dataset.poses
        if cam_poses == 'test' :
            c2ws = c2ws[::1, ...]
        else:
            c2ws = c2ws[::1,...]
        white_bg = trainer.test_dataset.white_bg
        ndc_ray = trainer.args.ndc_ray

        print('Save renderings to', save_dir)
        print('=== render path ======>', c2ws.shape)
        with torch.no_grad():
            evaluation_path(trainer.test_dataset, model, c2ws, trainer.renderer, save_dir,
                            palette=palette, new_palette=new_palette,
                            N_samples=N_samples, white_bg=white_bg, ndc_ray=ndc_ray, save_video=True, device=trainer.device,is_choose=is_choose,net1=net1,net2=net2,probability=probability,**kwargs)
    # %%


# %%

"""
read color
"""
# palette_editing()
# palette_prior = np.load('palette_rgb_11.npy')
""" color edit"""
palette = model.renderModule.palette.get_palette_array().detach()
# palette[...,0,:] = torch.tensor([1.,0.,0.])
#
# palette = palette[...,:4,:]
# save_palette(palette.cpu().numpy())
# palette = model.renderModule.palette.get_palette_array().detach()
# palette = palette.clip(0. ,1.)
# # palette_prior = palette_prior.clip(0.,1.)
# print(palette)
# # print(palette_prior)
# # save(palette_prior,palette)
# edit = 1
# new_palette = []
# input_num = []
# out_put_num = []
# device = trainer.device
"""
many palette
"""
# for i in range(edit):
#     indata = f"/home/ubuntu/Rencq/nerf_data/point_cloud/fern/opaque_3/eps007points40/out_point_cloud{i}.txt"
#
#     input = np.loadtxt(indata)
#     input = torch.tensor(input,dtype=torch.float64)
#     maxlabel = max(input[...,3])
#     output = int(maxlabel+1)
#     print(output)
#     input_num.append(256)
#     out_put_num.append(output)
#     new_palette.append(palette[i].repeat(output+1).reshape((-1,palette.shape[1])).type(torch.float64).to(device))

"""
sole palette
"""
#
# indata = f"/home/ubuntu/data/drums/point_cloud/out_point_clouds_correct_{edit}.txt"
# input = np.loadtxt(indata)
# input = torch.tensor(input,dtype=torch.float64)
# maxlabel = max(input[...,3])
# output = int(maxlabel)+1
# for i in range(len(palette)):
#     if i == edit:
#         new_palette.append(palette[edit].repeat(output).reshape((-1,palette.shape[1])).type(torch.float64).to(device))
#     else:
#         new_palette.append(palette[i].repeat(1).reshape((-1,palette.shape[1])).type(torch.float64).to(device))
#
#
#
# print("======>new_palette\n")
# print(new_palette)
#
# net1 = point_empty()
# # net1.load_state_dict(torch.load('./logs/point_cloud/model1_param_0.pth',map_location=device))
# print("========>  load net1 state_dict finished ")
# net2 = {}
# net2[f'model{edit}'] = point_cloud_classical(3,output)
#
# for i in range(len(palette)):
#     if i == edit:
#         net2[f'model{i}'].load_state_dict(torch.load(f'./logs/point_cloud/drums_model2_param_{i}.pth',map_location=device))
# # net2[f'model{edit}'].load_state_dict(torch.load(f'./logs/point_cloud/model2_param_{edit}.pth',map_location=device))
# print("========>  load net2 state_dict finished ")
# print("nSamples =====> ",args.nSamples)
# new_palette[edit][2] = torch.tensor([0.,0.,0.8],dtype=torch.float64)
# print("=======> changed new palette")
# print(new_palette)
# print(palette)
# save(palette.cpu(),new_palette=new_palette,is_choose=True,net1=net1,net2=net2,N_samples=args.nSamples,edit=edit,ret_color_correction_map=True)
# print("++++++++++++palette+++++++++++++")
# print(palette)
# print("++++++++++++newpalette+++++++++++")
# print(new_palette)
# new_palette = np.load("./data_palette/drums/rgb_palette_correct.npy")

palette[...,1,:] = torch.tensor([0,1,0])
save(palette.cpu(),new_palette=None,N_samples=args.nSamples,ret_color_correction_map=True)
