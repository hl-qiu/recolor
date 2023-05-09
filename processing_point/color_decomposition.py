import torch
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
def color_decomposition(rgbs,palette_rgb,dataset=None,fg=None,plt_vote_map_idx=0,**kwargs):
    """
    which palette the pixel color belongs to
    """
    # print(rgbs.shape)
    # print(palette_rgb.shape)
    dist = torch.reshape(rgbs,(rgbs.shape[0],1,-1)) - torch.reshape(palette_rgb,(1,palette_rgb.shape[0],-1))
    dist = torch.linalg.norm(dist,dim=-1)

    pixel_color_index = torch.argmin(dist,dim=-1)  # bs * 1

    # Eucli_distance

    # if dataset is not None and dataset.white_bg and fg is not None:
    #     all_rgb_cp = dataset.all_rgbs.clone().cpu().numpy()
    # else:
    #     all_rgb_cp = rgbs.copy()
    #
    # all_rgb_maps = all_rgb_cp.reshape(-1,h,w,3)
    return pixel_color_index

def opaque_decompostition(opaque):

    true_idx = torch.argmax(opaque,dim=-1)
    return true_idx


def color_Ed(rgbs,palette_rgb,idx,epos):

    dist = torch.linalg.norm(torch.reshape(rgbs,(-1,3)) - torch.reshape(palette_rgb[idx],(-1,3)),dim=-1)


    return dist<epos



def plot_color_decomposition_idx(ds_test_dataset,rgbs,palette_rgb,plot_palette_color_idx=0,dataset=None,fg=None):
    w, h = ds_test_dataset.img_wh
    # palette_rgb = palette_rgb.clip(0.,1.)
    rgbs = torch.tensor(rgbs/255)
    rgbs = torch.reshape(rgbs,(-1,3))
    palette_rgb = torch.tensor(palette_rgb)
    palette_number = palette_rgb.shape[0]
    color_deco = color_decomposition(rgbs,palette_rgb) # bs * h * w
    # print(rgbs)
    print(palette_rgb)
    plot_palette_color =  0
    true_idx = (color_deco != plot_palette_color) # bs * 1

    if dataset is not None and dataset.white_bg and fg is not None:
        all_rgb_cp = dataset.all_rgbs.clone().cpu()
        all_rgb_cp_original = dataset.all_rgbs.clone().cpu()
        fg[fg==True] = true_idx

        all_rgb_cp[fg] = 1.
    else:
        all_rgb_cp = torch.clone(rgbs)
        all_rgb_cp_original = torch.clone(rgbs)
        all_rgb_cp[true_idx] = 1.



    all_rgb_maps = torch.reshape(all_rgb_cp,(-1, h, w, 3))
    # print(all_rgb_maps)
    all_rgb_cp_original = torch.reshape(all_rgb_cp_original,(-1,h,w,3))




    fig,axes = plt.subplots(1,2)
    axes[0].imshow(all_rgb_maps[plot_palette_color_idx].clone().numpy())
    axes[1].imshow(all_rgb_cp_original[plot_palette_color_idx].clone().numpy())
#%%

##return different color
##not white fg
def plt_color_decomposition(opaque,rgb,palette_rgb,dataset=None,eps=None,is_opaque=True):

    # palette_rgb = palette_rgb.clip(0.,1.)

    if dataset is not None:

        rgbs = dataset.all_rgbs.clone()

    else:
        rgbs = rgb

    rgbs = torch.tensor(rgbs)
    rgbs = torch.reshape(rgbs,(-1,3))
    palette_rgb = torch.tensor(palette_rgb)
    palette_number = palette_rgb.shape[0]

    true_idx = []
    if not is_opaque:
        color_deco = color_decomposition(rgbs,palette_rgb) # bs * h * w

        # change color distance
        dist = color_Ed(rgbs,palette_rgb,color_deco,eps)
        # print(rgbs)
        print(palette_rgb)

        for i in range(palette_number):
            dist_tmp = dist.clone()
            idx = (color_deco == i)
            dist_tmp[idx==False] = False
            idx = dist_tmp.clone()
            true_idx.append(idx.tolist())
    else:
        opaque_deco = opaque_decompostition(opaque)
        for i in range(palette_number):
            idx = (opaque_deco == i)
            idx = idx.clone()
            true_idx.append(idx.tolist())
    # if dataset is not None and dataset.white_bg and fg is not None:
    #     all_rgb_cp = dataset.all_rgbs.clone().cpu()
    #     all_rgb_cp_original = dataset.all_rgbs.clone().cpu()
    #     fg[fg==True] = true_idx
    #
    #     all_rgb_cp[fg] = 1.
    # else:
    #     all_rgb_cp = torch.clone(rgbs)
    #     all_rgb_cp_original = torch.clone(rgbs)
    #     all_rgb_cp[true_idx] = 1.



    true_idx = torch.tensor(true_idx)

    return true_idx

