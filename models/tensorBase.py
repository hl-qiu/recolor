import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from einops import rearrange

from .sh import eval_sh_bases


RenderBufferProp = namedtuple(
    'RenderBufferProp',
    ['name', 'len', 'detach_weight', 'type'],
    defaults=[None, 0, False, ''])

def split_render_buffer(rend_buf, layout):
    ret = {}

    start_idx = 0
    for prop in layout:
        k = prop.name
        ret[k] = rend_buf[..., start_idx:start_idx + prop.len]
        start_idx += prop.len

    return ret

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma * dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]

def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = rearrange(positions[..., None] * freq_bands, 'N D F -> N (D F)')  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def SHRender(xyz_sampled, viewdirs, features, **kwargs):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features, **kwargs):
    rgb = features
    return rgb

class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, **kwargs):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_PE(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3) + inChanel  #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, **kwargs):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + inChanel
        self.viewpe = viewpe

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features, **kwargs):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]).to(
            self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1


class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, density_n_comp=8, appearance_n_comp=24, app_dim=27, alphaMask=None,
                 near_far=(2.0, 6.0), density_shift=-10, alphaMask_thres=0.001, distance_scale=25,
                 rayMarch_weight_thres=0.0001, step_ratio=2.0, fea2denseAct='softplus', **kwargs):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim  #表面特征维度
        self.aabb = aabb  #盒子大小
        self.alphaMask = alphaMask
        self.device = device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize)
        #矩阵取的维度
        self.matMode = [[0, 1], [0, 2], [1, 2]]
        #向量取得维度
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]

        self.init_svd_volume(gridSize[0], device)

        self.renderModule_kwargs = kwargs
        #渲染模型使用了子类的
        self.renderModule = self.init_render_func(**kwargs)
        self.n_dim = getattr(self.renderModule, 'n_dim', 3)
        self.render_buf_layout = getattr(
            self.renderModule, 'render_buf_layout',
            [RenderBufferProp('rgb', 3, False, 'RGB')])
        print('[TensorBase init] renderModule:', self.renderModule)
        print('[TensorBase init] render buffer layout:', self.render_buf_layout)

    def split_render_bufs(self, render_bufs, k):
        render_layout = getattr(self.renderModule, 'render_layout', {'rgb': (0, 3)})

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, **kwargs):
        print('[init_render_func]', "shadingMode", shadingMode, "pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        if shadingMode == 'MLP_PE':
            ret = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(self.device)
        elif shadingMode == 'MLP_Fea':
            ret = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(self.device)
        elif shadingMode == 'MLP':
            ret = MLPRender(self.app_dim, view_pe, featureC).to(self.device)
        elif shadingMode == 'SH':
            ret = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            ret = RGBRender
        else:
            raise ValueError("Unrecognized shading module")
        return ret

    def update_stepSize(self, gridSize):
        print("[update_stepSize] aabb", self.aabb.view(-1))
        print("[update_stepSize] grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        #对角线 除以 步长
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("[update_stepSize] sampling step size: ", self.stepSize)
        print("[update_stepSize] sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def compute_densityfeature(self, xyz_sampled):
        pass

    def compute_appfeature(self, xyz_sampled):
        pass

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize': self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            **self.renderModule_kwargs
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device),
                                           alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far # 0 1
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)  # 1 / nsample e.g.(0,0.002,0.004,0.006 , ...)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None] #0d -> 1d
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        stepsize = self.stepSize  # 步长 默认体素一半 长度
        near, far = self.near_far  # 远近采样
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        # t length
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        #start t_min not 0
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])  # （bs, nsample） e.g. (0,1,2,3,4,...)

        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[..., None] + step)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        #超出盒子范围
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1, 3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    #掩码操作，清除sigma比较低的点
    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha >= self.alphaMask_thres] = 1
        alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"[updateAlphaMask] bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100))
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240 * 5,is_depth=False,final_mask=None, bbox_only=False):
        print('[filtering_rays]', end=' ')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)  # .clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)  # .clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _, _ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox = (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rays.shape[:-1])

        print(f'Ray filtering done! takes {time.time() - tt} s. '
              f'ray mask ratio: {torch.count_nonzero(mask_filtered) / N}')
        all_rays_mask = all_rays[mask_filtered]
        all_rgbs_mask = all_rgbs[mask_filtered]
        if is_depth:
            final_mask = final_mask[mask_filtered]
            return all_rays_mask,all_rgbs_mask,final_mask
        else:
            return all_rays_mask,all_rgbs_mask

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            #计算theta特征
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            #激活函数
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return alpha

    '''
    训练头
    '''
    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, **kwargs):

        # sample points
        viewdirs = rays_chunk[:, 3:6]  #(bs,3)
        if ndc_ray: #ndc  depth [0,1]
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                                 N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:   #不是ndc
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                             N_samples=N_samples)
            #两个点之间距离
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)  #(bs,nsample,3)

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid  # (bs,443)  true false composition
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device) #(bs,443)
        # render buffer for each point: [RGB, component opaque, ...]
        render_buf = torch.zeros((*xyz_sampled.shape[:2], self.n_dim), device=xyz_sampled.device)
        #xyz_sampled (bs,sampled,3)

        if ray_valid.any():
            #计算theta
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid]) #(bs * nsample,)

            #激活函数
            validsigma = self.feature2density(sigma_feature) #(bs-,)
            #盒子内的坐标
            sigma[ray_valid] = validsigma    #(bs,nsample)

        #一个ray上的采样点 占的权重
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)#alpha (bs,nsample,1) weight (bs,nsample,1)

        #choose sample point
        app_mask = weight > self.rayMarch_weight_thres  #(bs*nsample,)

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])  #(bs*nsample-,27)
            # link PLT_blend
            valid_render_bufs= self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features, is_train, **kwargs)  #(bs*nsample-,9) + 颜色修正
            render_buf[app_mask] = valid_render_bufs.type(torch.float32)


        ret = {}

        rend_dict = split_render_buffer(render_buf, self.render_buf_layout) # rgb (bs,nsample,3) opaque (bs,nsample,palette_num) sparsity_norm (bs,nsample,1)
        """滤波loss"""
        # self.loss = self.get_color_and_sigma_and_alpha(app_mask,xyz_sampled, viewdirs,sigma,rend_dict)

        acc_map = torch.sum(weight, -1)
        # rgb_map = torch.sum(weight[..., None] * rend_dict['rgb'], -2)

        for buf_prop in self.render_buf_layout: #rgb opaque sparsity_norm
            k = buf_prop.name
            if k == 'rgb' or kwargs.get(f'ret_{k}_map', False):
                if buf_prop.detach_weight:
                    w = weight[..., None].detach()
                else:
                    w = weight[..., None]

                rend_map = torch.sum(w * rend_dict[k], -2)

                if buf_prop.type == 'RGB':
                    if white_bg or (is_train and torch.rand((1,)) < 0.5):
                        rend_map[..., :3] += 1. - acc_map[..., None]

                    rend_map = rend_map.clamp(0, 1)

                ret[f'{k}_map'] = rend_map

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)  #weight  which is depth location
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]  #accmap which is sum(weight) = 1

        ret['depth_map'] = depth_map


        if kwargs.get('ret_weight', False):
            ret['weight'] = weight
        if kwargs.get('ret_acc_map', False):
            ret['acc_map'] = acc_map
        if kwargs.get('ret_raw_render_buf', False):
            ret['raw_render_buf'] = render_buf

        return ret

    def get_color_and_sigma_and_alpha(self, app_mask, xyz_sample, viewdir, sigma_original, rend_dict_original):
        xyz_sampled = torch.reshape(xyz_sample[app_mask], (-1, 1, 3))  # (bs*nsample-,1,3)

        deta_xyz = xyz_sampled + torch.normal(0, 0.000001, (1, 10, 3)).to(xyz_sample.device)  # （bs*nsample,10,3）
        viewdir = torch.reshape(viewdir[app_mask], (-1, 1, 3)).expand(deta_xyz.shape[0], deta_xyz.shape[1],
                                                                      3)  # (bs*nsample,10,3)

        sigma_feature = self.compute_densityfeature(torch.reshape(deta_xyz, (-1, 3)))  # bs*nsample-,

        # 激活函数  得到sigma
        validsigma = self.feature2density(sigma_feature)  # bs*nsample-,

        app_features = self.compute_appfeature(torch.reshape(deta_xyz, (-1, 3)))
        # 获得 color 和 opaque
        render_bufs = self.renderModule(torch.reshape(deta_xyz, (-1, 3)), torch.reshape(viewdir, (-1, 3)), app_features,
                                        is_train=False)
        rend_dict = split_render_buffer(render_bufs, self.render_buf_layout)

        # rgb (bs*nsample*10,3) opaque (bs*nsample*10,palette_num) sparsity_norm (bs,nsample,1)
        return [xyz_sampled, deta_xyz, rend_dict_original['rgb'][app_mask], rend_dict['rgb'], sigma_original[app_mask],
                validsigma, rend_dict_original['opaque'][app_mask], rend_dict['opaque']]