import torch
from src.geometric_utils.get_ray import ndc_rays

def preprocess_rays(rays_o, rays_d, H, W, K, near, far):
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1,3]).float()
    
    sh = rays_d.shape # [..., 3]

    # if use_ndc
    rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    rays = torch.cat([rays, viewdirs], -1)
    return rays

