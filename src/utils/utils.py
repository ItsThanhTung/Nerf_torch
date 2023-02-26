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

def sample_pts_from_rays(ray_batch, n_samples, perturb, lindisp):
    N_rays = ray_batch.shape[0]                         # num rays in batch, llff num rays = 1024
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each

    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=n_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, n_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]     # r(t) = o + td

    return pts
