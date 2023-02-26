import torch
from src.utils.utils import raw2outputs, sample_pdf
torch.manual_seed(0)


class Rendering:
    def __init__(self, n_samples, n_importance, perturb, lindisp, raw_noise_std, white_bkgd):
        self.n_samples = n_samples
        self.n_importance = n_importance

        self.raw_noise_std = raw_noise_std
        self.white_bkgd = white_bkgd
        self.perturb = perturb
        self.lindisp = lindisp

    
    def set_rays(self, ray_batch):
        self.N_rays = ray_batch.shape[0]  
        self.rays_o, self.rays_d = ray_batch[:,0:3], ray_batch[:,3:6]
        self.viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None


    def sample_coarsed_pts(self, ray_batch):
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
        near, far = bounds[...,0], bounds[...,1] # [-1,1]

        t_vals = torch.linspace(0., 1., steps=self.n_samples)
        if not self.lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([self.N_rays, self.n_samples])

        if self.perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            z_vals = lower + (upper - lower) * t_rand

        pts = self.rays_o[...,None,:] + \
                self.rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]     # r(t) = o + td

        self.coarsed_z_vals = z_vals
        return pts


    def sample_fined_pts(self):
        z_vals_mid = .5 * (self.coarsed_z_vals[...,1:] + self.coarsed_z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, self.coarsed_weights[...,1:-1], self.n_importance, det=(self.perturb==0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([self.coarsed_z_vals, z_samples], -1), -1)
        pts = self.rays_o[...,None,:] + self.rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        self.fined_z_vals = z_vals
        return pts


    def rendering(self, raw, is_fined):
        if not is_fined:
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, self.coarsed_z_vals, \
                                    self.rays_d, self.raw_noise_std, self.white_bkgd)

            self.coarsed_rgb_map = rgb_map
            self.coarsed_disp_map = disp_map
            self.coarsed_acc_map = acc_map
            self.coarsed_weights = weights
            self.coarsed_depth_map = depth_map
        
        else:
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, self.fined_z_vals, \
                                    self.rays_d, self.raw_noise_std, self.white_bkgd)

            self.fined_rgb_map = rgb_map
            self.fined_disp_map = disp_map
            self.fined_acc_map = acc_map
            self.fined_weights = weights
            self.fined_depth_map = depth_map

        return rgb_map, disp_map, acc_map, weights, depth_map

    
    def render_rays(self, ray_batch, coarsed_net, fined_net):
        self.set_rays(ray_batch)

        pts = self.sample_coarsed_pts(input)
        raw = coarsed_net.forward(pts, self.viewdirs)
        rgb_map, disp_map, acc_map, weights, depth_map = self.rendering(raw, is_fined=False)

        fined_pts = self.sample_fined_pts()
        fined_raw = fined_net.forward(fined_pts, self.viewdirs)
        fined_rgb_map, fined_disp_map, fined_acc_map, fined_weights, fined_depth_map = self.rendering(fined_raw, is_fined=True)

        return {"rgb_map"                 : rgb_map,
                "disp_map"                : disp_map,
                "acc_map"                 : acc_map,
                "weights"                 : weights,
                "depth_map"               : depth_map,
                "fined_rgb_map"           : fined_rgb_map,
                "depthfined_disp_map_map" : fined_disp_map,
                "fined_acc_map"           : fined_acc_map,
                "fined_weights"           : fined_weights,
                "fined_depth_map"         : fined_depth_map,}


