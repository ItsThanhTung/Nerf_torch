from src.dataloader.llff import get_data_loader
import torch
import numpy as np
from src.utils.utils import preprocess_rays, sample_pts_from_rays


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train_loader, test_loader, camera_intrinsic = get_data_loader(1024, device)

    K, H, W, focal, near, far = camera_intrinsic

    for (rays_o, rays_d, target) in train_loader:
        rays = preprocess_rays(rays_o, rays_d, H, W, K, near, far)
        pts = sample_pts_from_rays(rays, n_samples=64, perturb=1, lindisp=False)
        print(rays.shape, pts)

        break
   
