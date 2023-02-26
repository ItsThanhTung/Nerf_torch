import numpy as np
from src.dataloader.llff_utils import load_llff_data, create_cim
from src.geometric_utils.get_ray import get_rays_np

from torch.utils.data import Dataset, DataLoader
import torch

class LlffDataset(Dataset):
    def __init__(self, data, device):
        self.data = torch.Tensor(data).to(device)
        self.device = device
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]   # (ro, rd), target



class LlffProcessor:
    """ LlffProcessor
    Usage: This class helps to load llff dataset and maps the pixels image to 3D
                                                 rays by projecting camera pose

        Use get_*_data function to get trainning or testing data
    """
    def __init__(self, device):
        self.llffhold = 8
        self.no_ndc = False

        self.basedir = r'C:\Users\Asus\Downloads\VINAI\ComputerGraphic\data\nerf_llff_data\trex'
        self.batch_size = 1024

        self.i_train = []
        self.i_test = [] 
        self.i_val = [] 

        self.K = None           # Camera intrinsic matrix
        self.H, self.W, self.focal = None, None, None

        self.train_data = None
        self.test_data = None

        self.device = device

        self.preprocess_dataset()

    def get_camera_intrinsic(self):
        return self.K, self.H, self.W, self.focal, self.near, self.far

    def get_train_data(self):
        return self.train_data


    def get_test_data(self):
        return self.test_data


    def train_test_split(self, dataset_size, hwf):
        H, W, self.focal = hwf
        self.H, self.W = int(H), int(W)

        if self.llffhold > 0:
            i_test = np.arange(dataset_size)[::self.llffhold]

        i_val = i_test

        i_train = np.array([i for i in np.arange(int(dataset_size)) if
                                        (i not in i_test and i not in i_val)])

        self.i_train = i_train
        self.i_test = i_test
        self.i_val = i_val


    def preprocess_dataset(self):
        images, poses, bds, render_poses, i_test = load_llff_data(self.basedir, factor=8, recenter=True, \
                                                                bd_factor=.75, spherify=False, path_zflat=False)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]

        # Modify self.i_train, self.i_test
        self.train_test_split(images.shape[0], hwf)
      
        if self.no_ndc:
            self.near = np.ndarray.min(bds) * .9
            self.far = np.ndarray.max(bds) * 1.
        else:
            self.near = 0.
            self.far = 1.

        self.K = create_cim(hwf)
        
        rays = np.stack([get_rays_np(self.H, self.W, self.K, p) \
                                for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]

        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        train_rays_rgb = np.stack([rays_rgb[i] for i in self.i_train], 0) # train images only
        test_rays_rgb = np.stack([rays_rgb[i] for i in self.i_test], 0) # train images only
        
        train_rays_rgb = np.reshape(train_rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        train_rays_rgb = train_rays_rgb.astype(np.float32)
        
        test_rays_rgb = np.reshape(test_rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        test_rays_rgb = test_rays_rgb.astype(np.float32)

        # np.random.shuffle(train_rays_rgb)  
        # np.random.shuffle(test_rays_rgb)  

        self.train_data = train_rays_rgb
        self.test_data = test_rays_rgb


def get_data_loader(batch_size, device):
    data_processor = LlffProcessor(device)
    train_data = data_processor.get_train_data()
    test_data = data_processor.get_test_data()

    LlffTrainData = LlffDataset(train_data, device)
    LlffTestData = LlffDataset(test_data, device)

    train_dataloader = DataLoader(LlffTrainData, batch_size=batch_size, shuffle=True, num_workers=0,\
                                             generator=torch.Generator(device='cuda'))
    test_dataloader = DataLoader(LlffTestData, batch_size=batch_size, shuffle=False, num_workers=0,\
                                                generator=torch.Generator(device='cuda'))

    return train_dataloader, test_dataloader, data_processor.get_camera_intrinsic()