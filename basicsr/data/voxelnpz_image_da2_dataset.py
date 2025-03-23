from torch.utils import data as data
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import os
from pathlib import Path
import random
import numpy as np
import torch
import cv2
from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    recursive_glob)
from basicsr.data.event_util import events_to_voxel_grid, voxel_norm
from basicsr.data.transforms import augment, triple_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, get_root_logger
from torch.utils.data.dataloader import default_collate


# def put_hot_pixels_in_voxel_(voxel, hot_pixel_range=20, hot_pixel_fraction=0.00002):
#     num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
#     x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
#     y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
#     for i in range(num_hot_pixels):
#         # voxel[..., :, y[i], x[i]] = random.uniform(-hot_pixel_range, hot_pixel_range)
#         voxel[..., :, y[i], x[i]] = random.randint(-hot_pixel_range, hot_pixel_range)

def put_hot_pixels_in_voxel(voxel, hot_pixel_range=30, hot_pixel_fraction=0.0001):
    # 计算热点像素数量
    # voxel hwc
    num_hot_pixels = int(hot_pixel_fraction * voxel.shape[0] * voxel.shape[1])
    
    # 随机生成热点像素的坐标
    x = np.random.randint(0, voxel.shape[1], size=num_hot_pixels)
    y = np.random.randint(0, voxel.shape[0], size=num_hot_pixels)
    
    # 随机生成热点像素的值
    hot_pixel_values = np.random.randint(-hot_pixel_range, hot_pixel_range + 1, size=(num_hot_pixels,voxel.shape[2]))
    
    # 将热点像素的值赋给数组
    #print(voxel.shape)
    #print(y,x)
    print('hot_pixel_values', hot_pixel_values.shape)
    print('x, y', x.shape, y.shape)

    voxel[y, x,:] = hot_pixel_values
    
    return np.array(voxel,dtype=np.float32)

# def add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.1):
#     noise = noise_std * torch.randn_like(voxel)  # mean = 0, std = noise_std
#     if noise_fraction < 1.0:
#         mask = torch.rand_like(voxel) >= noise_fraction
#         noise.masked_fill_(mask, 0)
#     return voxel + noise

def add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.4):
    # 生成噪声
    noise = noise_std * np.random.randn(*voxel.shape)  # mean = 0, std = noise_std
    
    # 如果 noise_fraction 小于 1.0，则生成掩码
    if noise_fraction < 1.0:
        mask = np.random.rand(*voxel.shape) >= noise_fraction
        noise[mask] = 0
    
    # 返回添加噪声后的体素
    return np.array((voxel + noise),dtype=np.float32)

def add_noise_to_img(img, noise_std=1.0, noise_fraction=0.1):
    # 生成噪声
    # noise = noise_std * np.random.randn(*voxel.shape)  # mean = 0, std = noise_std
    #noise_std = np.random.randn(10)+1
    noise_std = np.random.uniform(1,20) 
    noise = np.random.normal(0, noise_std, img.shape)/255.0
    
    # 如果 noise_fraction 小于 1.0，则生成掩码
    # if noise_fraction < 1.0:
    #     mask = np.random.rand(*voxel.shape) >= noise_fraction
    #     noise[mask] = 0
    
    # 返回添加噪声后的体素
    out = np.clip(img+noise,0,1,dtype=np.float32)
    return out



class VoxelnpzPngSingleDeblurDa2Dataset(data.Dataset):
    """Paired vxoel(npz) and blurry image (png) dataset for event-based single image deblurring.
    --HighREV
    |----train
    |    |----blur
    |    |    |----SEQNAME_%5d.png
    |    |    |----...
    |    |----voxel
    |    |    |----SEQNAME_%5d.npz
    |    |    |----...
    |    |----sharp
    |    |    |----SEQNAME_%5d.png
    |    |    |----...
    |----val
    ...

    
    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot (str): Data root path.
            io_backend (dict): IO backend type and other kwarg.
            num_end_interpolation (int): Number of sharp frames to reconstruct in each blurry image.
            num_inter_interpolation (int): Number of sharp frames to interpolate between two blurry images.
            phase (str): 'train' or 'test'

            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(VoxelnpzPngSingleDeblurDa2Dataset, self).__init__()
        self.opt = opt
        self.dataroot = Path(opt['dataroot'])
        self.dataroot_voxel = Path(opt['dataroot_voxel'])
        self.split = 'train' if opt['phase'] == 'train' else 'val'  # train or val
        self.norm_voxel = opt['norm_voxel']
        self.dataPath = []

        blur_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, 'blur'), suffix='.png'))
        blur_frames = [os.path.join(self.dataroot, 'blur', blur_frame) for blur_frame in blur_frames]
        
        sharp_frames = sorted(recursive_glob(rootdir=os.path.join(self.dataroot, 'sharp'), suffix='.png'))
        sharp_frames = [os.path.join(self.dataroot, 'sharp', sharp_frame) for sharp_frame in sharp_frames]

        event_frames = sorted(recursive_glob(rootdir=self.dataroot_voxel, suffix='.npz'))
        event_frames = [os.path.join(self.dataroot_voxel, event_frame) for event_frame in event_frames]
        
        assert len(blur_frames) == len(sharp_frames) == len(event_frames), f"Mismatch in blur ({len(blur_frames)}), sharp ({len(sharp_frames)}), and event ({len(event_frames)}) frame counts."

        for i in range(len(blur_frames)):
            self.dataPath.append({
                'blur_path': blur_frames[i],
                'sharp_path': sharp_frames[i],
                'event_paths': event_frames[i],
            })
        logger = get_root_logger()
        logger.info(f"Dataset initialized with {len(self.dataPath)} samples.")

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # import pdb; pdb.set_trace()

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        image_path = self.dataPath[index]['blur_path']
        gt_path = self.dataPath[index]['sharp_path']
        event_path = self.dataPath[index]['event_paths']

        # get LQ
        img_bytes = self.file_client.get(image_path)  # 'lq'
        img_lq = imfrombytes(img_bytes, float32=True)
        # get GT
        img_bytes = self.file_client.get(gt_path)    # 'gt'
        img_gt = imfrombytes(img_bytes, float32=True)

        voxel = np.load(event_path)['voxel']

        ## Data augmentation
        # voxel shape: h,w,c
        # random resize
        resize_prob = random.random() < 0.7
        # random_scale = (random.random()-0.5)*0.2/0.5 + 1.0
        random_scale = random.random() + 0.7

        random_scale_w = (random.random()-0.5)*2/5.0
        random_scale_h = (random.random()-0.5)*2/5.0

        if resize_prob and (gt_size is not None):
            if random.random() < 0.7:
                r_w = int(img_lq.shape[1]*(random_scale+random_scale_w))
                r_h = int(img_lq.shape[0]*(random_scale+random_scale_h))
            else:
                r_w = int(img_lq.shape[1]*(random_scale))
                r_h = int(img_lq.shape[0]*(random_scale))
            # print("img_lq:",img_lq.shape,r_w,r_h,random_scale,random_scale_w,random_scale_h)
            img_lq = cv2.resize(img_lq,(r_w,r_h))
            voxel = cv2.resize(voxel,(r_w,r_h))
            img_gt = cv2.resize(img_gt,(r_w,r_h))
        

        #print(img_lq.shape,voxel.shape,img_gt.shape)
        # crop
        if gt_size is not None:
            img_gt, img_lq, voxel = triple_random_crop(img_gt, img_lq, voxel, gt_size, scale, gt_path)

        noise_prob = random.random() < 0.5
        if noise_prob and (gt_size is not None):
            voxel = add_noise_to_voxel(voxel)
            voxel = put_hot_pixels_in_voxel(voxel)
            img_lq = add_noise_to_img(img_lq)

        # flip, rotate
        total_input = [img_lq, img_gt, voxel] 
        img_results = augment(total_input, self.opt['use_hflip'], self.opt['use_rot'])
        #print(img_results[0].shape,img_results[1].shape,img_results[2].shape)
        #print(img_results[0].dtype,img_results[1].dtype,img_results[2].dtype)
        img_results = img2tensor(img_results) # hwc -> chw
        img_lq, img_gt, voxel = img_results

        ## Norm voxel
        if self.norm_voxel:
            voxel = voxel_norm(voxel)

        origin_index = os.path.basename(image_path).split('.')[0]

        return {'frame': img_lq, 'frame_gt': img_gt, 'voxel': voxel, 'image_name': origin_index}

    def __len__(self):
        return len(self.dataPath)

if __name__ == "__main__":

    opt={}
    opt['dataroot']='/home/work/nvme1/baolong/EventDeblur/HighREV/train'
    opt['dataroot_voxel']='/home/work/nvme1/baolong/EventDeblur/HighREV/train/voxel30'
    opt['phase'] = 'train'
    opt['norm_voxel']= True
    opt['io_backend']={'type':'disk'}
    opt['scale']=1
    opt['gt_size']=None
    opt['use_hflip'] = True
    opt['use_rot']= True

    dataset = VoxelnpzPngSingleDeblurDa2Dataset(opt)

    print("len(dataset) ",len(dataset))
    # print("dataset.blur_length ",dataset.blur_length)
    for i in range(50):
        tmp = dataset[i]
        print("frame ",tmp['frame'].shape)
        print("frame_gt ",tmp['frame_gt'].shape)
        print("voxel ",tmp['voxel'].shape)
        print("image_name ",tmp['image_name'])
