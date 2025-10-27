import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop

class RadarRGBD(Dataset):
    def __init__(self, data_path, filelist_path, mode, size=(518, 518)):
        self.data_path = data_path
        self.mode = mode
        self.size = size

        # 1. 从 filelist_path 读取用于训练或验证的时间戳
        with open(filelist_path, 'r') as f:
            self.timestamps = [line.strip() for line in f.readlines()]

        # 2. 定义数据转换流程 (参考 hypersim.py)
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))

    def __getitem__(self, item):
        timestamp = self.timestamps[item]

        # 3. 构建图像和深度图的路径 (参考 show_fig.py)
        # 使用 glob 模糊匹配文件名
        image_pattern = os.path.join(self.data_path, 'image', f'*{timestamp}*')
        depth_pattern = os.path.join(self.data_path, 'depth', f'*{timestamp}*')
        
        image_path = glob.glob(image_pattern)[0]
        depth_path = glob.glob(depth_pattern)[0]

        # 4. 读取数据
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # 深度图是单通道的，直接读取
        # 注意：plt.imread 和 cv2.imread 读取的方式可能不同，这里统一用cv2
        # 如果深度图是 .png 格式，它可能被读取为多通道，需要确认
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # show_fig.py 中提到深度范围是 0-8m，并且值被归一化到 [0, 1]
        # 这里我们把它乘以8，转换成真实的米单位
        if depth.max() <= 1.0:
            depth = depth * 8.0
        
        # 如果深度图是3通道的灰度图，转为单通道
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]

        # 5. 应用数据转换
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        # 创建一个有效性掩码 (valid_mask)，只在深度 > 0 的地方进行损失计算
        sample['valid_mask'] = (sample['depth'] > 0)
        
        return sample

    def __len__(self):
        return len(self.timestamps)
