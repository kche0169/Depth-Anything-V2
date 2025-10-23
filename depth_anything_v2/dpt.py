import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet
from .dinov2_layers.attention import MultiModalFusion  # 导入我们刚才创建的多模态融合模块


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out


class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)
        
        return depth.squeeze(1)
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)


# 新添加的多模态深度估计模型类
class MultiModalDepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False
    ):
        super(MultiModalDepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        # RGB编码器
        self.encoder = encoder
        self.rgb_encoder = DINOv2(model_name=encoder)
        embed_dim = self.rgb_encoder.embed_dim
        
        # 深度图编码器 - 使用与RGB相同的架构但调整输入通道
        self.depth_encoder = DINOv2(model_name=encoder, in_chans=1)
        
        # 雷达数据编码器 - 同样使用相同架构但可能需要调整输入
        # 根据雷达数据的具体表示方式调整in_chans
        self.radar_encoder = DINOv2(model_name=encoder, in_chans=1)  # 假设雷达数据为单通道
        
        # 多模态特征融合模块
        self.modal_fusion = nn.ModuleList([
            MultiModalFusion(dim=embed_dim, num_heads=8)
            for _ in range(len(self.intermediate_layer_idx[encoder]))
        ])
        
        # 深度预测头
        self.depth_head = DPTHead(embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, rgb, depth=None, radar=None):
        """
        多模态前向传播
        rgb: RGB图像 [B, 3, H, W]
        depth: 深度图 [B, 1, H, W]，可选
        radar: 雷达数据 [B, C_radar, H, W]，可选
        """
        patch_h, patch_w = rgb.shape[-2] // 14, rgb.shape[-1] // 14
        
        # 获取RGB特征
        rgb_features = self.rgb_encoder.get_intermediate_layers(
            rgb, 
            self.intermediate_layer_idx[self.encoder], 
            return_class_token=True
        )
        
        # 如果只有RGB输入，则直接使用RGB特征
        if depth is None and radar is None:
            features = rgb_features
        else:
            # 获取深度特征（如果提供）
            depth_features = None
            if depth is not None:
                depth_features = self.depth_encoder.get_intermediate_layers(
                    depth, 
                    self.intermediate_layer_idx[self.encoder], 
                    return_class_token=True
                )
                
            # 获取雷达特征（如果提供）
            radar_features = None
            if radar is not None:
                radar_features = self.radar_encoder.get_intermediate_layers(
                    radar, 
                    self.intermediate_layer_idx[self.encoder], 
                    return_class_token=True
                )
            
            # 对每一层特征进行多模态融合
            features = []
            for i, rgb_feat in enumerate(rgb_features):
                depth_feat = depth_features[i] if depth_features else None
                radar_feat = radar_features[i] if radar_features else None
                
                # 进行特征融合，注意这里rgb_feat、depth_feat和radar_feat都是包含特征和类别token的元组
                # 我们只对特征部分进行融合
                rgb_tensor, cls_token = rgb_feat
                
                # 融合不同模态的特征
                if depth_feat is not None:
                    depth_tensor = depth_feat[0]  # 取出特征部分
                else:
                    depth_tensor = None
                    
                if radar_feat is not None:
                    radar_tensor = radar_feat[0]  # 取出特征部分
                else:
                    radar_tensor = None
                
                # 使用多模态融合模块融合特征
                fused_tensor = self.modal_fusion[i](rgb_tensor, depth_tensor, radar_tensor)
                
                # 重新组合成元组形式
                features.append((fused_tensor, cls_token))
        
        # 使用深度预测头生成深度图
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)
        
        return depth.squeeze(1)
    
    @torch.no_grad()
    def infer_multi_modal(self, rgb_image, depth_image=None, radar_data=None, input_size=518):
        """
        多模态推理函数
        rgb_image: RGB图像
        depth_image: 深度图，可选
        radar_data: 雷达数据，可选
        input_size: 输入大小
        """
        # 处理RGB图像
        rgb_tensor, (h, w) = self.image2tensor(rgb_image, input_size)
        
        # 处理深度图（如果有）
        depth_tensor = None
        if depth_image is not None:
            depth_tensor, _ = self.depth2tensor(depth_image, input_size)
        
        # 处理雷达数据（如果有）
        radar_tensor = None
        if radar_data is not None:
            radar_tensor, _ = self.radar2tensor(radar_data, input_size)
        
        # 前向传播
        depth = self.forward(rgb_tensor, depth_tensor, radar_tensor)
        
        # 调整输出尺寸
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)
    
    def depth2tensor(self, depth_image, input_size=518):
        """将深度图转换为tensor"""
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            PrepareForNet(),  # 不需要RGB归一化
        ])
        
        h, w = depth_image.shape[:2]
        
        # 确保深度图是归一化的
        if depth_image.max() > 1:
            depth_image = depth_image / depth_image.max()
            
        depth = transform({'image': depth_image})['image']
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        depth = depth.to(DEVICE)
        
        return depth, (h, w)
    
    def radar2tensor(self, radar_data, input_size=518):
        """将雷达数据转换为tensor"""
        # 这个函数需要根据雷达数据的具体格式进行调整
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            PrepareForNet(),
        ])
        
        h, w = radar_data.shape[:2]
        
        # 归一化雷达数据
        if radar_data.max() > 1:
            radar_data = radar_data / radar_data.max()
            
        radar = transform({'image': radar_data})['image']
        radar = torch.from_numpy(radar).unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        radar = radar.to(DEVICE)
        
        return radar, (h, w)