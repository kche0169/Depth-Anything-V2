# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 新增的多模态交叉注意力机制
class CrossModalAttention(nn.Module):
    """多模态交叉注意力模块，用于融合不同模态（如RGB、深度、雷达）的特征"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # 为查询和键/值分别定义投影
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_modal: Tensor, kv_modal: Tensor) -> Tensor:
        """
        交叉注意力计算
        q_modal: 查询模态的特征 [B, N_q, C]
        kv_modal: 键值模态的特征 [B, N_kv, C]
        """
        B, N_q, C = q_modal.shape
        N_kv = kv_modal.shape[1]
        
        # 计算查询、键和值
        q = self.q_proj(q_modal).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) * self.scale
        k = self.k_proj(kv_modal).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(kv_modal).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 计算注意力权重
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 应用注意力权重到值上
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MemEffCrossModalAttention(CrossModalAttention):
    """使用xFormers优化的内存高效多模态交叉注意力"""
    def forward(self, q_modal: Tensor, kv_modal: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(q_modal, kv_modal)

        B, N_q, C = q_modal.shape
        N_kv = kv_modal.shape[1]
        
        # 计算查询、键和值
        q = self.q_proj(q_modal).reshape(B, N_q, self.num_heads, C // self.num_heads) * self.scale
        k = self.k_proj(kv_modal).reshape(B, N_kv, self.num_heads, C // self.num_heads)
        v = self.v_proj(kv_modal).reshape(B, N_kv, self.num_heads, C // self.num_heads)

        # 使用xFormers的内存高效注意力计算
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N_q, C])

        # 投影输出
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MultiModalFusion(nn.Module):
    """多模态特征融合模块，整合RGB、深度和雷达特征"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_memory_efficient: bool = True,
    ) -> None:
        super().__init__()
        
        # 根据是否使用内存高效版本选择注意力实现
        attention_cls = MemEffCrossModalAttention if use_memory_efficient and XFORMERS_AVAILABLE else CrossModalAttention
        
        # 不同模态间的交叉注意力
        # RGB与深度之间的交互
        self.rgb_to_depth = attention_cls(dim, num_heads, qkv_bias, True, attn_drop, proj_drop)
        self.depth_to_rgb = attention_cls(dim, num_heads, qkv_bias, True, attn_drop, proj_drop)
        
        # RGB与雷达之间的交互
        self.rgb_to_radar = attention_cls(dim, num_heads, qkv_bias, True, attn_drop, proj_drop)
        self.radar_to_rgb = attention_cls(dim, num_heads, qkv_bias, True, attn_drop, proj_drop)
        
        # 深度与雷达之间的交互
        self.depth_to_radar = attention_cls(dim, num_heads, qkv_bias, True, attn_drop, proj_drop)
        self.radar_to_depth = attention_cls(dim, num_heads, qkv_bias, True, attn_drop, proj_drop)
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, rgb_feat: Tensor, depth_feat: Tensor = None, radar_feat: Tensor = None) -> Tensor:
        """
        多模态特征融合
        rgb_feat: RGB特征 [B, N, C]
        depth_feat: 深度特征 [B, N, C]，可选
        radar_feat: 雷达特征 [B, N, C]，可选
        """
        # 如果只有RGB特征可用，则直接返回
        if depth_feat is None and radar_feat is None:
            return rgb_feat
            
        features = []
        features.append(rgb_feat)  # 始终包含RGB特征
        
        # 计算RGB与深度之间的交互
        if depth_feat is not None:
            # 深度信息增强RGB特征
            depth_enhanced_rgb = rgb_feat + self.depth_to_rgb(depth_feat, rgb_feat)
            features.append(depth_enhanced_rgb)
            
        # 计算RGB与雷达之间的交互
        if radar_feat is not None:
            # 雷达信息增强RGB特征
            radar_enhanced_rgb = rgb_feat + self.radar_to_rgb(radar_feat, rgb_feat)
            features.append(radar_enhanced_rgb)
            
        # 如果同时有深度和雷达特征，计算它们之间的交互并融合
        if depth_feat is not None and radar_feat is not None:
            # 深度与雷达之间的交互
            depth_radar_interaction = self.depth_to_radar(depth_feat, radar_feat) + self.radar_to_depth(radar_feat, depth_feat)
            features.append(depth_radar_interaction)
            
        # 融合所有特征
        # 如果特征数量少于3个，用零填充
        while len(features) < 3:
            features.append(torch.zeros_like(rgb_feat))
            
        fused_features = torch.cat(features[:3], dim=-1)  # 只取前三个特征以适应fusion_layer的输入维度
        fused_output = self.fusion_layer(fused_features)
        
        return fused_output