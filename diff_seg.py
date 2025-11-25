import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Union
import warnings

import torchvision


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步编码，用于扩散模型"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """残差块，支持时间嵌入"""

    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.1, groups=8):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)[:, :, None, None]
            h = h + time_emb

        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """自注意力块"""

    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W)

        # Attention with proper scaling
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)
        # out = out.view(B, C, H, W)
        out = self.proj(out)

        return x + out


class SharedEncoder(nn.Module):
    """共享特征编码器"""

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # 多尺度特征提取
        self.down1 = nn.Sequential(
            ResBlock(base_channels, base_channels),
            ResBlock(base_channels, base_channels)
        )
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)

        self.down2 = nn.Sequential(
            ResBlock(base_channels * 2, base_channels * 2),
            ResBlock(base_channels * 2, base_channels * 2)
        )
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)

        self.down3 = nn.Sequential(
            ResBlock(base_channels * 4, base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 4)
        )
        self.downsample3 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ResBlock(base_channels * 8, base_channels * 8),
            AttentionBlock(base_channels * 8),
            ResBlock(base_channels * 8, base_channels * 8)
        )

    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {x.shape}")

        x = self.conv_in(x)

        # 保存多尺度特征
        skip1 = self.down1(x)
        x = self.downsample1(skip1)

        skip2 = self.down2(x)
        x = self.downsample2(skip2)

        skip3 = self.down3(x)
        x = self.downsample3(skip3)

        x = self.bottleneck(x)

        return x, [skip1, skip2, skip3]


class SegmentationBranch(nn.Module):

    def __init__(self, base_channels=64):
        super().__init__()

        # 上采样解码器
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.up_conv3 = nn.Sequential(
            ResBlock(base_channels * 8, base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 4)
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up_conv2 = nn.Sequential(
            ResBlock(base_channels * 4, base_channels * 2),
            ResBlock(base_channels * 2, base_channels * 2)
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up_conv1 = nn.Sequential(
            ResBlock(base_channels * 2, base_channels),
            ResBlock(base_channels, base_channels)
        )

        # 输出未遮挡部分的掩码
        self.mask_head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, features, skips):
        skip1, skip2, skip3 = skips

        x = self.up3(features)
        x = torch.cat([x, skip3], dim=1)
        x = self.up_conv3(x)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up_conv2(x)

        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up_conv1(x)

        # 输出未遮挡部分掩码
        visible_mask = torch.sigmoid(self.mask_head(x))

        return visible_mask
        # visible_mask = (visible_mask > 0.5).float()
        # return self.mask_head(x)
class DDIMSampler:
    """简化版DDIM采样器"""
    def __init__(self, scheduler, num_steps=200, eta=0.0):
        self.scheduler = scheduler
        self.num_steps = num_steps
        self.eta = eta

    def sample(self, model, occluded_rgb, visible_mask, device):
        B, C, H, W = occluded_rgb.shape
        x = torch.randn(B, 3, H, W, device=device)
        timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0, self.num_steps, dtype=torch.long, device=device)

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(B)
            with torch.no_grad():
                noise_pred = model(
                    occluded_rgb,
                    noisy_rgb=x,
                    timestep=t_batch,
                    mode='diffusion'
                )['denoised_rgb']

            alpha_cumprod_t = self.scheduler.alpha_cumprod[t_batch].view(-1, 1, 1, 1)
            alpha_cumprod_t_prev = self.scheduler.alpha_cumprod[timesteps[min(i+1, len(timesteps)-1)].long()].view(-1, 1, 1, 1)
            sqrt_alpha = torch.sqrt(alpha_cumprod_t)
            sqrt_alpha_prev = torch.sqrt(alpha_cumprod_t_prev)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod_t)

            pred_x0 = (x - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * noise_pred
            x = sqrt_alpha_prev * pred_x0 + dir_xt

            # 可选：不强行替换可见区域
            # x = x * (1 - visible_mask) + visible_apple * visible_mask

        return x


class DDPMScheduler:
    """标准DDPM噪声调度器"""

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # 线性beta调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]])

        # 计算去噪所需的系数
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha_cumprod = torch.sqrt(1.0 / self.alpha_cumprod)
        self.sqrt_recipm1_alpha_cumprod = torch.sqrt(1.0 / self.alpha_cumprod - 1)

        # DDPM采样相关系数
        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

    def to(self, device):
        """移动到指定设备"""
        for attr_name in ['betas', 'alphas', 'alpha_cumprod', 'alpha_cumprod_prev',
                          'sqrt_alpha_cumprod', 'sqrt_one_minus_alpha_cumprod',
                          'sqrt_recip_alpha_cumprod', 'sqrt_recipm1_alpha_cumprod',
                          'posterior_variance', 'posterior_log_variance_clipped']:
            setattr(self, attr_name, getattr(self, attr_name).to(device))
        return self

    def add_noise(self, x_start, noise, timesteps):
        """向清洁图像添加噪声"""
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[timesteps].view(-1, 1, 1, 1)

        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def sample_prev_timestep(self, x_t, noise_pred, t):
        """DDPM采样步骤"""
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_t_prev = self.alpha_cumprod_prev[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)

        # 预测x_0
        pred_original_sample = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 计算x_{t-1}的均值
        pred_sample_direction = torch.sqrt(1 - alpha_cumprod_t_prev) * noise_pred
        pred_prev_sample = torch.sqrt(alpha_cumprod_t_prev) * pred_original_sample + pred_sample_direction

        # 添加噪声（除了最后一步）
        if t.min() > 0:
            variance = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * noise

        return pred_prev_sample


class MultiScaleConditionalFusion(nn.Module):
    """多尺度条件融合模块 - 创新点1"""

    def __init__(self, base_channels=64):
        super().__init__()

        # 多尺度特征融合
        self.scale_fusion_8x = nn.Sequential(
            nn.Conv2d(base_channels * 8 + 1, base_channels * 8, 3, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU()
        )

        self.scale_fusion_4x = nn.Sequential(
            nn.Conv2d(base_channels * 4 + 1, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU()
        )

        self.scale_fusion_2x = nn.Sequential(
            nn.Conv2d(base_channels * 2 + 1, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )

        self.scale_fusion_1x = nn.Sequential(
            nn.Conv2d(base_channels + 1, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )

        # 跨尺度注意力
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=base_channels * 8, num_heads=8, batch_first=True
        )

    def forward(self, features_8x, features_4x, features_2x, features_1x, visible_mask):
        # 将可见掩码调整到不同尺度
        mask_8x = F.interpolate(visible_mask, size=features_8x.shape[-2:], mode='bilinear')
        mask_4x = F.interpolate(visible_mask, size=features_4x.shape[-2:], mode='bilinear')
        mask_2x = F.interpolate(visible_mask, size=features_2x.shape[-2:], mode='bilinear')
        mask_1x = F.interpolate(visible_mask, size=features_1x.shape[-2:], mode='bilinear')

        # 多尺度条件融合
        fused_8x = self.scale_fusion_8x(torch.cat([features_8x, mask_8x], dim=1))
        fused_4x = self.scale_fusion_4x(torch.cat([features_4x, mask_4x], dim=1))
        fused_2x = self.scale_fusion_2x(torch.cat([features_2x, mask_2x], dim=1))
        fused_1x = self.scale_fusion_1x(torch.cat([features_1x, mask_1x], dim=1))

        return fused_8x, fused_4x, fused_2x, fused_1x


class AdaptiveTimeEmbedding(nn.Module):

    def __init__(self, time_dim=256, condition_dim=1):
        super().__init__()
        self.base_time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # 条件调制网络
        self.condition_modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(condition_dim, time_dim // 4),
            nn.ReLU(),
            nn.Linear(time_dim // 4, time_dim),
            nn.Sigmoid()
        )

        # 动态权重生成
        self.weight_generator = nn.Sequential(
            nn.Linear(time_dim + condition_dim, time_dim),
            nn.Tanh()
        )

    def forward(self, timestep, condition_mask):
        # 基础时间嵌入
        base_emb = self.base_time_mlp(timestep)

        # 条件调制
        condition_mod = self.condition_modulator(condition_mask)

        # 自适应融合
        condition_global = torch.mean(condition_mask.view(condition_mask.size(0), -1), dim=1, keepdim=True)
        adaptive_weight = self.weight_generator(torch.cat([base_emb, condition_global], dim=1))

        # 最终时间嵌入
        final_emb = base_emb * (1 + condition_mod) + adaptive_weight

        return final_emb


class ProgressiveRefinementModule(nn.Module):
    """渐进式细化模块 - 创新点3"""

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        # 粗糙重建分支
        self.coarse_branch = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            ResBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )

        # 细节增强分支
        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_channels, 3, padding=1),
            ResBlock(base_channels, base_channels),
            ResBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )

        # 注意力门控 - 修复输出通道数
        self.attention_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),  # 输出通道数应该与detail_enhancement一致
            nn.Sigmoid()
        )

    def forward(self, noisy_input, condition_mask):
        # 粗糙重建
        coarse_result = self.coarse_branch(noisy_input)

        # 细节增强
        detail_input = torch.cat([coarse_result, noisy_input], dim=1)
        detail_enhancement = self.detail_branch(detail_input)

        # 注意力门控融合
        gate_input = torch.cat([coarse_result, detail_enhancement], dim=1)
        attention_weight = self.attention_gate(gate_input)

        # 渐进式融合
        refined_result = coarse_result + attention_weight * detail_enhancement

        return refined_result, coarse_result, detail_enhancement


class RGBDiffusionBranch(nn.Module):
    """RGB扩散分支 - 生成完整的前景苹果"""

    def __init__(self, base_channels=64, time_dim=256):
        super().__init__()
        self.time_dim = time_dim

        # 时间嵌入网络 - 修复维度问题
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # 噪声RGB输入处理
        self.noise_conv = nn.Conv2d(3, base_channels, 3, padding=1)

        # 条件融合 - 整合原始图像特征、分割掩码和噪声
        # 确保维度匹配
        self.condition_fusion = nn.Sequential(
            nn.Conv2d(base_channels * 8 + base_channels + 1, base_channels * 8, 1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU()
        )

        # 扩散UNet解码器
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.up_conv3 = nn.Sequential(
            ResBlock(base_channels * 8, base_channels * 4, time_dim),
            AttentionBlock(base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 4, time_dim)
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up_conv2 = nn.Sequential(
            ResBlock(base_channels * 4, base_channels * 2, time_dim),
            ResBlock(base_channels * 2, base_channels * 2, time_dim)
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up_conv1 = nn.Sequential(
            ResBlock(base_channels * 2, base_channels, time_dim),
            ResBlock(base_channels, base_channels, time_dim)
        )

        # 输出RGB图像 (3通道)
        self.out_conv = nn.Conv2d(base_channels, 3, 1)

    def forward(self, features, skips, visible_mask, noisy_rgb, timestep):
        skip1, skip2, skip3 = skips

        # 时间嵌入 - 直接使用传入的 timestep（已经过 adaptive_time_embedding 处理）
        if isinstance(timestep, torch.Tensor) and timestep.dim() == 1:
            # 如果是一维的时间步，使用自己的时间嵌入
            time_emb = self.time_mlp(timestep)
        else:
            # 如果已经是嵌入的特征，直接使用
            time_emb = timestep

        # 处理噪声RGB并调整到特征尺寸
        B, C, H, W = noisy_rgb.shape
        feat_h, feat_w = features.shape[-2:]

        # 使用双线性插值调整尺寸，保持更多信息
        noise_feat = F.interpolate(noisy_rgb, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
        noise_feat = self.noise_conv(noise_feat)

        # 调整可见掩码尺寸
        visible_mask_feat = F.interpolate(visible_mask, size=(feat_h, feat_w), mode='bilinear', align_corners=False)

        # 条件融合：原始特征 + 噪声特征 + 可见掩码
        x = torch.cat([features, noise_feat, visible_mask_feat], dim=1)
        x = self.condition_fusion(x)

        # 扩散去噪过程
        x = self.up3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.up_conv3[0](x, time_emb)
        x = self.up_conv3[1](x)
        x = self.up_conv3[2](x, time_emb)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up_conv2[0](x, time_emb)
        x = self.up_conv2[1](x, time_emb)

        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up_conv1[0](x, time_emb)
        x = self.up_conv1[1](x, time_emb)

        # 输出去噪后的RGB
        return self.out_conv(x)


class AppleCompletionNetwork(nn.Module):
    """苹果补全网络：从遮挡RGB到完整前景苹果"""

    def __init__(self, in_channels=3, base_channels=64, time_dim=256, num_timesteps=1000):
        super().__init__()

        self.shared_encoder = SharedEncoder(in_channels, base_channels)
        self.segmentation_branch = SegmentationBranch(base_channels)
        self.diffusion_branch = RGBDiffusionBranch(base_channels, time_dim)

        # 扩散调度器
        self.scheduler = DDPMScheduler(num_timesteps)

        # 多尺度条件融合模块
        self.multi_scale_fusion = MultiScaleConditionalFusion(base_channels)

        # 自适应时间嵌入模块
        self.adaptive_time_embedding = AdaptiveTimeEmbedding(time_dim)

        # 渐进式细化模块
        # self.refinement_module = ProgressiveRefinementModule(in_channels, base_channels)

    def forward(self, occluded_rgb, noisy_rgb=None, timestep=None, mode='both'):
        """
        Args:
            occluded_rgb: 被遮挡的RGB图像 [B, 3, H, W]
            noisy_rgb: 噪声RGB图像 [B, 3, H, W] (扩散模式)
            timestep: 时间步 [B] (扩散模式)
            mode: 'seg', 'diffusion', 'both'
        """
        # 输入验证
        if occluded_rgb.dim() != 4 or occluded_rgb.shape[1] != 3:
            raise ValueError(f"Expected occluded_rgb shape [B, 3, H, W], got {occluded_rgb.shape}")

        # 共享特征提取
        features, skips = self.shared_encoder(occluded_rgb)
        outputs = {}

        if mode in ['seg', 'both']:
            # 分割未遮挡部分
            visible_mask = self.segmentation_branch(features, skips)
            outputs['visible_mask'] = visible_mask

        if mode in ['diffusion', 'both']:
            if noisy_rgb is None or timestep is None:
                raise ValueError("noisy_rgb and timestep required for diffusion mode")

            # 获取或使用现有的可见掩码
            if 'visible_mask' not in outputs:
                visible_mask = self.segmentation_branch(features, skips)
            else:
                visible_mask = outputs['visible_mask']

            # 使用多尺度条件融合模块
            skip1, skip2, skip3 = skips
            fused_features, fused_skip2, fused_skip1, fused_skip0 = self.multi_scale_fusion(
                features, skip3, skip2, skip1, visible_mask
            )

            # 自适应时间嵌入
            time_emb = self.adaptive_time_embedding(timestep, visible_mask)

            # RGB扩散去噪 - 使用融合后的特征
            denoised_rgb = self.diffusion_branch(
                fused_features, [fused_skip0, fused_skip1, fused_skip2], visible_mask, noisy_rgb, time_emb
            )
            outputs['denoised_rgb'] = denoised_rgb

            # # 渐进式细化
            # refined_rgb, coarse_rgb, detail_rgb = self.refinement_module(denoised_rgb, visible_mask)
            # outputs['refined_rgb'] = refined_rgb
            # outputs['coarse_rgb'] = coarse_rgb
            # outputs['detail_rgb'] = detail_rgb
            #
            # # 最终图像直接使用扩散生成的图像
            outputs['final_result'] = denoised_rgb

        return outputs

    def extract_visible_apple(self, occluded_rgb):
        """提取可见的苹果部分"""
        with torch.no_grad():
            outputs = self.forward(occluded_rgb, mode='seg')
            visible_mask = outputs['visible_mask']

            # 提取可见部分
            visible_apple = occluded_rgb * visible_mask
            return visible_apple, visible_mask

    def complete_apple_with_fusion(self, occluded_rgb, num_steps=200, fusion_strength=0.8, use_eta=0.0):
        """完整苹果补全流程（改进的平滑融合版本）"""
        self.eval()
        device = occluded_rgb.device
        B, C, H, W = occluded_rgb.shape

        # 确保调度器在正确设备上
        self.scheduler = self.scheduler.to(device)

        with torch.no_grad():
            # 1. 提取可见部分
            visible_apple, visible_mask = self.extract_visible_apple(occluded_rgb)

            # 2. 使用标准DDPM采样
            x = torch.randn(B, 3, H, W, device=device)
            timesteps = torch.linspace(self.scheduler.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)

            for i, t in enumerate(timesteps):
                t_batch = t.repeat(B)

                # 预测噪声（注意：我们的模型输出的是去噪后的图像，需要转换为噪声预测）
                model_output = self.forward(
                    occluded_rgb,
                    noisy_rgb=x,
                    timestep=t_batch,
                    mode='diffusion'
                )
                predicted_denoised = model_output['denoised_rgb']

                # 从预测的去噪图像计算噪声
                alpha_cumprod_t = self.scheduler.alpha_cumprod[t_batch].view(-1, 1, 1, 1)
                sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
                sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)

                # 计算预测的噪声: noise = (x_t - sqrt_alpha * x_0) / sqrt_one_minus_alpha
                predicted_noise = (x - sqrt_alpha_cumprod * predicted_denoised) / sqrt_one_minus_alpha_cumprod

                # 标准DDPM采样步骤
                if i < len(timesteps) - 1:
                    x = self.scheduler.sample_prev_timestep(x, predicted_noise, t_batch)
                else:
                    # 最后一步，直接使用预测的去噪图像
                    x = predicted_denoised

            # 3. 简单的范围限制，不做过度后处理
            x = torch.clamp(x, -1, 1)

            # 4. 生成前景掩码
            features, skips = self.shared_encoder(x)
            apple_mask = self.segmentation_branch(features, skips)

            # 5. 改进的平滑融合策略
            final_result = self.smooth_blend_with_visible_parts(x, visible_apple, visible_mask)

            return {
                'final_result': final_result,
                'completed_apple': x,
                'harmonized_apple': predicted_denoised,
                'visible_mask': visible_mask,
                'apple_mask': apple_mask,
                'visible_apple': visible_apple
            }

    def smooth_blend_with_visible_parts(self, generated, visible_apple, visible_mask):
        """改进的平滑融合方法"""
        # 方法1: 高斯模糊掩码边界
        kernel_size = 15  # 较大的核用于更平滑的过渡
        sigma = 5.0

        # 创建高斯核
        kernel = self.get_gaussian_kernel(kernel_size, sigma).to(generated.device)

        # 对掩码进行高斯模糊，创建软边界
        soft_mask = F.conv2d(visible_mask, kernel, padding=kernel_size//2)
        soft_mask = torch.clamp(soft_mask, 0, 1)

        # 扩展到3通道
        soft_mask = soft_mask.expand_as(generated)

        # 使用软掩码进行平滑融合
        blended = generated * (1 - soft_mask) + visible_apple * soft_mask

        # 方法2: 在边界区域进行色彩匹配
        boundary_mask = self.get_boundary_mask(visible_mask, kernel_size=10)
        boundary_mask = boundary_mask.expand_as(generated)

        # 在边界区域对生成图像进行色彩校正
        if boundary_mask.sum() > 0:
            # 计算可见区域的平均颜色
            visible_mean = (visible_apple * soft_mask).sum(dim=[2,3], keepdim=True) / (soft_mask.sum(dim=[2,3], keepdim=True) + 1e-8)
            generated_mean = (generated * boundary_mask).sum(dim=[2,3], keepdim=True) / (boundary_mask.sum(dim=[2,3], keepdim=True) + 1e-8)

            # 色彩校正
            color_correction = visible_mean - generated_mean
            generated_corrected = generated + color_correction * boundary_mask * 0.5  # 部分校正，避免过度调整

            # 重新融合
            blended = generated_corrected * (1 - soft_mask) + visible_apple * soft_mask

        return torch.clamp(blended, -1, 1)

    def get_boundary_mask(self, mask, kernel_size=5):
        """获取掩码边界区域（改进版）"""
        # 使用形态学操作获取更精确的边界
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device) / (kernel_size * kernel_size)

        # 膨胀和腐蚀操作 - 确保输出尺寸与输入一致
        dilated = F.conv2d(mask, kernel, padding=kernel_size//2)
        eroded = F.conv2d(mask, -kernel, padding=kernel_size//2) + 1

        # 确保尺寸匹配 - 如果尺寸不匹配，进行裁剪或插值调整
        target_size = mask.shape[-2:]
        if dilated.shape[-2:] != target_size:
            dilated = F.interpolate(dilated, size=target_size, mode='bilinear', align_corners=False)
        if eroded.shape[-2:] != target_size:
            eroded = F.interpolate(eroded, size=target_size, mode='bilinear', align_corners=False)

        # 边界 = 膨胀 - 腐蚀
        boundary = (dilated > 0.7) & (eroded < 0.3)

        # 确保输出张量与输入掩码尺寸完全一致
        if boundary.shape != mask.shape:
            boundary = F.interpolate(boundary.float(), size=target_size, mode='nearest').bool()

        return boundary.float()

    def get_gaussian_kernel(self, kernel_size, sigma):
        """创建高斯卷积核"""
        # 创建1D高斯核
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()

        # 创建2D高斯核
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)

        # 重塑为卷积核格式 [out_channels, in_channels, height, width]
        kernel = gaussian_2d.unsqueeze(0).unsqueeze(0)

        return kernel
def total_variation_loss(img):
    return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
           torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))


from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor
class AppleCompletionLoss(nn.Module):
    """改进的苹果补全损失函数"""

    def __init__(self, seg_weight=1.0, rgb_weight=1.0, perceptual_weight=0.5,
                 consistency_weight=0.2, fusion_weight=0.1, vgg_weight=0.3,
                 detail_weight=0.2, edge_weight=0.1):
        super().__init__()
        self.seg_weight = seg_weight
        self.rgb_weight = rgb_weight
        self.perceptual_weight = perceptual_weight
        self.consistency_weight = consistency_weight
        self.fusion_weight = fusion_weight
        self.vgg_weight = vgg_weight
        self.detail_weight = detail_weight
        self.edge_weight = edge_weight

        self.seg_loss = nn.BCELoss()
        self.rgb_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # 使用更深层的VGG特征
        vgg = vgg16(pretrained=True).features.eval()
        self.vgg_extractor = create_feature_extractor(vgg, {
            '8': 'feat1',   # conv2_2
            '16': 'feat2',  # conv3_3
            '23': 'feat3'   # conv4_3
        })
        for p in self.vgg_extractor.parameters():
            p.requires_grad = False

    def perceptual_loss(self, pred, target):
        # 反归一化到[0,1]
        pred = (pred * 0.5 + 0.5).clamp(0, 1)
        target = (target * 0.5 + 0.5).clamp(0, 1)
        device = pred.device
        self.vgg_extractor = self.vgg_extractor.to(device)

        feat_pred = self.vgg_extractor(pred)
        feat_target = self.vgg_extractor(target)

        # 多层次感知损失
        loss = 0
        weights = [1.0, 0.8, 0.6]  # 不同层的权重
        for i, (feat_name, weight) in enumerate(zip(['feat1', 'feat2', 'feat3'], weights)):
            loss += weight * F.l1_loss(feat_pred[feat_name], feat_target[feat_name])

        return loss / len(weights)

    def forward(self, outputs, targets):
        losses = {}
        total_loss = 0

        # 分割损失
        if 'visible_mask' in outputs and 'visible_mask' in targets:
            seg_loss = self.seg_loss(outputs['visible_mask'], targets['visible_mask'])
            losses['segmentation'] = seg_loss
            total_loss += self.seg_weight * seg_loss

        # RGB重建损失
        if 'denoised_rgb' in outputs and 'target_rgb' in targets:
            rgb_loss = self.rgb_loss(outputs['denoised_rgb'], targets['target_rgb'])
            losses['rgb_reconstruction'] = rgb_loss
            total_loss += self.rgb_weight * rgb_loss

            # L1损失增强细节
            l1_loss = self.l1_loss(outputs['denoised_rgb'], targets['target_rgb'])
            losses['l1_detail'] = l1_loss
            total_loss += self.perceptual_weight * l1_loss

        # 一致性损失
        if 'visible_mask' in outputs and 'denoised_rgb' in outputs and 'target_rgb' in targets:
            mask = outputs['visible_mask']
            pred_rgb = outputs['denoised_rgb']
            target_rgb = targets['target_rgb']

            masked_pred = pred_rgb * mask
            masked_target = target_rgb * mask
            consistency_loss = self.l1_loss(masked_pred, masked_target)

            losses['consistency'] = consistency_loss
            total_loss += self.consistency_weight * consistency_loss
        if 'final_result' in outputs:
            tv_loss = total_variation_loss(outputs['final_result'])
            losses['tv'] = tv_loss
            total_loss += 0.05 * tv_loss
        if 'denoised_rgb' in outputs and 'target_rgb' in targets:
            vgg_loss = self.perceptual_loss(outputs['denoised_rgb'], targets['target_rgb'])
            losses['vgg'] = vgg_loss
            total_loss += self.vgg_weight * vgg_loss
        if 'final_result' in outputs and 'target_rgb' in targets:
            fusion_loss = self.rgb_loss(outputs['final_result'], targets['target_rgb'])
            losses['fusion'] = fusion_loss
            total_loss += self.fusion_weight * fusion_loss

        losses['total'] = total_loss
        return losses


def create_apple_completion_model(base_channels=64, time_dim=256, num_timesteps=1000):
    """创建苹果补全模型"""
    model = AppleCompletionNetwork(
        in_channels=3,
        base_channels=base_channels,
        time_dim=time_dim,
        num_timesteps=num_timesteps
    )
    return model


def train_step(model, occluded_batch, complete_batch, visible_mask_batch, optimizer, criterion):
    """修复后的训练步骤"""
    model.train()
    optimizer.zero_grad()
    # B = occluded_batch.shape[0]
    B = complete_batch.shape[0]
    # device = occluded_batch.device
    device = complete_batch.device

    # 确保调度器在正确设备上
    model.module.scheduler = model.module.scheduler.to(device)

    # print("time_step:",model.module.scheduler.num_timesteps)
    # 随机时间步
    timesteps = torch.randint(0, model.module.scheduler.num_timesteps, (B,), device=device)
    # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

    # 使用标准DDPM噪声添加
    noise = torch.randn_like(complete_batch)
    noisy_complete = model.module.scheduler.add_noise(complete_batch, noise, timesteps)


    # 前向传播
    outputs = model(occluded_batch, noisy_complete, timesteps, mode='both')

    # 准备目标
    targets = {
        'visible_mask': visible_mask_batch,
        'target_rgb': complete_batch
    }

    # 计算损失
    losses = criterion(outputs, targets)
    losses['total'].backward()
    optimizer.step()

    return losses

# 修改 train_step 函数，仅训练分割分支
def train_step_segmentation(model, occluded_batch, visible_mask_batch, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    # 只训练分割分支
    outputs = model(occluded_batch, mode='seg')

    # 准备目标
    targets = {'visible_mask': visible_mask_batch}

    # 计算分割损失
    losses = criterion(outputs, targets)
    losses['total'].backward()
    optimizer.step()

    return losses
