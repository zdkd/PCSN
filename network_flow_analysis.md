# 完整网络流信息分析（单个时间步）

假设输入图像尺寸：**[B, 3, 256, 256]**，base_channels=64

---

## 📊 完整前向传播流程

### 输入数据
```
occluded_rgb:  [B, 3, 256, 256]    # 遮挡的RGB图像
noisy_rgb:     [B, 3, 256, 256]    # 加噪后的RGB图像（扩散模式）
timestep:      [B]                 # 时间步标量
```

---

## 🔄 第一阶段：共享编码器 (SharedEncoder)

### 输入
```
occluded_rgb: [B, 3, 256, 256]
```

### 流程

#### 1. 初始卷积
```python
x = self.conv_in(occluded_rgb)
# 输入:  [B, 3, 256, 256]
# Conv2d(3 → 64, kernel=3, padding=1)
# 输出:  [B, 64, 256, 256]
```

#### 2. Down1 块 (保持尺寸)
```python
skip1 = self.down1(x)
# ResBlock(64 → 64)
# ResBlock(64 → 64)
# 输出:  [B, 64, 256, 256]  ← skip1 保存

x = self.downsample1(skip1)
# Conv2d(64 → 128, kernel=3, stride=2, padding=1)
# 输出:  [B, 128, 128, 128]
```

#### 3. Down2 块 (下采样)
```python
skip2 = self.down2(x)
# ResBlock(128 → 128)
# ResBlock(128 → 128)
# 输出:  [B, 128, 128, 128]  ← skip2 保存

x = self.downsample2(skip2)
# Conv2d(128 → 256, kernel=3, stride=2, padding=1)
# 输出:  [B, 256, 64, 64]
```

#### 4. Down3 块 (下采样)
```python
skip3 = self.down3(x)
# ResBlock(256 → 256)
# ResBlock(256 → 256)
# 输出:  [B, 256, 64, 64]  ← skip3 保存

x = self.downsample3(skip3)
# Conv2d(256 → 512, kernel=3, stride=2, padding=1)
# 输出:  [B, 512, 32, 32]
```

#### 5. 瓶颈层
```python
features = self.bottleneck(x)
# ResBlock(512 → 512)
# AttentionBlock(512)
# ResBlock(512 → 512)
# 输出:  [B, 512, 32, 32]  ← features
```

### 共享编码器输出
```
features: [B, 512, 32, 32]
skips:    [skip1: [B, 64, 256, 256],
           skip2: [B, 128, 128, 128],
           skip3: [B, 256, 64, 64]]
```

---

## 🎭 第二阶段A：分割分支 (SegmentationBranch)

### 输入
```
features: [B, 512, 32, 32]
skips:    [skip1, skip2, skip3]
```

### 流程

#### 1. Up3 (上采样 + 融合)
```python
x = self.up3(features)
# ConvTranspose2d(512 → 256, kernel=2, stride=2)
# 输出:  [B, 256, 64, 64]

x = torch.cat([x, skip3], dim=1)
# Cat along channel: [B, 256, 64, 64] + [B, 256, 64, 64]
# 输出:  [B, 512, 64, 64]

x = self.up_conv3(x)
# ResBlock(512 → 256)
# ResBlock(256 → 256)
# 输出:  [B, 256, 64, 64]
```

#### 2. Up2 (上采样 + 融合)
```python
x = self.up2(x)
# ConvTranspose2d(256 → 128, kernel=2, stride=2)
# 输出:  [B, 128, 128, 128]

x = torch.cat([x, skip2], dim=1)
# Cat: [B, 128, 128, 128] + [B, 128, 128, 128]
# 输出:  [B, 256, 128, 128]

x = self.up_conv2(x)
# ResBlock(256 → 128)
# ResBlock(128 → 128)
# 输出:  [B, 128, 128, 128]
```

#### 3. Up1 (上采样 + 融合)
```python
x = self.up1(x)
# ConvTranspose2d(128 → 64, kernel=2, stride=2)
# 输出:  [B, 64, 256, 256]

x = torch.cat([x, skip1], dim=1)
# Cat: [B, 64, 256, 256] + [B, 64, 256, 256]
# 输出:  [B, 128, 256, 256]

x = self.up_conv1(x)
# ResBlock(128 → 64)
# ResBlock(64 → 64)
# 输出:  [B, 64, 256, 256]
```

#### 4. 输出掩码
```python
visible_mask = torch.sigmoid(self.mask_head(x))
# Conv2d(64 → 1, kernel=1)
# Sigmoid
# 输出:  [B, 1, 256, 256]  ← visible_mask (软掩码，值域[0,1])
```

### 分割分支输出
```
visible_mask: [B, 1, 256, 256]  # 软掩码
```

---

## 🌊 第二阶段B：扩散分支准备

### 1. 多尺度条件融合 (MultiScaleConditionalFusion)

#### 输入
```
features:     [B, 512, 32, 32]   # 8x下采样特征
skip3:        [B, 256, 64, 64]   # 4x下采样特征
skip2:        [B, 128, 128, 128] # 2x下采样特征
skip1:        [B, 64, 256, 256]  # 1x原始特征
visible_mask: [B, 1, 256, 256]
```

#### 处理流程

##### a) 调整掩码到不同尺度
```python
mask_8x = F.interpolate(visible_mask, size=(32, 32), mode='bilinear')
# 输出: [B, 1, 32, 32]

mask_4x = F.interpolate(visible_mask, size=(64, 64), mode='bilinear')
# 输出: [B, 1, 64, 64]

mask_2x = F.interpolate(visible_mask, size=(128, 128), mode='bilinear')
# 输出: [B, 1, 128, 128]

mask_1x = visible_mask
# 输出: [B, 1, 256, 256]
```

##### b) 每个尺度的融合
```python
# 8x尺度融合
fused_8x = self.scale_fusion_8x(torch.cat([features, mask_8x], dim=1))
# 输入: Cat([B, 512, 32, 32] + [B, 1, 32, 32]) = [B, 513, 32, 32]
# Conv2d(513 → 512) + GroupNorm + SiLU
# 输出: [B, 512, 32, 32]

# 4x尺度融合
fused_4x = self.scale_fusion_4x(torch.cat([skip3, mask_4x], dim=1))
# 输入: Cat([B, 256, 64, 64] + [B, 1, 64, 64]) = [B, 257, 64, 64]
# Conv2d(257 → 256) + GroupNorm + SiLU
# 输出: [B, 256, 64, 64]

# 2x尺度融合
fused_2x = self.scale_fusion_2x(torch.cat([skip2, mask_2x], dim=1))
# 输入: Cat([B, 128, 128, 128] + [B, 1, 128, 128]) = [B, 129, 128, 128]
# Conv2d(129 → 128) + GroupNorm + SiLU
# 输出: [B, 128, 128, 128]

# 1x尺度融合
fused_1x = self.scale_fusion_1x(torch.cat([skip1, mask_1x], dim=1))
# 输入: Cat([B, 64, 256, 256] + [B, 1, 256, 256]) = [B, 65, 256, 256]
# Conv2d(65 → 64) + GroupNorm + SiLU
# 输出: [B, 64, 256, 256]
```

#### 输出
```
fused_features: [B, 512, 32, 32]   # fused_8x
fused_skip2:    [B, 256, 64, 64]   # fused_4x
fused_skip1:    [B, 128, 128, 128] # fused_2x
fused_skip0:    [B, 64, 256, 256]  # fused_1x
```

### 2. 自适应时间嵌入 (AdaptiveTimeEmbedding)

#### 输入
```
timestep:     [B]               # 时间步标量
visible_mask: [B, 1, 256, 256]
```

#### 处理流程

##### a) 基础时间嵌入
```python
base_emb = self.base_time_mlp(timestep)
# SinusoidalPositionEmbeddings(256) → [B, 256]
# Linear(256 → 1024) → [B, 1024]
# GELU
# Linear(1024 → 256) → [B, 256]
```

##### b) 条件调制
```python
condition_mod = self.condition_modulator(visible_mask)
# AdaptiveAvgPool2d(1) → [B, 1, 1, 1]
# Flatten → [B, 1]
# Linear(1 → 64) → [B, 64]
# ReLU
# Linear(64 → 256) → [B, 256]
# Sigmoid → [B, 256]
```

##### c) 动态权重生成
```python
condition_global = torch.mean(visible_mask.view(B, -1), dim=1, keepdim=True)
# 输出: [B, 1]

adaptive_weight = self.weight_generator(torch.cat([base_emb, condition_global], dim=1))
# Cat: [B, 256] + [B, 1] = [B, 257]
# Linear(257 → 256) → [B, 256]
# Tanh → [B, 256]
```

##### d) 最终融合
```python
time_emb = base_emb * (1 + condition_mod) + adaptive_weight
# 输出: [B, 256]
```

#### 输出
```
time_emb: [B, 256]  # 自适应时间嵌入
```

---

## 🎨 第三阶段：RGB扩散分支 (RGBDiffusionBranch)

### 输入
```
fused_features: [B, 512, 32, 32]
fused_skips:    [[B, 64, 256, 256], [B, 128, 128, 128], [B, 256, 64, 64]]
visible_mask:   [B, 1, 256, 256]
noisy_rgb:      [B, 3, 256, 256]
time_emb:       [B, 256]
```

### 流程

#### 1. 处理噪声RGB
```python
# 调整噪声图像尺寸到特征尺寸
noise_feat = F.interpolate(noisy_rgb, size=(32, 32), mode='bilinear')
# 输出: [B, 3, 32, 32]

noise_feat = self.noise_conv(noise_feat)
# Conv2d(3 → 64, kernel=3, padding=1)
# 输出: [B, 64, 32, 32]
```

#### 2. 调整掩码尺寸
```python
visible_mask_feat = F.interpolate(visible_mask, size=(32, 32), mode='bilinear')
# 输出: [B, 1, 32, 32]
```

#### 3. 条件融合
```python
x = torch.cat([fused_features, noise_feat, visible_mask_feat], dim=1)
# Cat: [B, 512, 32, 32] + [B, 64, 32, 32] + [B, 1, 32, 32]
# 输出: [B, 577, 32, 32]

x = self.condition_fusion(x)
# Conv2d(577 → 512, kernel=1) + GroupNorm + SiLU
# 输出: [B, 512, 32, 32]
```

#### 4. Up3 (上采样 + skip连接)
```python
x = self.up3(x)
# ConvTranspose2d(512 → 256, kernel=2, stride=2)
# 输出: [B, 256, 64, 64]

x = torch.cat([x, fused_skip2], dim=1)
# Cat: [B, 256, 64, 64] + [B, 256, 64, 64]
# 输出: [B, 512, 64, 64]

x = self.up_conv3[0](x, time_emb)
# ResBlock(512 → 256, time_emb=256)
# 输出: [B, 256, 64, 64]

x = self.up_conv3[1](x)
# AttentionBlock(256)
# 输出: [B, 256, 64, 64]

x = self.up_conv3[2](x, time_emb)
# ResBlock(256 → 256, time_emb=256)
# 输出: [B, 256, 64, 64]
```

#### 5. Up2 (上采样 + skip连接)
```python
x = self.up2(x)
# ConvTranspose2d(256 → 128, kernel=2, stride=2)
# 输出: [B, 128, 128, 128]

x = torch.cat([x, fused_skip1], dim=1)
# Cat: [B, 128, 128, 128] + [B, 128, 128, 128]
# 输出: [B, 256, 128, 128]

x = self.up_conv2[0](x, time_emb)
# ResBlock(256 → 128, time_emb=256)
# 输出: [B, 128, 128, 128]

x = self.up_conv2[1](x, time_emb)
# ResBlock(128 → 128, time_emb=256)
# 输出: [B, 128, 128, 128]
```

#### 6. Up1 (上采样 + skip连接)
```python
x = self.up1(x)
# ConvTranspose2d(128 → 64, kernel=2, stride=2)
# 输出: [B, 64, 256, 256]

x = torch.cat([x, fused_skip0], dim=1)
# Cat: [B, 64, 256, 256] + [B, 64, 256, 256]
# 输出: [B, 128, 256, 256]

x = self.up_conv1[0](x, time_emb)
# ResBlock(128 → 64, time_emb=256)
# 输出: [B, 64, 256, 256]

x = self.up_conv1[1](x, time_emb)
# ResBlock(64 → 64, time_emb=256)
# 输出: [B, 64, 256, 256]
```

#### 7. 输出去噪RGB
```python
denoised_rgb = self.out_conv(x)
# Conv2d(64 → 3, kernel=1)
# 输出: [B, 3, 256, 256]
```

### 扩散分支输出
```
denoised_rgb: [B, 3, 256, 256]  # 去噪后的RGB图像
```

---

## 📦 最终输出汇总

```python
outputs = {
    'visible_mask': [B, 1, 256, 256],    # 可见区域掩码（软掩码）
    'denoised_rgb':  [B, 3, 256, 256],   # 去噪后的RGB图像
    'final_result':  [B, 3, 256, 256]    # 最终输出（与denoised_rgb相同）
}
```

---

## 📊 关键维度变化总结表

| 阶段 | 模块 | 输入维度 | 输出维度 | 分辨率变化 |
|------|------|---------|---------|-----------|
| **输入** | - | [B, 3, 256, 256] | - | 256×256 |
| **编码器** | SharedEncoder | [B, 3, 256, 256] | [B, 512, 32, 32] | 256→32 (8x下采样) |
| **分割分支** | SegmentationBranch | [B, 512, 32, 32] | [B, 1, 256, 256] | 32→256 (8x上采样) |
| **多尺度融合** | MultiScaleConditionalFusion | 多尺度特征 | 融合后的多尺度特征 | 保持不变 |
| **时间嵌入** | AdaptiveTimeEmbedding | [B] + [B, 1, 256, 256] | [B, 256] | 标量→向量 |
| **扩散分支** | RGBDiffusionBranch | [B, 512, 32, 32] | [B, 3, 256, 256] | 32→256 (8x上采样) |
| **输出** | - | - | [B, 3, 256, 256] | 256×256 |

---

## 🔍 关键观察

### 1. 信息流路径
```
输入图像 [B, 3, 256, 256]
    ↓
编码器 (8x下采样) → [B, 512, 32, 32]
    ↓ (分叉)
    ├─→ 分割分支 → [B, 1, 256, 256] (软掩码)
    │                    ↓
    └─→ 多尺度融合 ←─────┘
            ↓
    自适应时间嵌入 ← [B]
            ↓
    扩散分支 (8x上采样) → [B, 3, 256, 256]
```

### 2. Skip连接的作用
- **分割分支**: 使用3个skip连接，保留细节用于精确的掩码预测
- **扩散分支**: 使用3个**融合后**的skip连接，结合了原始特征和掩码信息

### 3. 软掩码的信息传递
- `visible_mask` [B, 1, 256, 256] 包含**连续值** [0, 1]
- 通过多尺度融合模块，软掩码信息被注入到扩散分支的所有层级
- 软掩码的中间值（如0.3, 0.7）隐式编码了边界、置信度等信息

### 4. 时间嵌入的增强
- 基础时间步 [B] → [B, 256]
- 通过 `visible_mask` 调制，生成自适应的时间嵌入
- 时间嵌入在每个ResBlock中调制特征


