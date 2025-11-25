# Segmentation-Guided Diffusion for Pomegranate Image Completion
本仓库包含论文“A Conditional Segmentation-guided Network for Pomegranate Image Completion under Occlusion”（已被接收发表）中相关的实现与实验脚本。该工程实现了基于分割引导的扩散/去噪模型，用于在有遮挡情况下对石榴（pomegranate）等目标图像进行补全（image completion）

核心设计思想
- 使用分割（segmentation）分支作为条件引导，帮助扩散（diffusion）分支在保持语义一致性的同时更准确地复原被遮挡区域。
