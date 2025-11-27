# Segmentation-Guided Diffusion for Pomegranate Image Completion
本仓库包含论文“A Conditional Segmentation-guided Network for Pomegranate Image Completion under Occlusion”（已被接收发表）中相关的实现与实验脚本。该工程实现了基于分割引导的扩散/去噪模型，用于在有遮挡情况下对石榴（pomegranate）等目标图像进行补全（image completion）

核心设计思想
- 使用分割（segmentation）分支作为条件引导，帮助扩散（diffusion）分支在保持语义一致性的同时更准确地复原被遮挡区域。

数据集结构
data/ 
└── split_dataset/ 
├── train/ 
│ ├── occluded/ 
│ ├── complete/ 
│ └── mask/ 
│
├── val/ 
│ ├── occluded/ 
│ ├── complete/ 
│ └── mask/ 
│ 
└── test/ 
├── occluded/ 
├── complete/ 
└── mask/ 


对外贡献与许可证
- 本仓库按照原论文及作者意愿开源（请在使用时引用下方论文）。
- 请在遵循原作者许可的前提下使用与修改代码；若需添加许可证文件，请联系作者或依据目标期刊/机构要求添加（当前仓库含 LICENSE 文件，可查阅）。

引用
如果您使用本代码或引用本工作，请引用论文：

```bib
@article{zhang_conditional_2025,
	title = {A conditional segmentation-guided network for pomegranate image completion under occlusion},
	volume = {21},
	issn = {1746-4811},
	url = {https://doi.org/10.1186/s13007-025-01476-4},
	doi = {10.1186/s13007-025-01476-4},
	abstract = {In agricultural images acquired under natural conditions, pomegranate fruits are often partially occluded by leaves and branches, resulting in missing structural information that compromises the accuracy of yield estimation and automated harvesting. To overcome the challenges of recovering structural integrity in occluded agricultural imagery, we propose the Conditional Segmentation-guided Diffusion Network (CSD-Net). CSD-Net is a lightweight, unified framework, representing the first conditional diffusion model specifically designed for the joint tasks of pomegranate image completion and segmentation. CSD-Net aims to address the structural fidelity limitations of traditional completion methods. It utilizes a shared encoder, a segmentation branch, and an RGB diffusion branch. Crucially, the network leverages the segmentation mask as a key structural prior condition to guide the diffusion generation process. This innovative conditional guidance mechanism ensures high-fidelity reconstruction of fruit structures while maintaining spatial and textural consistency. Experimental results demonstrate that CSD-Net substantially outperforms conventional methods across metrics, achieving 30.37 dB in PSNR and 0.9490 in SSIM. Furthermore, its model size is only 117 MB, striking an effective balance between high completion quality and inference efficiency. This study offers a novel and highly effective solution for mitigating occlusion issues in agricultural visual perception tasks. Upon acceptance of this paper, the source code will be made publicly available at https://github.com/zdkd/PCSN.},
	number = {1},
	journal = {Plant Methods},
	author = {Zhang, Duokuo and Hou, Ruizhe and Guo, Jingjing and Zhao, Mingfu and Wang, Qi and Luo, Zhen and Xu, Kun},
	month = nov,
	year = {2025},
	pages = {153},
}
```
