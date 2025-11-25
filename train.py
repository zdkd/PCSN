# python
import os
from glob import glob
import random
import csv
import time
from datetime import datetime

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from my_diff_seg import create_apple_completion_model, AppleCompletionLoss, train_step, train_step_segmentation

def save_training_log_to_csv(log_file, epoch_data):
    """保存训练日志到CSV文件"""
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)

    file_exists = os.path.exists(log_file)

    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=epoch_data.keys())

        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()
            print(f"创建训练日志文件: {log_file}")

        # 写入数据
        writer.writerow(epoch_data)
        print(f"已保存第{epoch_data['epoch']}轮训练数据到CSV文件")

# 1. 数据集定义
class AppleDataset(Dataset):
    def __init__(self, occluded_dir, complete_dir, mask_dir, img_size=256, train_segmentation_only=False):
        self.occluded_paths = sorted(glob(os.path.join(occluded_dir, '*')))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*')))
        self.train_segmentation_only = train_segmentation_only

        if not train_segmentation_only:
            self.complete_paths = sorted(glob(os.path.join(complete_dir, '*')))
            assert len(self.occluded_paths) == len(self.complete_paths) == len(self.mask_paths), "文件数不一致"
        else:
            print("仅训练分割分支，跳过完整图像文件数检查")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # [-1,1]
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.occluded_paths)

    def __getitem__(self, idx):
        occluded = Image.open(self.occluded_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        occluded = self.transform(occluded)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()  # 二值化

        if not self.train_segmentation_only:
            complete = Image.open(self.complete_paths[idx]).convert('RGB')
            complete = self.transform(complete)
            return {
                'occluded_rgb': occluded,
                'complete_rgb': complete,
                'visible_mask': mask
            }
        else:
            return {
                'occluded_rgb': occluded,
                'visible_mask': mask
            }

def save_image(tensor, save_path):
    tensor = tensor.detach().cpu().squeeze(0)
    # 对掩码直接二值化并转为uint8
    if tensor.shape[0] == 1:  # 单通道掩码
        # tensor = (tensor > 0.5).float()
        img = transforms.ToPILImage(mode='L')(tensor)
        img = img.point(lambda x: 255 if x > 0 else 0)  # 强制0/255
    else:
        tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
        img = transforms.ToPILImage()(tensor)
    img.save(save_path)

def visualize_on_test_image(model, device, test_dir, model_ckpt_path, save_path, img_size=256, num_steps=50):
    # 随机选取一张测试图片
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    if not test_images:
        print("测试集为空，无法可视化")
        return
    img_name = random.choice(test_images)
    img_path = os.path.join(test_dir, img_name)

    # 加载图片
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(img_path).convert('RGB')
    input_img = transform(img).unsqueeze(0).to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        results = model.module.complete_apple_with_fusion(input_img, num_steps=num_steps) if hasattr(model, 'module') else model.complete_apple_with_fusion(input_img, num_steps=num_steps)
        output_img = results['final_result']
        # final_result

    # 保存结果
    save_image(output_img, save_path)
    # output_img = output_img.detach().cpu().squeeze(0)
    # output_img = (output_img * 0.5 + 0.5).clamp(0, 1)
    # out_pil = transforms.ToPILImage()(output_img)
    # out_pil.save(save_path)
    print(f"可视化已保存: {save_path}")


# 2. 训练主循环
def main():
    # 使用分割后的数据集路径
    base_data_dir = 'data/split_dataset'

    # 训练集路径
    train_occluded_dir = os.path.join(base_data_dir, 'train', 'occluded')
    train_complete_dir = os.path.join(base_data_dir, 'train', 'complete')
    train_mask_dir = os.path.join(base_data_dir, 'train', 'mask')

    # 验证集路径
    val_occluded_dir = os.path.join(base_data_dir, 'val', 'occluded')
    val_complete_dir = os.path.join(base_data_dir, 'val', 'complete')
    val_mask_dir = os.path.join(base_data_dir, 'val', 'mask')

    # 测试集路径
    test_complete_dir = os.path.join(base_data_dir, 'test', 'complete')
    test_mask_dir = os.path.join(base_data_dir, 'test', 'mask')
    test_occluded_dir = os.path.join(base_data_dir, 'test', 'occluded')

    # 测试集路径（用于可视化）
    test_occluded_dir = os.path.join(base_data_dir, 'test', 'occluded')

    img_size = 256
    batch_size = 8
    num_epochs = 300
    lr = 1e-4
    num_workers = 4
    resume_epoch = 243
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # 创建训练和验证数据集
    train_dataset = AppleDataset(train_occluded_dir, train_complete_dir, train_mask_dir, img_size, train_segmentation_only=False)
    val_dataset = AppleDataset(val_occluded_dir, val_complete_dir, val_mask_dir, img_size, train_segmentation_only=False)
    test_dataset = AppleDataset(test_occluded_dir, test_complete_dir, test_mask_dir, img_size, train_segmentation_only=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # 模型与优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_apple_completion_model()
    # 冻结其他分支
    # for param in model.segmentation_branch.parameters():
    #     param.requires_grad = True  # 确保分割分支可训练
    # for param in model.diffusion_branch.parameters():
    #     param.requires_grad = False
    # for param in model.fusion_module.parameters():
    #     param.requires_grad = False
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = AppleCompletionLoss()

    start_epoch = 0
    if resume_epoch > 0:
        ckpt_path = os.path.join(save_dir, f'epoch_{resume_epoch}.pth')
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"从第{resume_epoch}轮权重加载，继续训练")
            start_epoch = resume_epoch
        else:
            print(f"未找到{ckpt_path}，从头开始训练")

    # 训练循环
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        # ============= 训练阶段 =============
        model.train()
        train_total_loss = 0
        train_seg_loss = 0
        train_rgb_loss = 0
        train_iou = 0
        train_psnr = 0
        train_batches = 0

        for i, batch in enumerate(train_loader):
            occluded_rgb = batch['occluded_rgb'].to(device)
            complete_rgb = batch['complete_rgb'].to(device)
            visible_mask = batch['visible_mask'].to(device)

            losses = train_step(model, occluded_rgb, complete_rgb, visible_mask, optimizer, criterion)

            # 计算评估指标
            with torch.no_grad():
                # 获取模型输出用于评估
                model.eval()
                outputs = model(occluded_rgb, None, None, 'seg')  # 修复DataParallel参数传递
                pred_mask = outputs['visible_mask']

                # 计算IoU (分割分支评估)
                iou_score = compute_iou(pred_mask, visible_mask)
                train_iou += iou_score

                # 如果有扩散输出，计算PSNR (生成分支评估)
                if 'denoised_rgb' in outputs:
                    psnr_score = compute_psnr(outputs['denoised_rgb'], complete_rgb)
                    train_psnr += psnr_score
                else:
                    # 单独计算扩散分支的PSNR
                    noise = torch.randn_like(complete_rgb)
                    timesteps = torch.randint(0, 100, (complete_rgb.shape[0],), device=device)
                    noisy_complete = model.module.scheduler.add_noise(complete_rgb, noise, timesteps) if hasattr(model, 'module') else model.scheduler.add_noise(complete_rgb, noise, timesteps)
                    diff_outputs = model(occluded_rgb, noisy_complete, timesteps, 'diffusion')  # 修复DataParallel参数传递
                    psnr_score = compute_psnr(diff_outputs['denoised_rgb'], complete_rgb)
                    train_psnr += psnr_score

                model.train()

            # 累计损失
            train_total_loss += losses['total'].item()
            train_seg_loss += losses.get('segmentation', 0)
            train_rgb_loss += losses.get('rgb_reconstruction', 0)
            train_batches += 1

            if i % 20 == 0:
                print(f"Epoch {epoch} Step {i} | Total Loss: {losses['total'].item():.4f} | "
                      f"Seg: {losses.get('segmentation', 0):.4f} | "
                      f"RGB: {losses.get('rgb_reconstruction', 0):.4f} | "
                      f"IoU: {iou_score:.4f} | PSNR: {psnr_score:.2f}")

        # 计算训练集平均指标
        avg_train_total_loss = train_total_loss / train_batches
        avg_train_seg_loss = train_seg_loss / train_batches
        avg_train_rgb_loss = train_rgb_loss / train_batches
        avg_train_iou = train_iou / train_batches
        avg_train_psnr = train_psnr / train_batches

        # ============= 验证阶段 =============
        model.eval()
        val_total_loss = 0
        val_seg_loss = 0
        val_rgb_loss = 0
        val_iou = 0
        val_psnr = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                occluded_rgb = batch['occluded_rgb'].to(device)
                complete_rgb = batch['complete_rgb'].to(device)
                visible_mask = batch['visible_mask'].to(device)

                # 计算验证损失（不进行反向传播）
                # 修复DataParallel下的参数传递问题
                outputs_seg = model(occluded_rgb, None, None, 'seg')  # 显式传递所有参数
                pred_mask = outputs_seg['visible_mask']

                # 计算扩散损失
                noise = torch.randn_like(complete_rgb)
                timesteps = torch.randint(0, 1000, (complete_rgb.shape[0],), device=device)

                # 确保scheduler在正确的设备上
                if hasattr(model, 'module'):
                    model.module.scheduler = model.module.scheduler.to(device)
                    noisy_complete = model.module.scheduler.add_noise(complete_rgb, noise, timesteps)
                else:
                    model.scheduler = model.scheduler.to(device)
                    noisy_complete = model.scheduler.add_noise(complete_rgb, noise, timesteps)

                diff_outputs = model(occluded_rgb, noisy_complete, timesteps, 'diffusion')  # 显式传递所有参数

                # 合并输出用于损失计算
                combined_outputs = {
                    'visible_mask': pred_mask,
                    'denoised_rgb': diff_outputs['denoised_rgb']
                }

                # 准备目标
                targets = {
                    'visible_mask': visible_mask,
                    'target_rgb': complete_rgb
                }

                # 使用criterion计算所有损失
                losses = criterion(combined_outputs, targets)

                total_loss = losses['total']
                seg_loss = losses.get('segmentation', 0)
                rgb_loss = losses.get('rgb_reconstruction', 0)

                # 计算评估指标
                iou_score = compute_iou(pred_mask, visible_mask)
                psnr_score = compute_psnr(diff_outputs['denoised_rgb'], complete_rgb)

                # 累计验证指标
                val_total_loss += total_loss.item()
                val_seg_loss += seg_loss.item() if isinstance(seg_loss, torch.Tensor) else seg_loss
                val_rgb_loss += rgb_loss.item() if isinstance(rgb_loss, torch.Tensor) else rgb_loss
                val_iou += iou_score
                val_psnr += psnr_score
                val_batches += 1

        # 计算验证集平均指标
        avg_val_total_loss = val_total_loss / val_batches
        avg_val_seg_loss = val_seg_loss / val_batches
        avg_val_rgb_loss = val_rgb_loss / val_batches
        avg_val_iou = val_iou / val_batches
        avg_val_psnr = val_psnr / val_batches

        # 打印训练和验证结果
        print(f"\n=== Epoch {epoch} Summary ===")
        print("训练集:")
        print(f"  Average Total Loss: {avg_train_total_loss:.4f}")
        print(f"  Average Seg Loss: {avg_train_seg_loss:.4f}")
        print(f"  Average RGB Loss: {avg_train_rgb_loss:.4f}")
        print(f"  Average IoU: {avg_train_iou:.4f}")
        print(f"  Average PSNR: {avg_train_psnr:.2f} dB")
        print("验证集:")
        print(f"  Average Total Loss: {avg_val_total_loss:.4f}")
        print(f"  Average Seg Loss: {avg_val_seg_loss:.4f}")
        print(f"  Average RGB Loss: {avg_val_rgb_loss:.4f}")
        print(f"  Average IoU: {avg_val_iou:.4f}")
        print(f"  Average PSNR: {avg_val_psnr:.2f} dB")
        print("=" * 40)

        # 保存训练日志到CSV
        log_file = './train_logs/training_log.csv'
        epoch_data = {
            'epoch': epoch,
            'train_total_loss': avg_train_total_loss,
            'train_seg_loss': avg_train_seg_loss,
            'train_rgb_loss': avg_train_rgb_loss,
            'train_iou': avg_train_iou,
            'train_psnr': avg_train_psnr,
            'val_total_loss': avg_val_total_loss,
            'val_seg_loss': avg_val_seg_loss,
            'val_rgb_loss': avg_val_rgb_loss,
            'val_iou': avg_val_iou,
            'val_psnr': avg_val_psnr,
            'learning_rate': lr,
            'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        save_training_log_to_csv(log_file, epoch_data)

        # 早停机制
        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"验证损失改善，保存最佳模型 (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"验证损失未改善 ({patience_counter}/{patience})")

        # if patience_counter >= patience:
        #     print(f"早停：验证损失连续{patience}个epoch未改善")
        #     break

        # 每个epoch保存模型
        torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pth'))
        vis_save_path = f'./guochengimg/visual_epoch_{epoch}.jpg'
        # 使用测试集进行可视化
        visualize_on_test_image(model, device, test_occluded_dir, None, vis_save_path, num_steps=100)

        # ====== 拼接并保存四张图片 ======
        from torchvision.utils import make_grid, save_image
        # 取一张验证图像进行可视化
        with torch.no_grad():
            val_batch = next(iter(test_loader))
            test_occluded = val_batch['occluded_rgb'].to(device)[:1]
            # 使用改进的采样参数：更多步数，使用DDIM（eta=0）
            results = model.module.complete_apple_with_fusion(test_occluded, num_steps=100, use_eta=0.0) if hasattr(model, 'module') else model.complete_apple_with_fusion(test_occluded, num_steps=100, use_eta=0.0)

            # 比较不同的融合方式
            imgs = [
                test_occluded[0],  # 原始输入图像，用于对比
                results['completed_apple'][0],  # 纯扩散生成结果
                results['final_result'][0],  # 改进的平滑融合结果
                results['visible_mask'][0].repeat(3, 1, 1),  # 可见区域掩码
                results['visible_apple'][0]  # 可见苹果部分
            ]
            imgs = [(img * 0.5 + 0.5).clamp(0,1) for img in imgs]
            grid = make_grid(imgs, nrow=5)
            os.makedirs('./guochengimg', exist_ok=True)
            save_image(grid, f'./guochengimg/epoch_{epoch}_concat.jpg')
        # ==============================

def compute_iou(pred_mask, true_mask, threshold=0.5):
    """计算IoU，输入为tensor，shape: (B, 1, H, W)"""
    pred = (pred_mask > threshold).float()
    true = (true_mask > threshold).float()
    intersection = (pred * true).sum(dim=[1,2,3])
    union = ((pred + true) > 0).float().sum(dim=[1,2,3])
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def compute_psnr(pred_img, true_img, max_val=1.0):
    """计算PSNR，输入为tensor，shape: (B, 3, H, W)，像素范围[0,1]"""
    mse = torch.mean((pred_img - true_img) ** 2, dim=[1,2,3])
    max_val_tensor = torch.tensor(max_val, device=pred_img.device, dtype=pred_img.dtype)
    psnr = 20 * torch.log10(max_val_tensor) - 10 * torch.log10(mse + 1e-8)
    return psnr.mean().item()

if __name__ == '__main__':
    main()
