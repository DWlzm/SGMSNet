import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from code.compare_models.unet import UNet,CombinedLoss

import config
from utils import PolypDataset, Metrics

def save_visualized_predictions(model, loader, device, num_images, output_dir, model_name_str, threshold):
    print(f"为模型 {model_name_str} 生成可视化预测...")
    model.eval()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    with torch.no_grad():
        for batch_idx, (images, masks_gt) in enumerate(loader):
            if images is None or masks_gt is None:
                print(f"跳过批次 {batch_idx} 因为数据加载错误。")
                continue

            images, masks_gt = images.to(device), masks_gt.to(device)
            outputs = model(images)
            outputs_sigmoid = torch.sigmoid(outputs)
            preds = (outputs_sigmoid > threshold).float()

            # 将张量移至CPU并转换为NumPy数组
            # images_np = images.cpu().numpy() # 原始图像，如果需要保存
            masks_gt_np = masks_gt.cpu().numpy().squeeze(axis=1) # (B, H, W)
            preds_np = preds.cpu().numpy().squeeze(axis=1)       # (B, H, W)

            for i in range(images.size(0)):
                if count >= num_images:
                    break

                gt = masks_gt_np[i]  # (H, W)
                pred = preds_np[i] # (H, W)
                
                h, w = gt.shape
                visualization_mask = np.zeros((h, w, 3), dtype=np.uint8)

                # TP (白色), TN (黑色), FP (红色), FN (蓝色)
                for r_idx in range(h):
                    for c_idx in range(w):
                        if gt[r_idx, c_idx] == 1 and pred[r_idx, c_idx] == 1: # TP
                            visualization_mask[r_idx, c_idx] = [255, 255, 255] # 白色
                        elif gt[r_idx, c_idx] == 0 and pred[r_idx, c_idx] == 0: # TN
                            visualization_mask[r_idx, c_idx] = [0, 0, 0]       # 黑色
                        elif gt[r_idx, c_idx] == 0 and pred[r_idx, c_idx] == 1: # FP (多出)
                            visualization_mask[r_idx, c_idx] = [255, 0, 0]     # 红色
                        elif gt[r_idx, c_idx] == 1 and pred[r_idx, c_idx] == 0: # FN (少预测)
                            visualization_mask[r_idx, c_idx] = [0, 0, 255]     # 蓝色
                
                img_pil = Image.fromarray(visualization_mask)
                img_path = os.path.join(output_dir, f"image_{count:02d}_comparison.png")
                img_pil.save(img_path)
                count += 1
            
            if count >= num_images:
                break
    print(f"可视化图像已保存到 {output_dir}")

def train_one_epoch(model, loader, optimizer, criterion, device, model_name=None):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, masks) in enumerate(loader):
        if images is None or masks is None:
            print("跳过因加载错误导致的批次。")
            continue
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        
        if config.GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_NORM)
            
        optimizer.step()
        epoch_loss += loss.item()
        
        if batch_idx % config.LOG_INTERVAL == 0:
            print(f'Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}')
    
    return epoch_loss / len(loader) if len(loader) > 0 else 0

def evaluate(model, loader, device, criterion):
    model.eval()
    val_loss = 0
    all_preds_list = []
    all_targets_list = []

    with torch.no_grad():
        for images, masks in loader:
            if images is None or masks is None:
                continue
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            outputs_sigmoid = torch.sigmoid(outputs)
            preds = (outputs_sigmoid > config.THRESHOLD).float()

            all_preds_list.append(preds.cpu())
            all_targets_list.append(masks.cpu())
    
    avg_val_loss = val_loss / len(loader) if len(loader) > 0 else 0

    if not all_preds_list or not all_targets_list:
        print("警告：没有数据可以评估。返回空指标。")
        empty_metrics = {'Loss': avg_val_loss, 'Dice': 0, 'IoU': 0, 'Precision': 0, 'Recall': 0, 'F1': 0, 'Accuracy': 0}
        return empty_metrics

    all_preds_tensor = torch.cat(all_preds_list, dim=0)
    all_targets_tensor = torch.cat(all_targets_list, dim=0)
    
    metrics_summary = Metrics.calculate_all_metrics(all_preds_tensor, all_targets_tensor)
    metrics_summary['Loss'] = avg_val_loss
    
    return metrics_summary

def save_cam_visualizations(model, loader, device, output_dir, model_name_str):
    print(f"为模型 {model_name_str} 生成专家CAM热图...")
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for batch_idx, (images, masks_gt) in enumerate(loader):
            images = images.to(device)
            # 获取专家特征
            expert_feats = model.get_expert_features(images)
            # 只可视化第一个样本
            for i, feat in enumerate(expert_feats):
                # feat: [B, C, H, W]，取第一个batch和第一个通道
                cam = feat[0].mean(0).cpu().numpy()  # [H, W]
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                cam = np.uint8(255 * cam)
                cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
                expert_name = ["CoordAtt", "SKBlock", "Illumination"][i]
                img_path = os.path.join(output_dir, f"img{count:02d}_{expert_name}_cam.png")
                cv2.imwrite(img_path, cam)
            count += 1
            if count >= 15:
                break
    print(f"专家CAM热图已保存到 {output_dir}")

def main():
    # 设置随机种子
    set_seed(config.SEED)
    
    # 确定设备和可用GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"发现 {num_gpus} 个GPU。使用CUDA。")
        device = config.DEVICE
        if num_gpus == 0:
            print("CUDA可用但未找到GPU。使用CPU。")
            device = torch.device("cpu")
            num_gpus = 0
    else:
        print("CUDA不可用。使用CPU。")
        device = torch.device("cpu")
        num_gpus = 0

    # 数据加载和预处理
    dataset_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])

    full_dataset = PolypDataset(config.IMAGE_DIR, config.MASK_DIR, transform=dataset_transform)

    val_size = int(len(full_dataset) * config.VAL_SPLIT)
    if val_size == 0 and len(full_dataset) > 0:
        val_size = 1
    if val_size > len(full_dataset):
        val_size = len(full_dataset)
    
    train_size = len(full_dataset) - val_size
    
    if train_size <= 0 and len(full_dataset) > 0:
        print("警告：训练集大小为0或负数。如果可能，使用完整数据集进行验证，否则退出。")
        if len(full_dataset) > 0:
            train_dataset = None
            val_dataset = full_dataset
        else:
            print("错误：数据集为空。")
            return
    elif len(full_dataset) == 0:
        print("错误：数据集为空。")
        return
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    if train_dataset:
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                                num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    else:
        train_loader = None
        print("没有训练数据，跳过训练加载器创建。")

    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                          num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
     #对比实验
    model_configurations = [   
        # ('PAN', PAN),  # 直接引用 PAN 类
        # ('HRNet',HRNet),
        # ('SAM',SAM), 
        ('UNet',UNet),
        # ('DeepLabV3Plus_OS16',DeepLabV3Plus_OS16),
        # ('PSPNet',PSPNet),
        # ('SegNet', SegNet),
        # ('FCN_ResNet50',FCN_ResNet50),
        # ('MoEUNetV2',MoEUNetV2),
        # ('MoEUNetV3',MoEUNetV3),
    ]      


    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 为可视化创建一个基础目录
    visualizations_base_dir = os.path.join(config.RESULTS_DIR, "visualizations")
    if not os.path.exists(visualizations_base_dir):
        os.makedirs(visualizations_base_dir, exist_ok=True)

    for model_name, ModelClass in model_configurations:
        print(f'\n===== 训练 {model_name} =====')
        
        # 调用lambda函数来创建模型实例
        model = ModelClass(config.IN_CHANNELS, config.OUT_CHANNELS)
        model.to(device)
        
        if num_gpus > 1 and device.type == 'cuda':
            print(f"在 {num_gpus} 个GPU上使用DataParallel。")
            model = nn.DataParallel(model)
            
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        
        # 使用适合的损失函数
        if model_name == 'MoEUNetPlusPlus':
            criterion = CombinedLoss(alpha=config.LOSS_ALPHA, beta=config.LOSS_BETA)
            
        best_metric_val = 0
        best_metrics = None
        log_path = os.path.join(config.RESULTS_DIR, f'{model_name}.txt')
        
        with open(log_path, 'w') as log_file:
            log_file.write(f'{model_name} 的训练日志\n')
            log_file.write('Epoch\tTrainLoss\tValLoss\tValDice\tValIoU\n')
            
            if train_loader is None:
                print(f"{model_name} 没有训练数据，跳过训练循环。")
            else:
                for epoch in range(config.NUM_EPOCHS):
                    train_loss = train_one_epoch(
                        model, train_loader, optimizer, criterion, device,
                        model_name=model_name
                    )
                    
                    metrics = evaluate(model, val_loader, device, criterion)
                    
                    log_line = f'{epoch+1}\t{train_loss:.4f}\t{metrics["Loss"]:.4f}\t{metrics["Dice"]:.4f}\t{metrics["IoU"]:.4f}\n'
                    print(f'Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {metrics["Loss"]:.4f} | Val Dice: {metrics["Dice"]:.4f} | Val IoU: {metrics["IoU"]:.4f}')
                    log_file.write(log_line)
                    
                    current_dice = metrics['Dice']
                    if current_dice > best_metric_val:
                        best_metric_val = current_dice
                        best_metrics = metrics.copy()
                        if config.SAVE_BEST_ONLY:
                            model_save_path = os.path.join(config.RESULTS_DIR, f'{model_name}_best.pth')
                            try:
                                if isinstance(model, nn.DataParallel):
                                    torch.save(model.module.state_dict(), model_save_path)
                                else:
                                    torch.save(model.state_dict(), model_save_path)
                                print(f'最佳模型已保存到 {model_save_path}')
                            except Exception as e:
                                print(f"保存模型 {model_name} 时出错: {e}")
            
            print(f'\n完成 {model_name} 的训练。')
            
            print(f"\n{model_name} 的最佳模型指标:")
            if best_metrics:
                for metric_name, metric_value in best_metrics.items():
                    print(f"{metric_name}: {metric_value:.4f}")
                
                result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result.txt')
                result_exists = os.path.exists(result_path)
                
                with open(result_path, 'a' if result_exists else 'w') as result_file:
                    if result_exists:
                        result_file.write("\n\n")
                    
                    result_file.write(f"模型名称: {model_name}\n")
                    result_file.write("======================\n")
                    for metric_name, metric_value in best_metrics.items():
                        result_file.write(f"{metric_name}: {metric_value:.4f}\n")
                    result_file.write("======================")
                
                print(f"训练结果已保存到 {result_path}")
            else:
                if train_loader is None and len(val_dataset) > 0:
                    print(f"由于没有进行训练，在验证集上评估 {model_name}。")
                    final_metrics = evaluate(model, val_loader, device, criterion)
                    for metric_name, metric_value in final_metrics.items():
                        print(f"{metric_name}: {metric_value:.4f}")
                    
                    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result.txt')
                    result_exists = os.path.exists(result_path)
                    
                    with open(result_path, 'a' if result_exists else 'w') as result_file:
                        if result_exists:
                            result_file.write("\n\n")
                        
                        result_file.write(f"模型名称: {model_name} (仅评估)\n")
                        result_file.write("======================\n")
                        for metric_name, metric_value in final_metrics.items():
                            result_file.write(f"{metric_name}: {metric_value:.4f}\n")
                        result_file.write("======================")
                    
                    print(f"评估结果已保存到 {result_path}")
                else:
                    print("没有可用的指标。")

            # 在模型处理完毕后，生成并保存可视化预测
            if val_loader and len(val_loader.dataset) > 0:
                current_model_vis_dir = os.path.join(visualizations_base_dir, model_name)
                # 确保使用的是加载到正确设备上的模型实例
                model_to_visualize = model.module if isinstance(model, nn.DataParallel) else model
                save_visualized_predictions(model_to_visualize, val_loader, device, 15, current_model_vis_dir, model_name, config.THRESHOLD)
                # 新增CAM可视化
                cam_vis_dir = os.path.join(config.RESULTS_DIR, "cam_visualizations", model_name)
                if hasattr(model_to_visualize, "get_expert_features"):
                    save_cam_visualizations(model_to_visualize, val_loader, device, cam_vis_dir, model_name)
            else:
                print(f"没有验证数据加载器或验证数据集为空，跳过 {model_name} 的可视化。")

if __name__ == "__main__":
    main()