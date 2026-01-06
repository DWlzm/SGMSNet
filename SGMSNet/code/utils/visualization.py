import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from torchvision.utils import make_grid
import cv2

class TrainingVisualizer:
    def __init__(self, save_dir='results/visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建子目录
        self.metrics_dir = os.path.join(save_dir, 'metrics')
        self.predictions_dir = os.path.join(save_dir, 'predictions')
        self.supervision_dir = os.path.join(save_dir, 'supervision')
        self.experts_dir = os.path.join(save_dir, 'experts')
        
        for d in [self.metrics_dir, self.predictions_dir, self.supervision_dir, self.experts_dir]:
            os.makedirs(d, exist_ok=True)
        
        # 初始化指标记录
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': []
        }
    
    def update_metrics(self, train_loss, val_metrics, epoch):
        """更新并绘制训练指标"""
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_metrics['Loss'])
        self.metrics_history['val_dice'].append(val_metrics['Dice'])
        self.metrics_history['val_iou'].append(val_metrics['IoU'])
        
        # 绘制训练曲线
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(131)
        plt.plot(self.metrics_history['train_loss'], label='Train Loss')
        plt.plot(self.metrics_history['val_loss'], label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Dice系数曲线
        plt.subplot(132)
        plt.plot(self.metrics_history['val_dice'], label='Validation Dice')
        plt.title('Dice Coefficient')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.legend()
        plt.grid(True)
        
        # IoU曲线
        plt.subplot(133)
        plt.plot(self.metrics_history['val_iou'], label='Validation IoU')
        plt.title('IoU Score')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, f'metrics_epoch_{epoch}.png'))
        plt.close()
    
    def visualize_prediction(self, image, mask, prediction, epoch, batch_idx):
        """可视化预测结果"""
        # 转换为numpy数组，确保分离梯度
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
        mask = mask.detach().cpu().numpy().squeeze()
        prediction = prediction.detach().cpu().numpy().squeeze()
        
        # 标准化图像
        image = (image - image.min()) / (image.max() - image.min())
        
        plt.figure(figsize=(15, 5))
        
        # 原始图像
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # 真实掩码
        plt.subplot(132)
        plt.imshow(mask, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        # 预测掩码
        plt.subplot(133)
        plt.imshow(prediction, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.predictions_dir, f'pred_epoch_{epoch}_batch_{batch_idx}.png'))
        plt.close()
    
    def visualize_deep_supervision(self, outputs, epoch, batch_idx):
        """可视化深度监督的各层输出"""
        num_outputs = len(outputs)
        plt.figure(figsize=(3*num_outputs, 3))
        
        for i, output in enumerate(outputs):
            # 获取第一个样本的输出，确保分离梯度
            pred = torch.sigmoid(output[0]).detach().cpu().numpy().squeeze()
            
            plt.subplot(1, num_outputs, i+1)
            plt.imshow(pred, cmap='gray')
            plt.title(f'Level {i}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.supervision_dir, f'supervision_epoch_{epoch}_batch_{batch_idx}.png'))
        plt.close()
    
    def visualize_experts(self, expert_outputs, image, epoch, batch_idx):
        """可视化专家模块的激活图"""
        # 创建自定义热力图颜色映射
        colors = ['#313695', '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8', 
                 '#FFFFBF', '#FEE090', '#FDAE61', '#F46D43', '#D73027', '#A50026']
        custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
        
        plt.figure(figsize=(15, 3))
        
        # 原始图像
        plt.subplot(141)
        img_np = image.detach().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        plt.imshow(img_np)
        plt.title('Original Image')
        plt.axis('off')
        
        # 专家激活图
        for i, (name, activation) in enumerate(expert_outputs.items()):
            if name != 'weights' and name != 'fused':
                act_map = activation[0].mean(dim=0).detach().cpu().numpy()
                act_map = cv2.resize(act_map, (image.shape[2], image.shape[1]))
                
                plt.subplot(1, 4, i+2)
                plt.imshow(act_map, cmap=custom_cmap)
                plt.title(f'{name.capitalize()} Expert')
                plt.axis('off')
                
                # 添加颜色条
                plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experts_dir, f'experts_epoch_{epoch}_batch_{batch_idx}.png'))
        plt.close()
        
        # 绘制专家权重分布
        if 'weights' in expert_outputs:
            weights = expert_outputs['weights'].detach().cpu().numpy()
            plt.figure(figsize=(8, 4))
            sns.barplot(x=['Edge', 'Channel', 'Spatial'], y=weights.mean(axis=0))
            plt.title('Expert Weights Distribution')
            plt.ylabel('Average Weight')
            plt.savefig(os.path.join(self.experts_dir, f'weights_epoch_{epoch}_batch_{batch_idx}.png'))
            plt.close() 