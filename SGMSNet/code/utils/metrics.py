import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Metrics:
    @staticmethod
    def calculate_dice(pred, target):
        smooth = 1e-5
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    @staticmethod
    def calculate_iou(pred, target):
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)
    
    @staticmethod
    def calculate_pixel_metrics(pred, target):
        pred_flat = pred.view(-1).cpu().numpy()
        target_flat = target.view(-1).cpu().numpy()
        
        precision = precision_score(target_flat, pred_flat, zero_division=0)
        recall = recall_score(target_flat, pred_flat, zero_division=0)
        f1 = f1_score(target_flat, pred_flat, zero_division=0)
        accuracy = accuracy_score(target_flat, pred_flat)
        
        return precision, recall, f1, accuracy
    
    @staticmethod
    def calculate_all_metrics(pred, target):
        dice = Metrics.calculate_dice(pred, target)
        iou = Metrics.calculate_iou(pred, target)
        precision, recall, f1, accuracy = Metrics.calculate_pixel_metrics(pred, target)
        
        return {
            'Dice': dice.item(),
            'IoU': iou.item(),
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy
        } 