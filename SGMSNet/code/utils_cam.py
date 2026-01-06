import torch
import torch.nn.functional as F

def compute_cam(feature_map, target_size):
    # feature_map: (B, C, H, W)
    cam = feature_map.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam  # (B, 1, H, W) 