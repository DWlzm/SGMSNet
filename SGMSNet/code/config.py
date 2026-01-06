import os
import torch

# Get the absolute path of the directory containing config.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据路径 (相对于 train.py 和 evaluate_visualize.py 所在的 My 目录)
# #数据集1
IMAGE_DIR = os.path.join(CURRENT_DIR, 'data', 'ETIS-LaribPolypDB', 'images')
MASK_DIR = os.path.join(CURRENT_DIR, 'data', 'ETIS-LaribPolypDB', 'masks')
RESULTS_DIR = os.path.join(CURRENT_DIR, 'results', 'ETIS-LaribPolypDB', 'compare') # 用于保存可视化和对比结果


# #数据集2
# IMAGE_DIR = os.path.join(CURRENT_DIR, 'data', 'Kvasir-SEG', 'Kvasir-SEG', 'images')
# MASK_DIR = os.path.join(CURRENT_DIR, 'data', 'Kvasir-SEG', 'Kvasir-SEG', 'masks')
# RESULTS_DIR = os.path.join(CURRENT_DIR, 'results', 'Kvasir-SEG') # 用于保存可视化和对比结果

# 图像属性
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IN_CHANNELS = 3
OUT_CHANNELS = 1 # 二分类息肉分割

# 数据集与加载器配置
VAL_SPLIT = 0.2
BATCH_SIZE = 8 # 根据你的 GPU 显存调整 (原为8，尝试减小以缓解OOM)

# 训练参数
NUM_EPOCHS = 30 # 可按需增加
LEARNING_RATE = 1e-4
# MODEL_FEATURES = [64, 128, 256, 512] # 如果希望从配置中控制模型深度



# 可视化配置
NUM_VISUALIZATION_SAMPLES = 5 # 在 evaluate_visualize.py 中生成多少个样本的可视化

DILATION = 2  # CBAMUNet_Dilated的空洞卷积膨胀率，可根据需要调整
MGDC_DILATION_RATES = [1, 2, 4, 8]  # MGDC多尺度空洞率
MGDC_GROUPS = 4  # MGDC分组数 

# 数据加载配置
NUM_WORKERS = min(os.cpu_count()//2 if os.cpu_count() else 1, 8)  # 数据加载的工作进程数
PIN_MEMORY = True        # 是否使用固定内存

# 模型保存配置
SAVE_BEST_ONLY = True    # 是否只保存最佳模型
SAVE_INTERVAL = 10       # 模型保存间隔（轮数）

# 日志配置
LOG_INTERVAL = 10        # 日志记录间隔（批次）
VISUALIZATION_INTERVAL = 100  # 可视化间隔（批次）

# 损失函数配置
LOSS_ALPHA = 0.5        # 损失函数权重alpha
LOSS_BETA = 0.5         # 损失函数权重beta
DS_WEIGHTS = [1.0, 0.8, 0.6, 0.4]  # 深度监督权重

# 优化器配置
OPTIMIZER = "adam"       # 优化器类型
WEIGHT_DECAY = 1e-5      # 权重衰减
MOMENTUM = 0.9           # 动量（用于SGD）

# 学习率调度器配置
USE_LR_SCHEDULER = True  # 是否使用学习率调度器
LR_SCHEDULER = "cosine"  # 学习率调度器类型
MIN_LR = 1e-6           # 最小学习率
WARMUP_EPOCHS = 5       # 预热轮数

# 早停配置
EARLY_STOPPING = True    # 是否使用早停
PATIENCE = 20           # 早停耐心值
MIN_DELTA = 1e-4        # 最小改善阈值

# 混合精度训练配置
USE_AMP = True          # 是否使用混合精度训练

# 梯度裁剪配置
GRADIENT_CLIP = True    # 是否使用梯度裁剪
MAX_NORM = 1.0         # 最大梯度范数

# 模型评估配置
EVAL_METRICS = ["dice", "iou", "precision", "recall", "f1", "accuracy"]  # 评估指标
THRESHOLD = 0.5         # 预测阈值

# 数据增强配置
USE_AUGMENTATION = True  # 是否使用数据增强
AUGMENTATION_PROB = 0.5  # 数据增强概率

# 随机种子配置
SEED = 42               # 随机种子

# 分布式训练配置
DISTRIBUTED = False     # 是否使用分布式训练
WORLD_SIZE = 1         # 世界大小（GPU数量）
DIST_URL = "tcp://localhost:23456"  # 分布式训练URL 