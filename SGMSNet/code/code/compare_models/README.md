# 分割模型对比实验

本目录包含用于对比实验的分割模型实现，包括FCN、DeepLabv3+和SegNet。这些模型是图像分割领域的经典和先进模型，可以与其他自定义模型进行性能对比。

## 模型说明

### 1. FCN (全卷积网络)

FCN是第一个端到端的分割网络，将分类网络改造为分割任务。

- **特点**：通过跳跃连接结合不同尺度的特征，保留位置信息
- **骨干网络选项**：ResNet50或VGG16
- **参数量**：取决于骨干网络 (ResNet50: ~23M, VGG16: ~134M)

```python
from code.compare_models.fcn import FCN

# 使用ResNet50作为骨干网络
model_resnet = FCN(in_channels=3, out_channels=1, backbone='resnet50')

# 使用VGG16作为骨干网络
model_vgg = FCN(in_channels=3, out_channels=1, backbone='vgg16')
```

### 2. DeepLabv3+

DeepLabv3+是语义分割领域的先进模型，结合了空洞空间金字塔池化(ASPP)和编码器-解码器结构。

- **特点**：使用空洞卷积捕获多尺度上下文信息，结合低层特征恢复边界细节
- **输出步幅选项**：8或16
- **参数量**：~41M

```python
from code.compare_models.deeplabv3plus import DeepLabV3Plus

# 输出步幅为16的配置
model_os16 = DeepLabV3Plus(in_channels=3, out_channels=1, output_stride=16)

# 输出步幅为8的配置(特征分辨率更高，但需要更多内存)
model_os8 = DeepLabV3Plus(in_channels=3, out_channels=1, output_stride=8)
```

### 3. SegNet

SegNet是一个端到端的编码器-解码器架构，特别适用于场景理解应用。

- **特点**：对称的编码器-解码器结构，保留池化索引进行上采样
- **骨干网络**：VGG16（作为编码器）
- **参数量**：~29M

```python
from code.compare_models.segnet import SegNet

model = SegNet(in_channels=3, out_channels=1)
```

## 使用方法

这些模型已集成到训练脚本(train.py)中，可以通过以下方式使用：

1. 在train.py中取消注释相应的模型配置
2. 运行train.py进行训练和评估

```bash
python train.py
```

## 模型比较

各模型具有不同的特点和优势：

- **FCN**：结构简单，易于理解和实现，适合作为基线模型
- **DeepLabv3+**：在边界精度和多尺度目标捕获方面表现优异，适合高精度需求
- **SegNet**：内存效率较高，适合资源有限的场景

根据具体的应用需求和资源限制，可以选择合适的模型进行实验。 