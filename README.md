# SGMSNet: Spatial-Gated Memory Network for Chest X-ray Image Segmentation

Complex anatomical structures, blurred boundaries, and significant scale variations make automatic segmentation of chest X-ray images a highly challenging task. This paper proposes a novel segmentation network, SGMSNet, to improve segmentation accuracy and robustness. SGMSNet incorporates a Memory Module in the high-level encoder layers to enhance the modeling of global context and long-range dependencies, enabling dynamic integration of semantic information. The decoder adopts a multi-stage upsampling structure and introduces a Gated Attention mechanism at skip connections to adaptively suppress redundant feature information. In addition, a Spatial Attention Module is employed to further strengthen the model’s focus on critical anatomical regions. Experimental results on publicly available chest X-ray segmentation datasets demonstrate that SGMSNet consistently outperforms several state-of-the-art methods in terms of both segmentation accuracy and robustness, validating the effectiveness and superiority of the proposed approach.

If you encounter any issues while using this code repository, you can contact 2033771388@qq.com.

## Paper and Datasets
[Papaer](https://ieeexplore.ieee.org/document/11239364/)

[Datasets:Qata-Cov19](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

## Cite
```
@INPROCEEDINGS{11239364,
  author={Zhongming, Liu and Huang, Xin and Li, Xiao and Zou, Xiang},
  booktitle={2025 19th International Conference on Complex Medical Engineering (CME)}, 
  title={SGMSNet: Spatial-Gated Memory Network for Chest X-ray Image Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={111-115},
  keywords={Image segmentation;Adaptation models;Accuracy;Attention mechanisms;Memory modules;Anatomical structure;Logic gates;Robustness;X-ray imaging;Biomedical imaging;Chest X - ray Segmentation;SGMSNet;Deep Learning;Memory Mechanism},
  doi={10.1109/CME67420.2025.11239364}}

```


![模型框架图](Readme.assets/%E6%A8%A1%E5%9E%8B%E6%A1%86%E6%9E%B6%E5%9B%BE.png)


## MemoryModule
```python
class MemoryModule(nn.Module):
    """记忆模块，用于捕获长距离依赖"""
    def __init__(self, channels, num_memories=64):
        super(MemoryModule, self).__init__()
        self.num_memories = num_memories
        
        # 内存键和值
        self.memory_keys = nn.Parameter(torch.randn(1, num_memories, channels))
        self.memory_values = nn.Parameter(torch.randn(1, num_memories, channels))
        
        # 查询转换
        self.query_transform = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 输出转换
        self.output_transform = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 转换输入为查询
        queries = self.query_transform(x)  # B, C, H, W
        queries = queries.view(b, c, -1).permute(0, 2, 1)  # B, HW, C
        
        # 计算查询和内存键之间的相似度
        keys = self.memory_keys.expand(b, -1, -1)  # B, M, C
        attn_logits = torch.matmul(queries, keys.transpose(1, 2))  # B, HW, M
        attn_weights = F.softmax(attn_logits, dim=2)  # B, HW, M
        
        # 获取内存值
        values = self.memory_values.expand(b, -1, -1)  # B, M, C
        
        # 加权聚合内存值
        memory_output = torch.matmul(attn_weights, values)  # B, HW, C
        memory_output = memory_output.permute(0, 2, 1).view(b, c, h, w)  # B, C, H, W
        
        # 融合原始特征和内存输出
        output = x + memory_output
        output = self.output_transform(output)
        
        return output

```



## GatedAttention
```python


class GatedAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(GatedAttention, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: 主分支特征 (decoder)
        # x: skip特征 (encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

```


## SpatialAttentionModule

```python
class SpatialAttentionModule(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 空间注意力权重
        attn = self.conv1(x)
        attn = F.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        
        # 应用注意力
        out = x * attn
        return out, attn
```
