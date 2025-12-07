import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
from swinTModel.layers.swin3d_layer import SwinTransformerForClassification

# 指定使用的GPU设备（显卡1）
device = torch.device('cuda:0')

class SwinTransformerForFeature(SwinTransformerForClassification):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.classifier

    def forward(self, x):
        x = self.swin_transformer(x)
        x = self.global_avg_pool(x[-1])
        return torch.flatten(x, 1)

class TabTransformerForFeature(TabTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.original_mlp = self.mlp
        self.mlp = nn.Identity()

    def forward(self, x_categ, x_cont):
        return super().forward(x_categ, x_cont)

class FusionModel(nn.Module):
    def __init__(
        self, swin_params, tab_params, num_categories, num_classes, fusion_dim=512, dropout=0.2
    ):
        super().__init__()

        # 初始化模型时直接创建在目标设备上
        self.swin = SwinTransformerForFeature(**swin_params)
        self.tab_transformer = TabTransformerForFeature(**tab_params)

        # 动态获取特征维度（直接在目标设备上测试）
        with torch.no_grad():
            dummy_ct = torch.randn(2, 1, *swin_params["img_size"])
            swin_feat_dim = self.swin(dummy_ct).shape[1]
            
            categories = tab_params["categories"]
            dummy_categ = torch.stack([
                torch.randint(0, num_classes, (2,)) 
                for num_classes in categories
            ], dim=1).long()

            # dummy_categ = torch.randint(0, 10, (2, num_categories))
            dummy_cont = torch.randn(2, tab_params["num_continuous"])
            tab_feat_dim = self.tab_transformer(dummy_categ, dummy_cont).shape[1]

        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Linear(swin_feat_dim + tab_feat_dim, fusion_dim),
            # nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            # nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, ct_input, tab_categorical, tab_continuous):
        # 输入数据已在目标设备，直接前向传播
        ct_feat = self.swin(ct_input)
        tab_feat = self.tab_transformer(tab_categorical, tab_continuous)
        combined = torch.cat([ct_feat, tab_feat], dim=1)
        fused = self.fusion(combined)
        return self.classifier(fused)