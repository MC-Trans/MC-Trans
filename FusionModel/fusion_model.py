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
        self.swin = SwinTransformerForFeature(**swin_params).to(device)
        self.tab_transformer = TabTransformerForFeature(**tab_params).to(device)

        # 动态获取特征维度（直接在目标设备上测试）
        with torch.no_grad():
            dummy_ct = torch.randn(2, 1, *swin_params["img_size"], device=device)
            swin_feat_dim = self.swin(dummy_ct).shape[1]
            
            dummy_categ = torch.randint(0, 10, (2, num_categories), device=device)
            dummy_cont = torch.randn(2, tab_params["num_continuous"], device=device)
            tab_feat_dim = self.tab_transformer(dummy_categ, dummy_cont).shape[1]

        self.ct_proj = nn.Sequential(
            nn.Linear(swin_feat_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
        ).to(device)
        
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_feat_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
        ).to(device)

        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 2),
            nn.Softmax(dim=-1),
        ).to(device)

        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        ).to(device)

        self.classifier = nn.Linear(fusion_dim // 2, num_classes).to(device)

    def forward(self, ct_input, tab_categorical, tab_continuous):
        ct_feat = self.ct_proj(self.swin(ct_input))
        tab_feat = self.tab_proj(self.tab_transformer(tab_categorical, tab_continuous))
        weights = self.gate(torch.cat([ct_feat, tab_feat], dim=1))
        merged = weights[:, 0:1] * ct_feat + weights[:, 1:2] * tab_feat
        fused = self.fusion(torch.cat([ct_feat, tab_feat], dim=1) + merged.repeat(1, 2))
        return self.classifier(fused)

if __name__ == "__main__":
    swin_config = {
        "img_size": (128, 128, 128),
        "num_classes": 1,
        "in_channels": 1,
        "feature_size": 48,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.1,
        "dropout_path_rate": 0.1,
        "use_checkpoint": True,
        "out_channels":768,        
    }

    tab_config = {
        "categories": (10, 5, 8),  # 对应3个分类特征
        "num_continuous": 15,
        "dim": 32,
        "depth": 4,
        "heads": 8,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
    }

    # 初始化模型
    model = FusionModel(
        swin_params=swin_config,
        tab_params=tab_config,
        num_categories=3,  # 分类特征数量
        num_classes=2
    ).eval().to(device)

    # 生成测试数据（直接在目标设备创建）
    ct_input = torch.randn(1, 1, 128, 128, 128, device=device)
    tab_categorical = torch.randint(0, 10, (1, 3), device=device)  # 3个分类特征
    tab_continuous = torch.randn(1, 15, device=device)

    # 前向传播
    output = model(ct_input, tab_categorical, tab_continuous)
    print(f"Output shape: {output.shape}")  # 预期输出: torch.Size([1, 2])
    print("模型运行成功！")