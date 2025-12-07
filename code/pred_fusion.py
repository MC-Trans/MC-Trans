import csv
import os
import sys

sys.path.append(os.getcwd())
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from FusionModel.fusion_dataloader import FusionDataset
from torch.utils.data import Subset, DataLoader
from FusionModel.model import FusionModel
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 配置参数
config = {
    "roi_size": (128,128,128),
    "data_path": "data/train_table.xlsx",
    "batch_size": 1,
    "lr": 0.01,
    "max_epochs": 50,
    "k_folds": 5,
    "save_split": 1,  # 每隔 save_split 个循环保存一次权重
    "categorical_cols": [
        "性别",
        "分期",
        "T",
        "N",
        "M ",
        "药物",
        "既往手术",
        "既往一二代",
        "既往化疗",
        "既往放疗",
        "联合PD-1/PD-L1",
        "肝M",
        "骨M",
        "脑M",
        "EGFR扩增",
        "19del",
        "19突变",
        "19扩增",
        "21-L858R",
        20,
        "T790M",
        "21其他突变",
        "18突变G719x",
        "TP53突变",
        "KRAS",
        "EZH2",
        "KEAP1",
        "BRAF",
        "BRCA1",
        "LATS1",
        "HER-2",
        "ERBB3扩增",
        "ALK",
        "ATK1",
        "ROS",
        "MET扩增",
        "ATM突变",
        "RET",
        "KIT扩增",
        "SMARCA4",
        "SOX9",
        "CDK4扩增",
        "CDK6",
        "CDK8突变",
        "MDM2扩增",
        "CDKN2A缺失",
        "TGFBR2",
        "PMS2扩增",
        "MYC扩增",
        "PIK3CA",
        "VHL ",
        "GRM3",
        "SRMS",
        "NKX2-1",
        "NTRK3",
        "IRS2",
        "IGF1R",
    ],
    "continuous_cols": [
        "年龄",
        "突变个数",
        "白细胞",
        "单核细胞计数",
        "红细胞",
        "红细胞压积",
        "淋巴细胞计数",
        "平均血小板体积",
        "嗜碱细胞计数",
        "嗜酸细胞计数",
        "血红蛋白",
        "血小板计数",
        "中性粒细胞计数",
        "NLR",
        "白蛋白",
        "白球比",
        "谷氨酰氨转肽酶",
        "谷丙转氨酶",
        "谷草转氨酶",
        "间接胆红素",
        "碱性磷酸酶",
        "前白蛋白",
        "球蛋白",
        "乳酸脱氢酶",
        "直接胆红素",
        "总胆红素",
        "总胆汁酸",
        "总蛋白",
        "β2微球蛋白",
        "肌酐",
        "尿素",
        "尿酸",
        "癌胚抗原",
        "甲胎蛋白",
        "糖类抗原125",
        "糖类抗原199",
        "部分凝血酶原时间",
        "国际标准化比值(INR)",
        "凝血酶时间",
        "凝血酶原时间",
        "纤维蛋白原",
    ],
    "seed_base": 42,
}

def get_table_categories():
    # 定义分类列和连续列（需要根据实际数据调整）
    categorical_cols = config["categorical_cols"]
    continuous_cols = config["continuous_cols"]

    df_train = pd.read_excel(
        config["data_path"], sheet_name="训练集&验证集"
    )  # 读取训练集&验证集的数据
    df_test = pd.read_excel(config["data_path"], sheet_name="测试集")  # 读取测试集
    # 合并训练测试集用于统一处理
    full_df = pd.concat([df_train, df_test], ignore_index=True)
    full_df.fillna(full_df.mean(), inplace=True)

    # 初始化编码器和归一化器
    label_encoders = {}  # 存储各列的LabelEncoder
    scaler = StandardScaler()

    # 处理分类数据
    for col in categorical_cols:
        le = LabelEncoder()
        full_df[col] = le.fit_transform(full_df[col].astype(int))
        label_encoders[col] = le

    # 处理连续列
    cont_data = full_df[continuous_cols].values
    scaler.fit(cont_data)
    cont_data_scaled = scaler.transform(cont_data)
    full_df = np.hstack([full_df[categorical_cols].values, cont_data_scaled])
    return tuple(
        [len(np.unique(full_df[:, i])) for i in range(len(config["categorical_cols"]))]
    )


def accuracy_score(labels, preds):
    return np.mean(np.array(labels) == preds)

def evaluate_test(model, loader, output_test_txt_path, output_test_roc_path):
    model.eval()
    all_outputs, all_labels = [], []
    
    with torch.no_grad():
        for inputs, cat, con, labels in loader:
            inputs = inputs.to(device).float()
            cat = cat.to(device).long()
            con = con.to(device).float()
            labels = labels.float().to(device).float()
            outputs = model(inputs, cat, con).squeeze(1)
            
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_score = np.array(all_outputs)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    y_pred = (y_score >= best_threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
        "best_threshold": best_threshold,
    }

    # 保存指标
    with open(output_test_txt_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.2f}')
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(output_test_roc_path, dpi=300)
    plt.close()

    # 保存分类报告
    report = classification_report(y_true, y_pred)
    with open(output_test_txt_path, "a") as f:
        f.write("\n")
        f.write(report)
    
    return metrics["auc"]


def main(output_test_txt_path="test_fusion_result_fusion_pred.txt", output_test_roc_path="test_fusion_roc_fusion_pred.png", model_path="checkpoints/fusion_best_model.pth"):
    from FusionModel.fusion_dataloader import get_test_dataloader

    swin_model_config = {
        "img_size": (128, 128, 128),
        "num_classes": 1,
        "in_channels": 1,
        "feature_size": 48,
        "drop_rate": 0.5,
        "attn_drop_rate": 0.5,
        "dropout_path_rate": 0.5,
        "use_checkpoint": True,
        "out_channels":768,        
    }
    tab_model_config = {
        "categories": get_table_categories(),
        "num_continuous": len(config["continuous_cols"]),
        "dim": 32,
        "depth": 2,
        "heads": 6,
        "attn_dropout": 0.5,
        "ff_dropout": 0.5,
        "mlp_hidden_mults": (4, 2),
    }
    best_model = FusionModel(
        swin_params=swin_model_config,
        tab_params=tab_model_config,
        num_categories=len(tab_model_config['categories']),  # 分类特征数量
        num_classes=1
    ).to(device)
    
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    test_dataloader = get_test_dataloader("data/train_table.xlsx", batch_size=config['batch_size'], target_shape=(128,128,128))
    auc = evaluate_test(best_model, test_dataloader, output_test_txt_path, output_test_roc_path)
    print(f"Loaded {name}, Test AUC: {auc:.4f}")


if __name__ == "__main__":
    # 添加命令行参数支持
    import argparse
    parser = argparse.ArgumentParser(description='Fusion Model Prediction')
    parser.add_argument('--output_txt', type=str, default='test_fusion_result_fusion_pred.txt', 
                        help='Path to save test results text file')
    parser.add_argument('--output_roc', type=str, default='test_fusion_roc_fusion_pred.png', 
                        help='Path to save ROC curve image')
    parser.add_argument('--model_path', type=str, default='checkpoints/fusion_best_model.pth', 
                        help='Path to trained model weights')
    args = parser.parse_args()
    
    main(output_test_txt_path=args.output_txt, 
         output_test_roc_path=args.output_roc, 
         model_path=args.model_path)