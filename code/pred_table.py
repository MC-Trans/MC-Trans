import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, TensorDataset, random_split
from tab_transformer_pytorch import TabTransformer
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 配置参数
config = {
    "data_path": "data/train_table.xlsx",
    "batch_size": 64,
    "lr": 0.001,
    "num_epochs": 50,
    "cv": 5,
    "save_split": 2,  # 每隔 save_split 个循环保存一次权重
    "model_params": {
        "dim": 32,
        "depth": 2,
        "heads": 6,
        "attn_dropout": 0.5,
        "ff_dropout": 0.5,
        "mlp_hidden_mults": (4, 2),
    },
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
}


# 读取数据
def read_data(csv_path):
    """
    读取训练集&验证集的数据, 以及读取测试集的数据
    并将其中的连续性数据进行归一化,
    对于离散型数据, 进行LabelEncoder进行编码
    
    :param csv_path: 训练集&验证集的路径
    :return: X_train_all, y_train_all, X_test_processed, y_test, scaler
    """
    # 定义分类列和连续列（需要根据实际数据调整）
    categorical_cols = config["categorical_cols"]
    continuous_cols = config["continuous_cols"]

    df_train = pd.read_excel(
        csv_path, sheet_name="训练集&验证集"
    )  # 读取训练集&验证集的数据
    df_test = pd.read_excel(csv_path, sheet_name="测试集")  # 读取测试集

    # 移除序号列
    df_train = df_train.drop(columns=["序号"])
    df_test = df_test.drop(columns=["序号"])

    # 分离标签
    y_train = df_train["是否原发耐药"].values
    y_test = df_test["是否原发耐药"].values
    df_train = df_train.drop(columns=["是否原发耐药"])
    df_test = df_test.drop(columns=["是否原发耐药"])

    df_train.fillna(df_train.mean(), inplace=True)
    df_test.fillna(df_test.mean(), inplace=True)

    # 初始化编码器和归一化器
    label_encoders = {}  # 存储各列的LabelEncoder
    scaler = StandardScaler()

    # 处理分类数据
    for col in categorical_cols:
        le = LabelEncoder()
        # 合并训练测试集数据进行编码以保证一致性
        combined = pd.concat([df_train[col], df_test[col]]).astype(int)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col].astype(int))
        df_test[col] = le.transform(df_test[col].astype(int))
        label_encoders[col] = le

    # 处理连续数据
    X_train_cont = df_train[continuous_cols].values
    X_test_cont = df_test[continuous_cols].values

    # 拟合和转换连续数据
    scaler.fit(X_train_cont)
    X_train_cont_scaled = scaler.transform(X_train_cont)
    X_test_cont_scaled = scaler.transform(X_test_cont)

    # 合并特征
    X_train_processed = np.hstack([df_train[categorical_cols].values, X_train_cont_scaled])
    X_test_processed = np.hstack([df_test[categorical_cols].values, X_test_cont_scaled])

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 合并训练集和验证集
    X_train_all = np.concatenate([X_train, X_val], axis=0)
    y_train_all = np.concatenate([y_train, y_val], axis=0)

    return X_train_all, y_train_all, X_test_processed, y_test, scaler


# ======================
# 测试评估模块
# ======================
def evaluate_and_save(
    model, X_test, y_test, scaler, output_test_txt_path, output_test_roc_path
):
    # 创建测试数据集
    test_cat = torch.tensor(
        X_test[:, : len(config["categorical_cols"])], dtype=torch.long
    ).to(device)
    test_cont = torch.tensor(
        X_test[:, len(config["categorical_cols"]) :], dtype=torch.float32
    ).to(device)

    # 预测结果
    model.eval()
    with torch.no_grad():
        outputs = model(test_cat, test_cont)
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()

    fpr, tpr, thresholds = roc_curve(y_test, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    preds = (probs >= best_threshold).astype(int)

    # 计算指标
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs),
        "best_threshold": best_threshold,
    }

    # 保存指标
    with open(output_test_txt_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test, probs)
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
    report = classification_report(y_test, preds)
    with open(output_test_txt_path, "a") as f:
        f.write("\n")
        f.write(report)


def main(output_test_txt_path="test_tab_result_pred.txt", output_test_roc_path="test_tab_roc_pred.png", model_path="checkpoints/tab_best_model.pth"):
    # 读取数据并对其进行编码与归一化, 得到训练集、验证集和测试集的数据和标签
    print(" -------------- 数据读取与预处理 -------------- ")
    X_train_all, y_train_all, X_test, y_test, scaler = read_data(config["data_path"])
    print(X_train_all.shape, y_train_all.shape, X_test.shape, y_test.shape)
    print(" --------------------------------------------- ")

    # 利用训练好的模型, 对测试集进行测试, 将acc,recall,auc值保存到txt中, 将ROC图像保存
    print(" -------------- 模型测试 -------------- ")
    best_model = TabTransformer(
        categories=tuple(
            [
                len(np.unique(X_train_all[:, i]))
                for i in range(len(config["categorical_cols"]))
            ]
        ),
        num_continuous=len(config["continuous_cols"]),
        **config["model_params"],
        mlp_act=nn.ReLU(),
        dim_out=len(np.unique(y_train_all)),  # 自动检测类别数
    ).to(device)
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    evaluate_and_save(
        best_model, X_test, y_test, scaler, output_test_txt_path, output_test_roc_path
    )
    print(" --------------------------------------------- ")


if __name__ == "__main__":
    # 添加命令行参数支持
    import argparse
    parser = argparse.ArgumentParser(description='Tab Transformer Model Prediction')
    parser.add_argument('--output_txt', type=str, default='test_tab_result_pred.txt', 
                        help='Path to save test results text file')
    parser.add_argument('--output_roc', type=str, default='test_tab_roc_pred.png', 
                        help='Path to save ROC curve image')
    parser.add_argument('--model_path', type=str, default='checkpoints/tab_best_model.pth', 
                        help='Path to trained model weights')
    args = parser.parse_args()


    main(output_test_txt_path=args.output_txt, 
         output_test_roc_path=args.output_roc, 
         model_path=args.model_path)
