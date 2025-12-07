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
print(f"Using device: {device}")

# 配置参数
config = {
    "roi_size": (128,128,128),
    "data_path": "data/train_table.xlsx",
    "batch_size": 4,
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
    "save_dir": "results_fusion_11_18",
    "seed_base": 42,
    "output_test_txt_path": "results_fusion_11_18/test_result_fusion.txt",
    "output_test_roc_path": "results_fusion_11_18/test_roc_fusion.png",
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


def main():
    # 读取 CT 数据并整理表格数据
    print(" -------------- 数据读取与预处理 -------------- ")
    df = pd.read_excel("data/train_table.xlsx", sheet_name="训练集&验证集")
    df["nrrd_path"] = df.apply(lambda row: f"data/dataset/{int(row['是否原发耐药'])}/{int(row['序号'])}.nrrd", axis=1)
    full_dataset = FusionDataset(
        data_list=df["nrrd_path"].tolist(),
        labels=df["是否原发耐药"].astype(int).tolist(),
        target_shape=config["roi_size"]
    )
    # 创建checkpoints目录
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(f"{config['save_dir']}/checkpoints", exist_ok=True)

    # 初始化最佳模型
    best_model = None
    best_auc = 0

    # 五折交叉验证
    skf = StratifiedKFold(n_splits=config["k_folds"], shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["是否原发耐药"])):
        print(f"\n=== Fold {fold+1}/{config['k_folds']} ===")
        save_fold_dir = os.path.join(f"{config['save_dir']}/checkpoints/fold{fold+1}")
        os.makedirs(save_fold_dir, exist_ok=True)

        # 设置不同的随机种子
        torch.manual_seed(config["seed_base"] + fold)
        np.random.seed(config["seed_base"] + fold)

        # 数据加载器
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            drop_last=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=config["batch_size"]*2,
            shuffle=False,
            num_workers=4,
        )

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

        # 模型初始化
        model = FusionModel(
            swin_params=swin_model_config,
            tab_params=tab_model_config,
            num_categories=len(tab_model_config['categories']),  # 分类特征数量
            num_classes=1
        ).to(device)

        # 优化器和损失函数
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        # 添加学习率调度器（每10个epoch衰减为原来的0.1倍）
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=10,  # 每10个epoch衰减一次
            gamma=0.1      # 衰减率为0.1
        )
        # 日志文件
        log_file = f"{config['save_dir']}/training_log_fold{fold+1}.csv"
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'train_auc', 'train_recall',
                           'val_loss', 'val_acc', 'val_recall', 'val_auc'])
        # 训练循环
        for epoch in range(config["max_epochs"]):
            # 训练阶段
            model.train()
            epoch_train_loss = 0.0
            all_outputs, all_labels = [], []

            for batch_idx, (inputs, cat, con, labels) in enumerate(train_loader):
                inputs = inputs.to(device).float()
                cat = cat.to(device).long()
                con = con.to(device).float()
                labels = labels.float().to(device).float()

                # 前向传播
                outputs = model(inputs, cat, con).squeeze(1)
                loss = criterion(outputs, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 收集指标
                epoch_train_loss += loss.item()
                all_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # 计算训练指标
            train_loss = epoch_train_loss / len(train_loader)
            train_acc = accuracy_score(all_labels, np.array(all_outputs) > 0.5)
            train_recall = recall_score(all_labels, np.array(all_outputs) > 0.5)
            train_auc = roc_auc_score(all_labels, all_outputs)

            scheduler.step()  # 更新学习率

            # 验证阶段
            val_loss, val_acc, val_recall, val_auc = evaluate(model, val_loader, criterion, device)

            # 保存最佳模型
            if val_auc > best_auc:
                best_auc = val_auc
                best_model = model
                torch.save(model.state_dict(), f"{config['save_dir']}/checkpoints/best_model.pth")

            # 记录日志
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch+1, train_loss, train_acc, train_auc, train_recall,
                    val_loss, val_acc, val_recall, val_auc
                ])

            # 保存模型
            if (epoch + 1) % 2 == 0:
                ckpt_path = os.path.join(
                    save_fold_dir,
                    f"fold{fold+1}_epoch{epoch+1}.pth"
                )
                torch.save(model.state_dict(), ckpt_path)

            print(f"Epoch {epoch+1}/{config['max_epochs']}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}\n")
    return best_model

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for inputs, cat, con, labels in dataloader:
            inputs = inputs.to(device).float()
            cat = cat.to(device).long()
            con = con.to(device).float()
            labels = labels.float().to(device).float()

            outputs = model(inputs, cat, con).squeeze(1)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, np.array(all_outputs) > 0.5)
    recall = recall_score(all_labels, np.array(all_outputs) > 0.5)
    auc = roc_auc_score(all_labels, all_outputs)
    return avg_loss, accuracy, recall, auc

def accuracy_score(labels, preds):
    return np.mean(np.array(labels) == preds)

def evaluate_test(model, loader):
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
    auc = roc_auc_score(y_true, y_score)
    return auc

if __name__ == "__main__":
    main()
