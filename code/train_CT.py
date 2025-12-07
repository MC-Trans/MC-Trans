# encoding: utf-8
import os
import sys
import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from torch.utils.data import Subset, DataLoader

sys.path.append(os.getcwd())
from swinTModel.dataloader.ct_dataloader import CTDataset, get_train_dataloader, get_test_dataloader
from swinTModel.layers.swin3d_layer import SwinTransformerForClassification

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
print(f"Using device: {device}")

# 基础配置
config = {
    "roi_size": (128, 128, 128),
    "batch_size": 4,
    "lr": 0.01,
    "max_epochs": 50,
    "num_classes": 1,
    "k_folds": 5,
    "save_dir": "results_single_CT_11_15",
    "seed_base": 43,
    "output_test_txt_path": "results_single_CT_11_15/test_result.txt",
    "output_test_roc_path": "results_single_CT_11_15/test_roc.png",
}

def main():
    # 初始化完整数据集
    df = pd.read_excel("data/train_table.xlsx", sheet_name="训练集&验证集")
    df["nrrd_path"] = df.apply(lambda row: f"data/dataset/{int(row['是否原发耐药'])}/{int(row['序号'])}.nrrd", axis=1)
    
    full_dataset = CTDataset(
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
            collate_fn=CTDataset.collate_fn
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=config["batch_size"]*2,
            shuffle=False,
            num_workers=4,
            collate_fn=CTDataset.collate_fn
        )

        # 模型初始化
        model = SwinTransformerForClassification(
            img_size=config["roi_size"],
            num_classes=config["num_classes"],
            in_channels=1,
            out_channels=768,
            feature_size=48,
            drop_rate=0.5,
            attn_drop_rate=0.5,
            dropout_path_rate=0.5,
            use_checkpoint=True,
        ).to(device)

        # 优化器和损失函数
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

        # 添加学习率调度器（每20个epoch衰减为原来的0.1倍）
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=20,  # 每20个epoch衰减一次
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

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                # 前向传播
                outputs = model(inputs).squeeze(1)
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
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            outputs = model(inputs).squeeze(1)
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
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze(1)
            
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 转换为numpy数组
    y_true = np.array(all_labels)
    y_pred = np.array(all_outputs) > 0.5
    y_score = np.array(all_outputs)

    # 计算指标
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
    }

    # 保存指标
    with open(config['output_test_txt_path'], "w") as f:
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
    plt.savefig(config['output_test_roc_path'], dpi=300)
    plt.close()

    # 保存分类报告
    report = classification_report(y_true, y_pred)
    with open(config['output_test_txt_path'], "a") as f:
        f.write("\n")
        f.write(report)


if __name__ == '__main__':
    main()
    # best_model = main()
    # test_dataloader = get_test_dataloader("data/train_table.xlsx", batch_size=config['batch_size'],target_shape=(128,128,128))
    # evaluate_test(best_model, test_dataloader)