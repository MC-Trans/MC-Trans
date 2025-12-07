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
    "max_epochs": 50,
    "num_classes": 1,
    "k_folds": 5,
    "save_dir": "results_single_CT_11_15",
    "seed_base": 43,
}

def accuracy_score(labels, preds):
    return np.mean(np.array(labels) == preds)

def evaluate_test(model, loader, output_result_txt, output_test_roc_path):
    model.eval()
    all_outputs, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze(1)
            
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_score = np.array(all_outputs)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]
    y_pred = (y_score >= best_threshold).astype(int)
    print(f"Best threshold from ROC: {best_threshold:.4f}")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
    }

    with open(output_result_txt, "w") as f:
        f.write(f"best_threshold: {best_threshold:.6f}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.2f}')
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(output_test_roc_path, dpi=300)
    plt.close()

    report = classification_report(y_true, y_pred)
    with open(output_result_txt, "a") as f:
        f.write("\n")
        f.write(report)

def main(output_result_txt="test_single_CT_result_pred.txt", output_test_roc_path="test_single_CT_roc_pred.png", model_path="checkpoints/single_CT_best_model.pth"):
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_dataloader = get_test_dataloader("data/train_table.xlsx", batch_size=config['batch_size'], target_shape=(128,128,128))
    evaluate_test(model, test_dataloader, output_result_txt, output_test_roc_path)

if __name__ == '__main__':
    # 添加命令行参数支持
    import argparse
    parser = argparse.ArgumentParser(description='CT Model Prediction')
    parser.add_argument('--output_txt', type=str, default='test_single_CT_result_pred.txt', 
                        help='Path to save test results text file')
    parser.add_argument('--output_roc', type=str, default='test_single_CT_roc_pred.png', 
                        help='Path to save ROC curve image')
    parser.add_argument('--model_path', type=str, default='checkpoints/single_CT_best_model.pth', 
                        help='Path to trained model weights')
    args = parser.parse_args()
    
    main(output_result_txt=args.output_txt, 
         output_test_roc_path=args.output_roc, 
         model_path=args.model_path)