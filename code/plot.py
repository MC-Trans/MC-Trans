import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_for_tab(data_dir, column_name, save_dir):
    for name in os.listdir(data_dir):
        if name.endswith('.csv'):
            csv_file = os.path.join(data_dir, name)
            df = pd.read_csv(csv_file)

            # 目标列
            target_columns = [f'train_{column_name}', f'val_{column_name}']

            # 提取目标列数据
            epochs = df['epoch']  # x 轴通常是 epoch 列
            train_column = df[target_columns[0]]
            val_column = df[target_columns[1]]

            # 创建图像
            plt.figure(figsize=(10, 6))

            # 绘制 train_loss 曲线
            plt.plot(epochs, train_column, label=target_columns[0], color='blue', linestyle='-', marker='o')

            # 绘制 val_loss 曲线
            plt.plot(epochs, val_column, label=target_columns[1], color='red', linestyle='--', marker='x')

            # 添加标题和标签
            plt.title(f'{target_columns[0]} and {target_columns[0]} over Epochs', fontsize=16)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(column_name, fontsize=12)

            # 添加图例
            plt.legend(fontsize=12)

            # 显示网格
            plt.grid(True)

            # 显示图像
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{name.replace('.csv', '')}_{column_name}.png", dpi=300)  # 保存为 PNG 格式
            plt.close()

if __name__ == '__main__':
    # # 画出 tab 的所有图像
    # data_dir = "tab_results_11_15/csvs"
    # save_dir = "tab_results_11_15/pngs"
    # os.makedirs(save_dir, exist_ok=True)
    # plot_for_tab(data_dir, 'loss', save_dir)
    # plot_for_tab(data_dir, 'acc', save_dir)
    # plot_for_tab(data_dir, 'recall', save_dir)
    # plot_for_tab(data_dir, 'auc', save_dir)

    # 画出 fusion 的所有图像
    data_dir = "results_fusion_11_15"
    save_dir = "results_fusion_11_15/pngs"
    os.makedirs(save_dir, exist_ok=True)
    plot_for_tab(data_dir, 'loss', save_dir)
    plot_for_tab(data_dir, 'acc', save_dir)
    plot_for_tab(data_dir, 'recall', save_dir)
    plot_for_tab(data_dir, 'auc', save_dir)

    # 画出 CT 的所有图像
    # data_dir = "results_single_CT_11_15"
    # save_dir = "results_single_CT_11_15/pngs"
    # os.makedirs(save_dir, exist_ok=True)
    # plot_for_tab(data_dir, 'loss', save_dir)
    # plot_for_tab(data_dir, 'acc', save_dir)
    # plot_for_tab(data_dir, 'recall', save_dir)
    # plot_for_tab(data_dir, 'auc', save_dir)