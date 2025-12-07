# encoding: utf-8
import os
import torch
import pandas as pd
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def uniform_resize_ct(image, target_size=(128,128,128), mode='trilinear'):
    """
    医学影像专用重采样函数
    参数：
    image: SimpleITK图像对象
    target_size: 目标尺寸(z,y,x)
    mode: 插值方式(推荐trilinear)
    """
    # 转换为numpy数组并调整维度顺序
    array = sitk.GetArrayFromImage(image)  # (z,y,x)
    tensor = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)  # (1,1,z,y,x)

    # 计算原始物理间距
    original_spacing = image.GetSpacing()  # (x,y,z)
    original_size = image.GetSize()  # (x,y,z)

    # 智能重采样策略
    if _need_isotropic_resample(original_spacing):
        # 各向异性数据需要特殊处理
        tensor = _anisotropic_resample(tensor, original_spacing, target_size)
    else:
        # 常规三线性插值
        tensor = F.interpolate(
            tensor, 
            size=target_size,
            mode=mode,
            align_corners=False
        )

    # 转换为numpy并重建SimpleITK图像
    resized_array = tensor.squeeze().cpu().numpy()
    result_image = sitk.GetImageFromArray(resized_array)
    
    # 设置新间距（关键！保持物理尺寸一致）
    new_spacing = [
        original_spacing[0] * original_size[0] / target_size[2],
        original_spacing[1] * original_size[1] / target_size[1],
        original_spacing[2] * original_size[2] / target_size[0]
    ]
    result_image.SetSpacing(new_spacing)
    
    return result_image

def _need_isotropic_resample(spacing, threshold=3.0):
    """判断是否需要各向同性处理（层厚差异过大时）"""
    return max(spacing)/min(spacing) > threshold

def _anisotropic_resample(tensor, original_spacing, target_size):
    """处理各向异性数据（如层厚5mm vs 像素0.5mm）"""
    # 1. 先在最高分辨率平面进行插值
    scale_factors = [
        1.0,  # batch
        1.0,  # channel
        target_size[0] / tensor.shape[2],
        (original_spacing[1]/original_spacing[0]) * (target_size[1]/tensor.shape[3]),
        (original_spacing[0]/original_spacing[0]) * (target_size[2]/tensor.shape[4])
    ]
    tensor = F.interpolate(
        tensor,
        scale_factor=scale_factors[2:],
        mode='trilinear',
        align_corners=False
    )
    
    # 2. 中心裁剪到目标尺寸
    _,_,d,h,w = tensor.shape
    start_z = (d - target_size[0])//2
    start_y = (h - target_size[1])//2
    start_x = (w - target_size[2])//2
    return tensor[
        :, :, 
        start_z:start_z+target_size[0],
        start_y:start_y+target_size[1],
        start_x:start_x+target_size[2]
    ]

class CTDataset(Dataset):
    """
    CT影像训练数据集
    特征：
    - 自动处理3D/2D数据
    - 支持内存缓存
    - 自动添加通道维度
    - 数据类型自动转换
    """

    def __init__(self, data_list, labels, target_shape=(128, 128, 128), transform=None, cache=False):
        """
        参数：
        data_list: list of str, nrrd文件路径列表
        labels: list of int, 对应标签(0/1)
        transform: 数据增强函数
        cache: 是否缓存数据到内存
        """
        self.data_list = data_list
        self.labels = labels
        self.target_shape = target_shape  # (D, H, W)
        self.transform = transform
        self.cache = cache
        self.cache_dict = {}  # 内存缓存字典

        # 预验证所有文件存在
        for p in data_list:
            if not os.path.exists(p):
                raise FileNotFoundError(f"NRRD文件不存在: {p}")

    def __len__(self):
        return len(self.data_list)

    def _load_nrrd(self, path):
        """加载NRRD文件并标准化为float32"""
        if self.cache and path in self.cache_dict:
            return self.cache_dict[path]

        try:
            image = sitk.ReadImage(path)

            # 执行智能重采样
            resized_image = uniform_resize_ct(image, target_size=self.target_shape)

            tensor = torch.from_numpy(sitk.GetArrayFromImage(resized_image)).float()

            # 数值标准化（保留原始CT值范围，可根据实际需求调整）
            tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)

            # 添加通道维度：[C, D, H, W]
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)  # 3D数据
            elif tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # 2D切片

            if self.cache:
                self.cache_dict[path] = tensor

            return tensor
        except Exception as e:
            raise RuntimeError(f"加载NRRD文件失败: {path}, 错误: {str(e)}")

    def __getitem__(self, idx):
        # 加载数据
        data = self._load_nrrd(self.data_list[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # 数据增强
        if self.transform:
            data = self.transform(data)

        return data, label

    @staticmethod
    def collate_fn(batch):
        """自定义批次处理函数"""
        data = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return data, labels


# 自动区分dataloader
def get_train_dataloader(
    train_path, batch_size=8, num_workers=4, random_seed=42, cache=True
):
    """
    生成训练和验证数据加载器
    参数:
    csv_path: 包含id和cls列的CSV文件路径
    batch_size: 批次大小
    num_workers: 数据加载线程数
    random_seed: 随机种子
    cache: 是否缓存数据
    """
    df = pd.read_excel(train_path, sheet_name="训练集&验证集")

    # 转换为整数标签（如果原始是字符串类别）
    if df["是否原发耐药"].dtype == object:
        cls_mapping = {cls: idx for idx, cls in enumerate(df["是否原发耐药"].unique())}
        df["cls_label"] = df["是否原发耐药"].map(cls_mapping)
    else:
        df["cls_label"] = df["是否原发耐药"].astype(int)

    # 生成文件路径
    df["nrrd_path"] = df.apply(
        lambda row: f"data/dataset/{str(int(row['是否原发耐药']))}/{str(int(row['序号']))}.nrrd",
        axis=1,
    )

    # 分层拆分数据集
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["cls_label"],  # 保持类别分布
        random_state=random_seed,
    )

    # 创建数据集
    train_dataset = CTDataset(
        data_list=train_df["nrrd_path"].tolist(),
        labels=train_df["cls_label"].tolist(),
        cache=cache,
    )

    val_dataset = CTDataset(
        data_list=val_df["nrrd_path"].tolist(),
        labels=val_df["cls_label"].tolist(),
        cache=False,  # 验证集通常不需要缓存
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=CTDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # 验证集可以使用更大批次
        shuffle=False,
        num_workers=num_workers,
        collate_fn=CTDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader

def get_test_dataloader(
    test_path, batch_size=8, num_workers=4, cache=True, target_shape=(64,64,64)
):
    df = pd.read_excel(test_path, sheet_name="测试集")

    # 转换为整数标签（如果原始是字符串类别）
    if df["是否原发耐药"].dtype == object:
        cls_mapping = {cls: idx for idx, cls in enumerate(df["是否原发耐药"].unique())}
        df["cls_label"] = df["是否原发耐药"].map(cls_mapping)
    else:
        df["cls_label"] = df["是否原发耐药"].astype(int)

    # 生成文件路径
    df["nrrd_path"] = df.apply(
        lambda row: f"data/dataset/{str(int(row['是否原发耐药']))}/{str(int(row['序号']))}.nrrd",
        axis=1,
    )

    test_dataset = CTDataset(
        data_list=df["nrrd_path"].tolist(),
        labels=df["cls_label"].tolist(),
        target_shape=target_shape,
        cache=False,  # 验证集通常不需要缓存
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # 验证集可以使用更大批次
        shuffle=False,
        num_workers=num_workers,
        collate_fn=CTDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return test_dataloader


if __name__ == "__main__":
    train_loader, val_loader = get_train_dataloader("data/train_table.xlsx")

    # 验证数据加载
    for batch in train_loader:
        images, labels = batch
        print(f"训练批次维度: {images.shape}, 标签: {labels}")
        # break
