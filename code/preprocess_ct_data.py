import os
import pydicom
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm

def get_id2cls(train_table_path):
    id2cls = {}
    train_table_train_df = pd.read_excel(train_table_path, sheet_name="训练集&验证集")
    train_table_test_df = pd.read_excel(train_table_path, sheet_name="测试集")
    for index, row in train_table_train_df.iterrows():
        id = int(row['序号'])
        isOK = int(row['是否原发耐药'])
        id2cls[id] = isOK
    for index, row in train_table_test_df.iterrows():
        id = int(row['序号'])
        isOK = int(row['是否原发耐药'])
        id2cls[id] = isOK
    return id2cls

def copy(dcm_dir, save_path):
    """
    读取dcm_dir下的DICOM序列，自动选择最薄层(最小SliceThickness)序列保存为NRRD
    
    参数:
    dcm_dir (str): 包含DICOM文件的目录路径
    save_path (str): 输出的NRRD文件路径
    
    异常:
    ValueError: 如果没有DICOM序列或无法读取厚度参数
    RuntimeError: 文件读取失败
    """
    # 获取所有DICOM系列ID
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcm_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {dcm_dir}")

    # 遍历所有系列获取厚度信息
    series_info = []
    for sid in series_ids:
        try:
            # 获取该系列文件列表
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dcm_dir, sid)
            
            # 读取第一个文件的DICOM头信息
            ds = pydicom.dcmread(files[0], stop_before_pixels=True)
            
            # 提取关键参数
            thickness = float(getattr(ds, 'SliceThickness', 0))  # 兼容不同tag
            instance_num = int(getattr(ds, 'InstanceNumber', 0))
            
            series_info.append({
                'id': sid,
                'thickness': thickness,
                'instance_num': instance_num,
                'files': files
            })
        except Exception as e:
            raise RuntimeError(f"Failed to read series {sid}: {str(e)}")

    if not series_info:
        raise ValueError("No valid DICOM series with thickness information")

    # 选择最薄层序列（厚度相同则选InstanceNumber较小的）
    target_series = min(series_info, 
                       key=lambda x: (x['thickness'], x['instance_num']))

    # 读取选定序列
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(target_series['files'])
    image = reader.Execute()

    # 保存为NRRD
    sitk.WriteImage(image, save_path)
    print(f"Saved {save_path} with slice thickness: {target_series['thickness']}mm")



def main(id2cls):
    normal_thin = "data/影像/平扫"
    enhance_thin_path = "data/影像/增强"
    save_dir = "data/dataset"
    CT_info_df = pd.read_excel(CT_info_path)
    p_bar = tqdm(total=len(CT_info_df), desc="正在复制与处理中")
    for index, row in CT_info_df.iterrows():
        id = int(row['序号'])
        cls = id2cls[id]
        is_thin = int(row['平扫薄层'])
        enhance_thin = int(row['增强薄层'])
        if is_thin == 1:
            dcm_dir = os.path.join(normal_thin, str(id))
        if enhance_thin == 1:
            dcm_dir = os.path.join(enhance_thin_path, str(id))
        save_path = os.path.join(save_dir, str(cls), f"{id}.nrrd")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        copy(dcm_dir, save_path)
        p_bar.update(1)


if __name__ == '__main__':
    # 获取ID和分类之间的关系
    train_table_path = "data/train_table.xlsx"
    CT_info_path = "data/CT扫描明细.xlsx"
    id2cls = get_id2cls(train_table_path)
    
    # 将指定的CT复制
    main(id2cls)



