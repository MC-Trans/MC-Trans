This project implements a multimodal fusion model that combines 3D CT scan images with structured tabular data (such as patient clinical information, biochemical indicators, etc.) for binary classification tasks in the medical field. The model employs a Swin Transformer to process CT image features and a TabTransformer to handle tabular data features. Finally, the features from both modalities are fused through a fully connected layer for classification prediction.Due to file size limits, the model checkpoints and demo folder are stored in the "Releases" section.

Installation Guide

Python 3.9

CUDA 11.8 (Recommended for optimal performance)

Environment Setup Steps

# Create and activate conda environment
conda create -n model python=3.9
conda activate model

# Install dependencies
pip install -r requirements.txt
# If the network connection is poor, use the following command (for users in China)
pip install -r requirements.txt -i [https://pypi.mirrors.ustc.edu.cn/simple/](https://pypi.mirrors.ustc.edu.cn/simple/)

# (Optional) When installing torch, it is recommended to use the following command
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)


# Data Preparation

Data Privacy and Availability

This project releases the full source code and pre-trained model weights. However,the patient-level data generated in this retrospective study are protected patient health information and are not publicly available due to privacy and ethical restrictions imposed by the Ethics Committees of The Affiliated Yixing Hospital of Jiangsu University and Lianyungang Clinical College of Nanjing Medical University. De-identified data may be made available to qualified researchers for the purpose of replicating the study findings, upon reasonable request to the corresponding author and with approval from the aforementioned Ethics Committees. Requests will be evaluated based on scientific merit and data use agreement compliance.

To facilitate testing and code verification, we have provided 1 public CT sample scans from the [kaggle-dlwpt-volumetric-dicom-lung] and tabular data for two completely fictitious patients in the [demo_folder_path] directory. These samples are used only for testing the code pipeline and do not represent the distribution of our private dataset.

# Data Explanation

The data needs to be organized into the format mentioned above to conduct experiments. The following script is used to generate the organized data (if you are using the data already organized in this repo, you do not need to run this):

# Run CT data preprocessing script
python code/preprocess_ct_data.py


Usage Instructions

Quick Verification (Recommended)

# Run pred_table.py example
python code/pred_table.py --output_txt ./results/table_results.txt --output_roc ./results/table_roc.png --model_path ./checkpoints/tab_best_model.pth

# Run pred_CT.py example
python code/pred_CT.py --output_txt ./results/ct_results.txt --output_roc ./results/ct_roc.png --model_path ./checkpoints/single_CT_best_model.pth

# Run pred_fusion.py example
python code/pred_fusion.py --output_txt ./results/fusion_results.txt --output_roc ./results/fusion_roc.png --model_path ./checkpoints/fusion_best_model.pth


Training Custom Models

Note that parameters can be adjusted at the beginning of each training and prediction script, for example:

# Configuration parameters
config = {
    "roi_size": (128,128,128),
    "data_path": "data/train_table.xlsx",
    "batch_size": 4,
    "lr": 0.01,
    "max_epochs": 50,
    "k_folds": 5,
    "save_split": 1,  # Save weights every 'save_split' epochs
    "categorical_cols": [...],
    "continuous_cols": [...],
    "save_dir": "results_fusion_11_18",
    "seed_base": 42,
    "output_test_txt_path": "results_fusion_11_18/test_result_fusion.txt",
    "output_test_roc_path": "results_fusion_11_18/test_roc_fusion.png",
}


1. Train Multimodal Fusion Model

# Train multimodal fusion model (5-fold cross-validation)
python code/train_fusion.py


2. Train Single-Modality Models

# Train CT single-modality model
python code/train_CT.py

# Train tabular single-modality model
python code/train_table.py


Prediction

# Multimodal fusion model prediction
python code/pred_fusion.py

# CT single-modality prediction
python code/pred_CT.py

# Tabular single-modality prediction
python code/pred_table.py


Evaluation Metrics

The model is evaluated using the following metrics:

Accuracy

Precision

Recall

F1 Score

ROC Curve and AUC Value

Results

Training results are saved in the following directories:

results_fusion_11_18/ - Multimodal fusion model results

Contains training logs, best model weights, test results, and ROC curves.

results_CT_11_18/ - CT single-modality model results

Contains training logs, best model weights, test results, and ROC curves.

results_table_11_18/ - Tabular single-modality model results

Contains training logs, best model weights, test results, and ROC curves.

Notes

Ensure the CUDA environment is correctly configured to fully utilize GPU acceleration.

Data preprocessing has a significant impact on model performance; please ensure the data format meets the requirements.

The training process automatically performs 5-fold cross-validation; parameters can be adjusted in the configuration.

To change model parameters, please modify the config dictionary in the corresponding script.
