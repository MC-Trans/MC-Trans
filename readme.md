This project implements a multimodal fusion model that combines 3D CT scan images with structured tabular data (such as patient clinical information, biochemical indicators, etc.) for binary classification tasks in the medical field. The model employs a Swin Transformer to process CT image features and a TabTransformer to handle tabular data features. Finally, the features from both modalities are fused through a fully connected layer for classification prediction.Due to file size limits, the model checkpoints are stored in the "Releases" section.



Installation Guide



Python 3.9



CUDA 11.8 (Recommended for optimal performance)



Environment Setup Steps



\# Create and activate conda environment

conda create -n model python=3.9

conda activate model



\# Install dependencies

pip install -r requirements.txt

\# If the network connection is poor, use the following command (for users in China)

pip install -r requirements.txt -i \[https://pypi.mirrors.ustc.edu.cn/simple/](https://pypi.mirrors.ustc.edu.cn/simple/)



\# (Optional) When installing torch, it is recommended to use the following command

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url \[https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)





Data Preparation



Data Privacy and Availability



This project releases the full source code and pre-trained model weights. However, the original dataset contains sensitive patient clinical records. Due to privacy protection regulations and ethical considerations (such as HIPAA/GDPR), the original data cannot be publicly shared.



To facilitate testing and code verification, we have provided 1 public CT sample scans from the \[kaggle-dlwpt-volumetric-dicom-lung] and tabular data for two completely fictitious patients in the \[demo\_folder\_path] directory. These samples are used only for testing the code pipeline and do not represent the distribution of our private dataset.



Data Explanation



The data needs to be organized into the format mentioned above to conduct experiments. The following script is used to generate the organized data (if you are using the data already organized in this repo, you do not need to run this):



\# Run CT data preprocessing script

python code/preprocess\_ct\_data.py





Usage Instructions



Quick Verification (Recommended)



\# Run pred\_table.py example

python code/pred\_table.py --output\_txt ./results/table\_results.txt --output\_roc ./results/table\_roc.png --model\_path ./checkpoints/tab\_best\_model.pth



\# Run pred\_CT.py example

python code/pred\_CT.py --output\_txt ./results/ct\_results.txt --output\_roc ./results/ct\_roc.png --model\_path ./checkpoints/single\_CT\_best\_model.pth



\# Run pred\_fusion.py example

python code/pred\_fusion.py --output\_txt ./results/fusion\_results.txt --output\_roc ./results/fusion\_roc.png --model\_path ./checkpoints/fusion\_best\_model.pth





Training Custom Models



Note that parameters can be adjusted at the beginning of each training and prediction script, for example:



\# Configuration parameters

config = {

    "roi\_size": (128,128,128),

    "data\_path": "data/train\_table.xlsx",

    "batch\_size": 4,

    "lr": 0.01,

    "max\_epochs": 50,

    "k\_folds": 5,

    "save\_split": 1,  # Save weights every 'save\_split' epochs

    "categorical\_cols": \[...],

    "continuous\_cols": \[...],

    "save\_dir": "results\_fusion\_11\_18",

    "seed\_base": 42,

    "output\_test\_txt\_path": "results\_fusion\_11\_18/test\_result\_fusion.txt",

    "output\_test\_roc\_path": "results\_fusion\_11\_18/test\_roc\_fusion.png",

}





1\. Train Multimodal Fusion Model



\# Train multimodal fusion model (5-fold cross-validation)

python code/train\_fusion.py





2\. Train Single-Modality Models



\# Train CT single-modality model

python code/train\_CT.py



\# Train tabular single-modality model

python code/train\_table.py





Prediction



\# Multimodal fusion model prediction

python code/pred\_fusion.py



\# CT single-modality prediction

python code/pred\_CT.py



\# Tabular single-modality prediction

python code/pred\_table.py





Evaluation Metrics



The model is evaluated using the following metrics:



Accuracy



Precision



Recall



F1 Score



ROC Curve and AUC Value



Results



Training results are saved in the following directories:



results\_fusion\_11\_18/ - Multimodal fusion model results



Contains training logs, best model weights, test results, and ROC curves.



results\_CT\_11\_18/ - CT single-modality model results



Contains training logs, best model weights, test results, and ROC curves.



results\_table\_11\_18/ - Tabular single-modality model results



Contains training logs, best model weights, test results, and ROC curves.



Notes



Ensure the CUDA environment is correctly configured to fully utilize GPU acceleration.



Data preprocessing has a significant impact on model performance; please ensure the data format meets the requirements.



The training process automatically performs 5-fold cross-validation; parameters can be adjusted in the configuration.



To change model parameters, please modify the config dictionary in the corresponding script.





