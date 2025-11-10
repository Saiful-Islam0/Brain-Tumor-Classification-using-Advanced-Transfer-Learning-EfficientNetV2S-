# 🧠 Brain Tumor Classification using Advanced Transfer Learning (EfficientNetV2S)

## Overview
This repository contains a comprehensive Jupyter notebook detailing the analysis, preprocessing, and training pipeline for a multi-class brain tumor classification model using MRI images.  
The primary goal is to accurately classify MRI scans into four distinct categories: Glioma, Meningioma, Pituitary tumor, and No Tumor (healthy). The solution leverages Transfer Learning with a pre-trained EfficientNetV2S model, enhanced by modern deep learning optimizations like mixed precision training and multi-GPU support.

## Key Features
- **Model Architecture:** Utilizes the state-of-the-art EfficientNetV2S CNN architecture, pre-trained on ImageNet, for superior feature extraction.
- **Transfer Learning Strategy:** Implements fine-tuning by freezing the initial layers of the base model and training a custom classification head.
- **Performance Optimization:** Includes Mixed Precision training (for NVIDIA T4/GPU environments) and TensorFlow's MirroredStrategy for distributed training across multiple GPUs, significantly accelerating the training loop.
- **Robust Training:** Employs effective callbacks:
  - Early Stopping (with patience) to prevent overfitting.
  - Model Checkpointing to save only the best-performing weights.
  - ReduceLROnPlateau for adaptive learning rate scheduling.
- **Comprehensive Evaluation:** Provides detailed performance metrics, including a Classification Report, Confusion Matrix, ROC curves, and Precision-Recall curves.

## Data Summary
The model was trained on a dataset containing 4 distinct classes of brain MRI images.  

| Class         | Count | Description |
|---------------|-------|-----------------------------------------------------------|
| Glioma        | 3,754 | The most common type of primary brain tumor.              |
| Pituitary     | 2,706 | Tumors originating in the pituitary gland.                |
| Meningioma    | 2,343 | Tumors arising from the meninges (membranes surrounding the brain/spinal cord). |
| No Tumor      | 1,757 | Healthy brain scans.                                      |
| **Total**     | 10,560|                                                         |

The data was split 70% for Training, 15% for Validation, and 15% for Testing, with stratification maintained across all splits to ensure balanced class representation.

## Results Highlights
The model achieved high performance metrics on the independent test set, demonstrating strong generalization capabilities.  
- **Test Accuracy:** ~99.99%
- **Average F1-Score (Weighted):** ~0.99

### Classification Report (Test Set)
| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Glioma    | 1.00      | 0.99   | 0.99     |
| Meningioma| 0.98      | 0.99   | 0.99     |
| No Tumor  | 1.00      | 0.99   | 1.00     |
| Pituitary | 1.00      | 1.00   | 1.00     |

## Visual Analysis
The final notebook includes comprehensive plots for in-depth analysis:  
- Confusion Matrix: Shows the few misclassifications (e.g., Meningioma samples mistaken for other classes), indicating high prediction confidence overall.
- ROC Curves (One-vs-Rest): Illustrates the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR) for each class, with high Area Under the Curve (AUC) scores (all close to 1.00).
- Precision-Recall Curves (AP): Confirms model stability and accuracy across different classification thresholds.
- Per-Class Performance Bar Chart: Visually compares precision, recall, and f1-score for easy metric comparison.

## Technologies and Libraries
- **Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras (using EfficientNetV2S)
- **Data Handling:** Pandas, NumPy
- **Visualization & Metrics:** Matplotlib, Seaborn, Scikit-learn (`classification_report`, `confusion_matrix`)

## How to Run the Notebook

1. **Clone the Repository:**
git clone https://github.com/Saiful-Islam0/-Brain-Tumor-Classification-using-Advanced-Transfer-Learning-EfficientNetV2S-.git
cd -Brain-Tumor-Classification-using-Advanced-Transfer-Learning-EfficientNetV2S-
2. **Dependencies:**  
Ensure you have Python 3 and the following core libraries installed. The notebook uses an explicit protobuf downgrade for stable TensorFlow operation in some environments:pip install tensorflow pandas numpy matplotlib seaborn scikit-learn kagglehub "protobuf~=3.20.0"
3. **Run the Notebook:**  
Execute the cells in `brain-tumor-train-model.ipynb` sequentially. The notebook handles the dataset download automatically using kagglehub.

---

**Repository Link:**  
[Brain Tumor Classification using Advanced Transfer Learning (EfficientNetV2S)](https://github.com/Saiful-Islam0/-Brain-Tumor-Classification-using-Advanced-Transfer-Learning-EfficientNetV2S-.git)