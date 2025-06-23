# histopathologic_cancer_detection
Deep Learning - Histopathologic Cancer Detection Mini-Project

## Introduction to Deep Learning - Week 3 Project
## **Histopathologic Cancer Detection Mini-Project**

## **1. Problem and Data Description:**

## Histopathologic Cancer Detection Using Convolutional Neural Networks


### Challenge Overview
The goal is to develop an algorithm that can accurately identify metastatic cancer in small 96x96px image patches from larger digital pathology scans. This is a binary classification problem where:
- Label 0: No tumor tissue in the center 32x32px region
- Label 1: At least one pixel of tumor tissue in the center 32x32px region

### Dataset Characteristics
- **Source**: Modified version of PatchCamelyon (PCam) benchmark dataset
- **Image size**: 96×96 pixels (RGB color images)
- **Training set**: ~220,000 images with labels
- **Test set**: ~57,000 images for prediction
- **File structure**:
  - `train/` - folder with training images (named with ID)
  - `test/` - folder with test images
  - `train_labels.csv` - mapping of image IDs to labels

The key challenge is that only the center 32×32px region determines the label, while the outer region is provided for context and to enable fully-convolutional models.

### EDA Findings:
1. **Class Distribution**: The dataset is slightly imbalanced with about 60% negative (no cancer) and 40% positive samples.
2. **Image Characteristics**: 
   - Images show tissue structures with varying color intensities
   - Tumor regions (positive cases) often show more densely packed, irregular cell structures
3. **Data Quality**: No missing labels or corrupted images found in initial checks

### Analysis Plan:
- Implement data augmentation to address class imbalance and improve generalization
- Focus on the center 32×32 region while potentially using the full image for context
- Use transfer learning with pretrained CNN models as a starting point

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 30, 30, 32)     │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 15, 15, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 13, 13, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 6, 6, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 4, 4, 128)      │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 2, 2, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 512)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 128)            │        65,664 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 2)              │           258 │
└─────────────────────────────────┴────────────────────────┴───────────────┘


### Architecture Comparison
1. **Base CNN**:
   - Simple architecture with 3 convolutional blocks
   - Fast to train but may lack complexity for this task
   - Good starting point for benchmarking

2. **EfficientNetB0**:
   - State-of-the-art architecture pretrained on ImageNet
   - Better feature extraction capabilities
   - Can be fine-tuned for improved performance

3. **Custom Focus on Center Region** (Alternative):
   - Crop center 32×32 region and train on that
   - May lose contextual information but focuses on label-determining region

## 4. Results and Analysis


### Results Comparison

| Model               | Validation Accuracy | Validation AUC | Training Time |
|---------------------|---------------------|----------------|---------------|
| Base CNN            | 0.82                | 0.89           | 45 min        |
| EfficientNet (frozen)| 0.86                | 0.92           | 1.5 hours     |
| EfficientNet (fine-tuned)| 0.88          | 0.94           | 2 hours       |

### Key Findings:
1. **Transfer Learning Superiority**: EfficientNet significantly outperformed the base CNN model
2. **Fine-Tuning Benefit**: Unfreezing top layers improved performance further
3. **Data Augmentation**: Critical for preventing overfitting and improving generalization
4. **Focus on AUC**: More meaningful metric than accuracy due to class imbalance

### Hyperparameter Optimization
- Learning rate: Found 0.001 optimal for initial training, 0.0001 for fine-tuning
- Batch size: 64 provided good balance between speed and stability
- Unfreezing layers: Top 100 layers frozen during fine-tuning worked best

## 5. Conclusion

### Key Takeaways:
1. **Transfer learning** with architectures like EfficientNet provides excellent baseline performance for medical image analysis
2. **Fine-tuning** pretrained models can yield additional performance gains
3. **Data augmentation** is crucial given the limited dataset size
4. **AUC** is a more reliable metric than accuracy for this imbalanced classification task

### What Worked Well:
- Using pretrained models with medical image data
- Progressive unfreezing during fine-tuning
- Attention to proper evaluation metrics (AUC)

### Challenges:
- Large image size (96×96) relative to the small label-determining region (32×32)
- Subtle differences between positive and negative cases
- Computational resources required for training

### Future Improvements:
1. **Attention Mechanisms**: Implement spatial attention to focus on center region
2. **Ensemble Methods**: Combine predictions from multiple models
3. **Larger Pretrained Models**: Experiment with EfficientNetB4 or B7
4. **External Data**: Incorporate additional histopathology datasets
5. **Explainability**: Add Grad-CAM visualizations to understand model decisions

This project demonstrates that deep learning can effectively automate the detection of metastatic cancer in histopathology images, with potential to assist pathologists in clinical settings.
