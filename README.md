# Food & Fruit Recognition with Calorie Estimation

A comprehensive **Computer Vision pipeline** for food and fruit recognition, few-shot learning, and segmentation using modern deep learning architectures.  
This project explores multiple tasks including binary classification, multi-class classification, metric learning, and semantic segmentation, aiming toward an end-to-end system for **diet analysis and calorie estimation**.



## Project Overview

Understanding food types and estimating calories is a key step toward healthier lifestyle guidance.  
This project builds and evaluates multiple deep learning models capable of:

- Distinguishing **Food vs Fruit**
- Recognizing food categories using **few-shot learning**
- Performing **multi-class classification**
- Segmenting fruit items from images
- Preparing the foundation for calorie estimation pipelines

## Dataset Overview

- **Food Training Images:** 2538  
- **Fruit Training Images:** 1759  
- **Food Validation Images:** 261  
- **Fruit Validation Images:** 150  

**Preprocessing Includes**
- Image resizing & normalization
- ImageNet mean/std normalization
- Extensive data augmentation
- Class balancing


## Tasks & Models

### **Part A – Binary Classification (Food vs Fruit)**
**Models Used**
- Custom CNN with Residual Blocks
- ResNet-18 (Frozen Backbone + Fine-Tuning)
- MobileNetV2
- Attention-enhanced MobileNetV2 (Multi-Fruits)

**Key Techniques**
- Transfer Learning (ImageNet)
- Class-Weighted Loss
- Data Augmentation
- ReduceLROnPlateau Scheduler



### **Part B – Few-Shot Food Recognition**
**Model**
- Siamese Network (Triplet Loss)
- Fine Tuned CLIP

**Encoders Tested For Siamese Network**
- Xception
- ConvNeXt
- EfficientNet

**Key Techniques**
- Metric Learning with Triplet Loss
- L2-Normalized Embeddings
- Euclidean Distance Comparison



### **Part C – Multi-Class Classification On Fruits**
**Models Used**
- ResNet-50
- EfficientNet-B4
- MobileNetV2
- ConvNeXt-Tiny
- Custom CNN

**Key Techniques**
- End-to-End Fine-Tuning
- Global Average Pooling
- Dropout Regularization
- GELU & Layer Normalization



### **Part D – Binary Segmentation On Fruits**
**Models Used**
- U-Net
- DeepLabV3
- SegNet

**Key Techniques**
- Encoder-Decoder Architectures
- Skip Connections
- Atrous Spatial Pyramid Pooling (ASPP)



### **Part E – Multi Segmentation On Fruits**
**Model**
- ResNet50V2-UNet

**Key Techniques**
- Pretrained Backbone
- U-Net-style Skip Connections
- Decoder Feature Refinement



## Evaluation & Visualization

- Accuracy, Precision, Recall
- Training vs Validation Curves
- Siamese embedding distance analysis
- Segmentation output visualizations


## Deployment

The Project is deployed using Streamlit for an interactive and user-friendly web interface..

🔗 [Launch the App](https://)


## Team Members

- **[Toka Khaled](https://github.com/TokaKhaled4)**
- **[Jana Essam](https://github.com/janaessam31)**
- **[Jana Hani](https://github.com/JH-33)**
- **[Manar Mostafa](https://github.com/2004Manar)**
- **[Rana Nasser](https://github.com/rananasser760)**
- **[Rawan Taha](https://github.com/rawanmohamed2023)**


