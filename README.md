# 🩺 Diabetic Retinopathy Detection (GCN + GAT Ensemble)

This repository contains an implementation of a **Diabetic Retinopathy detection system** using **Graph Neural Networks (GNNs)**.  
We use an **ensemble of Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)** to classify retinal fundus images into different stages of **Diabetic Retinopathy (DR)**.



## 🌟 Motivation
Diabetic Retinopathy is one of the leading causes of preventable blindness. Traditional image-based models (like CNNs) capture local patterns well but often miss structural relationships in retinal images.  
By representing an image as a **graph** (where nodes represent regions of interest and edges capture their relationships), GNNs provide a **richer representation** of retinal data.  

The **ensemble of GCN + GAT** ensures:
- **GCN** captures global feature propagation.  
- **GAT** highlights important local regions using attention weights.  
- The combination improves **accuracy, robustness, and interpretability**.  



## 🚀 Features
- 🧠 **Hybrid GNN Ensemble** → Combines **GCN** and **GAT** predictions.  
- 🔍 **Explainability** → Attention maps from GAT highlight critical retinal areas.  
- 📊 **Metrics Tracked** → Accuracy, F1-score, and Quadratic Weighted Kappa (QWK).  
- 🖼️ **Preprocessing** → Image normalization, resizing, and graph construction.  
- ⚡ **End-to-End Pipeline** → From raw fundus images → graph construction → model training → inference.  



## 📂 Dataset
This project can be trained on publicly available datasets such as:
- **Kaggle Diabetic Retinopathy Detection**  
- **APTOS 2019 Blindness Detection**  

Each image is preprocessed into a graph structure before being passed into the GNN models.



## ⚙️ Tech Stack
- **Python 3.x**  
- **PyTorch & PyTorch Geometric**  
- **TorchVision, OpenCV** (image preprocessing)  
- **NumPy, Pandas, Matplotlib** (data handling & visualization)  



## 🏗️ Project Workflow
1. **Data Preprocessing**  
   - Resize and normalize fundus images  
   - Construct graph representations  

2. **Model Training**  
   - Train GCN and GAT models separately  
   - Use weighted loss functions for class imbalance  

3. **Ensembling**  
   - Combine outputs from both models (e.g., weighted averaging or stacking)  

4. **Evaluation**  
   - Compute Accuracy, F1-score, and QWK  
   - Compare individual models vs ensemble  



## 📊 Results (Sample Template)
| Model   | Accuracy | F1-score | QWK   |
|---------|----------|----------|-------|
| CNN Baseline | 78% | 0.74 | 0.70 |
| GCN     | 82% | 0.77 | 0.75 |
| GAT     | 84% | 0.79 | 0.77 |
| **Ensemble (GCN + GAT)** | **87%** | **0.82** | **0.81** |



## 🔮 Future Improvements
- Extend ensemble with **GraphSAGE** and **GIN**  
- Experiment with **self-supervised learning** on graphs  
- Build a **FastAPI / Streamlit web app** for clinicians to upload fundus images and view predictions with heatmaps  

