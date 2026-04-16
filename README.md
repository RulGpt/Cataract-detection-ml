# Cataract Detection using Deep Learning

## 🧠 Project Overview
This project uses a pretrained MobileNetV2 model to detect cataract from eye images.

The model is trained using transfer learning and can classify images into:
- Cataract
- Normal

---

## 🚀 Features
- Image classification using CNN
- Transfer learning (MobileNetV2)
- Training + Inference pipeline
- Batch prediction from folder

---

## 📁 Project Structure
cataract-detection-ml/
│
├── dataset/
├── images_to_process/
├── models/
├── src/
├── requirements.txt
└── README.md


---

## ⚙️ Installation

```bash
pip install -r requirements.txt



🏋️ Training
python src/train.py

🔍 Prediction
python src/predict.py

Model
Architecture: MobileNetV2
Framework: PyTorch
Approach: Transfer Learning

Note

This is a proof-of-concept model trained on a small dataset.
Performance will improve with more data.




---Author
Rahul-Gupta

# 🌐 5. Create GitHub Repo

## Step-by-step:

```bash
cd cataract-detection-ml
git init
git add .
git commit -m "Initial commit - cataract detection ML model"