# 🎗️ Multimodal Fusion of Text-Image Classification for Breast Cancer

## 📌 Overview
This project implements a multimodal deep learning approach that fuses **text data** and **medical images** for breast cancer classification. By combining both modalities, the model achieves better diagnostic accuracy than single-modality approaches.

## 🗂️ Project Structure
```
├── app.py               # Main application / Gradio UI
├── dataset.py           # Dataset loading and preprocessing
├── model.py             # Multimodal fusion model architecture
├── train.py             # Model training script
├── setup_and_run.py     # Setup and execution helper
├── requirements.txt     # Python dependencies
└── best_model.pth       # Saved best model weights
```

## 🧠 Model Architecture
- **Text Branch:** Processes clinical/pathology text using Transformers
- **Image Branch:** Processes breast cancer images using CNN
- **Fusion Layer:** Combines both modalities for final classification
- **Output:** Breast cancer binary classification

## 📦 Installation
```bash
git clone https://github.com/Manya1892005/Multimodal-Fusion-Of-Text-Image-Classification-for-Breast-Cancer.git
cd Multimodal-Fusion-Of-Text-Image-Classification-for-Breast-Cancer
pip install -r requirements.txt
```

## 🚀 How to Run
```bash
python train.py
python app.py
```

## 📊 Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 100%   |
| F1 Score  | 1.0000 |
| ROC-AUC   | 1.0000 |
| Val Acc   | 100%   |

> Training completed in 3 epochs with final Train Acc: 0.995, Loss: 0.010

## 🛠️ Technologies Used
- Python
- PyTorch
- Torchvision
- Transformers (HuggingFace)
- Gradio
- Pillow
- NumPy

## 👩‍💻 Author
**Manya** — [GitHub Profile](https://github.com/Manya1892005)

## 📄 License
This project is open-source under the [MIT License](LICENSE).
