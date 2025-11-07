

---

# 🥗 **Food Calorie Estimator AI**

### 🍔 *Deep Learning App to Estimate Food Calories from Images Trained With 101,000+ Images Dataset*

![Food Calorie Estimator]

> 🚀 **AI-powered calorie prediction** from food images — built using **PyTorch**, **Gradio**, and **EfficientNet**.
> Predicts food type and estimated calories in real-time, hosted on **Render** with a smooth, interactive UI.

---

## 🌟 **Overview**

This project is an end-to-end **AI system** that:

1. Detects and classifies the food in an image 🍱
2. Estimates its calorie content 🔥
3. Displays the result through a beautiful web interface 🌐

The model was trained using the **Food-101 dataset** (over 101,000 images across 101 classes) and fine-tuned with custom **USDA nutrition data** for calorie estimation.

---

## 🧠 **Core Features**

✅ Food classification using a fine-tuned **EfficientNet-B0** CNN
✅ Calorie prediction using **USDA-based mapping**
✅ Clean and interactive **Gradio** interface
✅ Fully deployable on **Render** (or Hugging Face, Colab, etc.)
✅ Lightweight and optimized for **CPU inference**

---

## 🏗️ **Project Architecture**

```
📦 food-calorie-ai
├── app.py                        # Main Gradio application
├── requirements.txt              # Dependencies
├── data/
│   ├── class_calorie_mapping.json   # Food class → Calorie mapping
│   ├── metadata.csv                # Training metadata
│   ├── models/
│   │   ├── mh_net.pth              # PyTorch model (for retraining)
│   │   └── mh_net_traced.pt        # TorchScript model (for deployment)
│   └── usda/                       # USDA dataset folder
├── README.md
└── sample_images/                 # Example test images
```

---

## 🔬 **Model Development Pipeline**

### **Step 1. Dataset**

* **Dataset Used:** [Food-101](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)
* 101 classes × 1000 images each
* Cleaned and preprocessed using torchvision

### **Step 2. Preprocessing**

```python
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
```

### **Step 3. Model Architecture**

* Base: `EfficientNet-B0` pretrained on ImageNet
* Custom head for:

  * Softmax classification → Food Type
  * Linear regression → Calorie Prediction

### **Step 4. Training**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion_cls = torch.nn.CrossEntropyLoss()
criterion_cal = torch.nn.L1Loss()
```

✅ 8 Epochs on Colab Pro (with CUDA)
✅ Validation accuracy ~79.6%
✅ Mean Absolute Error ≈ **6.7 kcal**

### **Step 5. Export**

```python
traced = torch.jit.trace(model, torch.randn(1,3,224,224))
torch.jit.save(traced, "data/models/mh_net_traced.pt")
```

---

## 🌐 **Web App (Gradio + Render)**

### **Frontend UI**

![Gradio Interface Example]

Built using **Gradio Interface**:

```python
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Food Image"),
    outputs=[gr.Markdown()],
    title="🍔 Food Calorie Estimator AI",
    description="Upload a food image to estimate its calorie content."
)
```

### **Deployment Command**

```bash
python app.py
```

---

## ☁️ **Render Deployment Guide**

| Step                  | Command / Setting                 |
| --------------------- | --------------------------------- |
| **1️⃣ Build Command** | `pip install -r requirements.txt` |
| **2️⃣ Start Command** | `python app.py`                   |
| **3️⃣ Port**          | Render uses `8080` automatically  |
| **4️⃣ Environment**   | Python 3                          |
| **5️⃣ Public Link**   | Appears once deployed ✅           |

Ensure your `app.py` ends with:

```python
demo.launch(server_name="0.0.0.0", server_port=8080)
```

---

## ⚙️ **Local Installation**

### **1. Clone Repo**

```bash
git clone https://github.com/YourUsername/food-calorie-ai.git
cd food-calorie-ai
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run App**

```bash
python app.py
```

Open your browser at:
👉 `http://127.0.0.1:7860`

---

## 📊 **Performance Metrics**

| Metric                             | Value                          |
| ---------------------------------- | ------------------------------ |
| **Accuracy (Food Classification)** | 79.6%                          |
| **MAE (Calorie Estimation)**       | 6.76 kcal                      |
| **Inference Speed (CPU)**          | ~0.4 sec/image                 |
| **Model Size**                     | 21 MB (EfficientNet-B0 traced) |

---

## 📈 **Future Enhancements**

* [ ] Add **multi-food detection** (object detection)
* [ ] Integrate **portion-size estimation** using depth data
* [ ] Train on **custom Indian food dataset**
* [ ] Add **voice assistant** via Gradio chatbot
* [ ] Deploy on **mobile (Flutter/React Native)** using ONNX model

---

## 🧩 **Tech Stack**

| Category        | Tools                                      |
| --------------- | ------------------------------------------ |
| **Language**    | Python 3.10                                |
| **Frameworks**  | PyTorch, Torchvision                       |
| **UI**          | Gradio                                     |
| **Deployment**  | Render                                     |
| **Dataset**     | Food-101 + USDA Nutrition Data             |
| **Environment** | Google Colab (Training) + Render (Hosting) |

---

## 📚 **References**

* [Food-101 Paper](https://data.vision.ee.ethz.ch/cvl/food-101/)
* [USDA FoodData Central](https://fdc.nal.usda.gov/)
* [PyTorch](https://pytorch.org/)
* [Gradio](https://gradio.app/)
* [Render Hosting](https://render.com/)

---

## 🪪 **License**

```
MIT License

Copyright (c) 2025 Shubham Mohite

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 💡 **About the Developer**

👨‍💻 **Shubham Mohite**
AI & Full-Stack Developer | Passionate about Computer Vision and Health-Tech
🌐 [LinkedIn](https://linkedin.com/in/elite-shubham) • [GitHub](https://github.com/Shubham-Mohite7)

---

✨ *If you like this project, please give it a ⭐ on GitHub! It helps others find it too.*

---
