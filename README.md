```markdown
<p align="center">
  <img src="https://raw.githubusercontent.com/YourUsername/food-calorie-ai/main/sample_images/food-calorie-banner.png" alt="Food Calorie Estimator AI - Banner" width="100%"/>
</p>

<div align="center">

# 🥗 **Food Calorie Estimator AI**

### 🍔 *AI-Powered Calorie Prediction from Food Photos*

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-15C27A?logo=gradio&logoColor=white)
![Render](https://img.shields.io/badge/Deploy-Render-6E4AFF?logo=render&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?logo=github)

> **Upload a food photo → Get instant food type + calorie estimate**  
> Powered by **EfficientNet**, **PyTorch**, and **USDA nutrition data**.

[Live Demo](https://your-app.onrender.com) • [Model Details](#-model-pipeline) • [Run Locally](#-local-run)

</div>

---

## 🌟 **Overview**

A **complete AI pipeline** that:

1. **Detects** food in images  
2. **Classifies** into **101 categories** (Food-101)  
3. **Estimates calories** using USDA data  
4. **Serves results** via a sleek **Gradio web app**

---

## 🧠 **Core Features**

| Feature | Description |
|--------|-------------|
| Food Classification | Fine-tuned **EfficientNet-B0** → **79.6% accuracy** |
| Calorie Prediction | **MAE ≈ 6.7 kcal** per serving |
| Interactive UI | Drag & drop → instant results |
| Fast & Lightweight | **~0.4s/image** on CPU, **21 MB** model |
| Deploy Anywhere | Render, Hugging Face, or local |

---

## 🏗️ **Project Structure**

```bash
food-calorie-ai/
├── app.py                        # Gradio web app
├── requirements.txt              # Dependencies
├── data/
│   ├── class_calorie_mapping.json
│   ├── metadata.csv
│   ├── models/
│   │   ├── mh_net.pth
│   │   └── mh_net_traced.pt
│   └── usda/
├── sample_images/
│   ├── pizza.jpg
│   └── salad.jpg
└── README.md
```

---

## 🔬 **Model Pipeline**

### 1. Dataset
- **Food-101**: 101 classes × 1,000 images
- Augmented with `torchvision`

### 2. Preprocessing
```python
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### 3. Architecture
```text
EfficientNet-B0 (ImageNet)
├── Classifier → 101 classes (CrossEntropy)
└── Regressor  → kcal (L1 Loss)
```

### 4. Training
```python
optimizer = Adam(lr=3e-4)
criterion_cls = CrossEntropyLoss()
criterion_cal = L1Loss()
```
- **8 epochs** on **Colab Pro (GPU)**
- **Accuracy**: **79.6%**
- **MAE**: **6.76 kcal**

### 5. Export
```python
traced = torch.jit.trace(model, torch.randn(1,3,224,224))
traced.save("data/models/mh_net_traced.pt")
```

---

## 🌐 **Web App (Gradio)**

<p align="center">
  <img src="https://raw.githubusercontent.com/YourUsername/food-calorie-ai/main/sample_images/ui_demo.gif" alt="Gradio UI" width="80%"/>
  <br><em>↑ Upload image → Get prediction in <1 sec</em>
</p>

```python
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Markdown(),
    title="Food Calorie Estimator AI",
    examples=[["sample_images/pizza.jpg"]]
)
```

---

## ⚙️ **Local Run**

```bash
git clone https://github.com/YourUsername/food-calorie-ai.git
cd food-calorie-ai
pip install -r requirements.txt
python app.py
```

**Open**: [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## ☁️ **Deploy on Render**

| Setting | Value |
|-------|-------|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python app.py` |
| Port | `8080` |
| Python | `3.10` |

> In `app.py`:
> ```python
> demo.launch(server_name="0.0.0.0", server_port=8080)
> ```

---

## 📊 **Performance**

| Metric | Value |
|-------|-------|
| Accuracy | **79.6%** |
| Calorie MAE | **6.76 kcal** |
| Inference (CPU) | **~0.4 sec** |
| Model Size | **21 MB** |

---

## Future Enhancements

<details>
<summary><strong>Planned Features</strong></summary>

- Multi-food detection (YOLOv8)
- Portion size estimation
- Indian food dataset
- Mobile app (Flutter + ONNX)
- Voice input (Gradio mic)

</details>

---

## 🧩 **Tech Stack**

| Category | Tools |
|--------|-------|
| Language | Python 3.10 |
| ML | PyTorch, Torchvision |
| UI | Gradio |
| Deploy | Render |
| Data | Food-101 + USDA |

---

## 📚 **References**

- [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [USDA FoodData Central](https://fdc.nal.usda.gov/)
- [PyTorch](https://pytorch.org/)
- [Gradio](https://gradio.app/)
- [Render](https://render.com/)

---

## 🪪 **License**

```text
MIT License (c) 2025 Shubham Mohite
```

---

## 💡 **About the Developer**

<div align="center">

**Shubham Mohite**  
*AI & Full-Stack Developer | Computer Vision | Health-Tech*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/shubhammohite)  
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/YourUsername)

</div>

---

<p align="center">
  <strong>Love this project? Star it on GitHub!</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/YourUsername/food-calorie-ai?style=social" alt="Stars"/>
  <img src="https://img.shields.io/github/forks/YourUsername/food-calorie-ai?style=social" alt="Forks"/>
</p>
```

---

### 2. Add This **Custom Banner Image**

**File:** `sample_images/food-calorie-banner.png`

**Image URL (hosted via GitHub):**
```
https://raw.githubusercontent.com/YourUsername/food-calorie-ai/main/sample_images/food-calorie-banner.png
```

#### Download & Add This Banner (1200×300px)

![Food Calorie Estimator AI Banner](https://i.imgur.com/9kR5vL2.png)

> **Right-click → Save Image As** → Save as:  
> `sample_images/food-calorie-banner.png`

**Then push to GitHub:**
```bash
git add sample_images/food-calorie-banner.png
git commit -m "Add professional banner"
git push
```

---

### Final Folder Structure
```
food-calorie-ai/
├── app.py
├── requirements.txt
├── data/
│   └── models/
├── sample_images/
│   ├── food-calorie-banner.png   ← NEW
│   ├── pizza.jpg
│   └── salad.jpg
└── README.md                     ← Paste above content
```

---
