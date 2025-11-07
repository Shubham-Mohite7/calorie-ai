import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import json
import os

# --- Paths ---
MODEL_PATH = "data/models/mh_net_traced.pt"
MAPPING_PATH = "data/class_calorie_mapping.json"

# --- Load model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# --- Load mapping ---
with open(MAPPING_PATH) as f:
    mapping = json.load(f)
classes = list(mapping.keys())

# --- Transform ---
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Prediction ---
def predict(img):
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out_cls, out_cal = model(x)
        food = classes[out_cls.argmax().item()]
        cal = out_cal.item()
    return f"🍽️ **{food}**\n🔥 **{cal:.1f} kcal**"

# --- Gradio Interface ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Food Image"),
    outputs=gr.Markdown(),
    title="🍔 Food Calorie Estimator AI",
    description="Upload a food image to estimate calories using a deep learning model.",
)

if __name__ == "__main__":
    # Render expects the app to bind to 0.0.0.0:8080
    demo.launch(server_name="0.0.0.0", server_port=8080)
