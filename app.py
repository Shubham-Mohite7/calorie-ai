import os
import json
import torch
import torchvision.transforms as T
from PIL import Image
import gradio as gr

# --- Load model ---
MODEL_PATH = "data/models/mh_net_traced.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please ensure it’s uploaded to your repository.")

model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# --- Load class-calorie mapping ---
MAPPING_PATH = "data/class_calorie_mapping.json"
if not os.path.exists(MAPPING_PATH):
    raise FileNotFoundError(f"Mapping file not found at {MAPPING_PATH}. Please ensure it’s uploaded to your repository.")

with open(MAPPING_PATH, "r") as f:
    mapping = json.load(f)
classes = list(mapping.keys())

# --- Image preprocessing ---
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# --- Prediction function ---
def predict(img):
    try:
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            out_cls, out_cal = model(x)
            pred_class = classes[out_cls.argmax().item()]
            pred_cal = out_cal.item()
        return {f"{pred_class}": f"{pred_cal:.1f} kcal"}
    except Exception as e:
        return {"error": str(e)}

# --- Gradio UI ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Food Image"),
    outputs="label",
    title="🍽️ Food Calorie Estimator AI",
    description="Upload a food image to estimate its calorie content using deep learning."
)

# --- Render-compatible server launch ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
