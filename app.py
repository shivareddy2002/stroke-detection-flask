from flask import Flask, render_template, request, url_for, send_from_directory
import math
import numpy as np
import time
import os
import uuid
import tempfile
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFilter

# Optional TensorFlow model load
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

app = Flask(__name__)

# -----------------------
# Directory setup
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
RUNTIME_DIR = os.path.join(tempfile.gettempdir(), "brainstroke_runtime")
UPLOADS_DIR = os.path.join(RUNTIME_DIR, "uploads")
IMAGES_DIR = os.path.join(RUNTIME_DIR, "images")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "GA_BiGRU_Improved.h5")
CHROMOSOME_PATH = os.path.join(BASE_DIR, "GA_BiGRU_best_chromosome.npy")

model = None
selected_features = None

try:
    if load_model is not None and os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    if os.path.exists(CHROMOSOME_PATH):
        best_chromosome = np.load(CHROMOSOME_PATH)
        selected_features = np.flatnonzero(best_chromosome)
    print("✅ Model & features initialized.")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")

# -----------------------
# Helper Functions
# -----------------------
def preprocess_image(img_path, image_size=(128, 128), selected_features=None):
    img = Image.open(img_path).convert("L").resize(image_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    flat_img = img_array.flatten()
    if selected_features is not None:
        valid_idx = [i for i in selected_features if 0 <= i < len(flat_img)]
        flat_img = flat_img[valid_idx] if valid_idx else flat_img
    return flat_img.reshape(1, 1, -1)


def heuristic_predict(img_path, image_size=(128, 128)):
    """Lightweight fallback predictor for serverless deployments without TensorFlow."""
    img = Image.open(img_path).convert("L").resize(image_size)
    arr = np.array(img, dtype=np.float32) / 255.0

    mean_intensity = float(arr.mean())
    std_intensity = float(arr.std())

    gx = np.abs(np.diff(arr, axis=1)).mean()
    gy = np.abs(np.diff(arr, axis=0)).mean()
    edge_strength = float((gx + gy) / 2)

    center = arr[36:92, 36:92]
    border_mask = np.ones_like(arr, dtype=bool)
    border_mask[24:104, 24:104] = False
    border = arr[border_mask]
    center_contrast = float(center.mean() - border.mean())

    # Weighted feature score, then mapped to [0,1] with sigmoid.
    score = (
        2.4 * (mean_intensity - 0.42)
        + 2.0 * (std_intensity - 0.19)
        + 3.2 * (edge_strength - 0.13)
        + 1.8 * center_contrast
    )
    prob = 1 / (1 + math.exp(-score))
    return float(np.clip(prob, 0.01, 0.99)), {
        "mean_intensity": round(mean_intensity, 4),
        "std_intensity": round(std_intensity, 4),
        "edge_strength": round(edge_strength, 4),
        "center_contrast": round(center_contrast, 4),
    }

def generate_demo_scan(output_path, size=(512, 512), seed=None):
    if seed is not None:
        np.random.seed(seed)
    w, h = size
    base = Image.new("L", size)
    cx, cy = w / 2, h / 2
    maxr = (cx**2 + cy**2) ** 0.5
    pixels = base.load()
    for x in range(w):
        for y in range(h):
            r = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            val = int(40 + 200 * (1 - (r / maxr)))
            pixels[x, y] = max(0, min(255, val))
    draw = ImageDraw.Draw(base)
    for i in range(6):
        pad = 15 + i * 18
        bbox = [pad, int(pad * 0.8), w - pad, int(h - pad * 0.8)]
        outline = max(0, min(255, int(100 - i * 8 + np.random.randint(-5, 5))))
        draw.ellipse(bbox, outline=outline, width=2)
    lesion_r = int(min(w, h) * 0.08)
    lesion_cx, lesion_cy = int(cx + np.random.randint(-40, 40)), int(cy + np.random.randint(-40, 40))
    draw.ellipse(
        [lesion_cx - lesion_r, lesion_cy - lesion_r, lesion_cx + lesion_r, lesion_cy + lesion_r],
        fill=220
    )
    arr = np.array(base, dtype=np.int16)
    noise = np.random.normal(0, 10, (h, w)).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype("uint8")
    Image.fromarray(arr, "L").filter(ImageFilter.GaussianBlur(radius=1.0)).save(output_path)

# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("result.html", error="❌ No file selected.")

    try:
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        save_path = os.path.join(UPLOADS_DIR, unique_name)
        file.save(save_path)

        start = time.time()
        engine = "Heuristic"
        feature_count = 4
        insights = None

        if model is not None and selected_features is not None:
            img = preprocess_image(save_path, selected_features=selected_features)
            prob = model.predict(img, verbose=0)[0][0]
            engine = "GA-BiGRU"
            feature_count = len(selected_features)
        else:
            prob, insights = heuristic_predict(save_path)

        duration = round(time.time() - start, 2)

        label = "Stroke" if prob > 0.5 else "Normal"
        conf = float(prob * 100 if prob > 0.5 else (1 - prob) * 100)
        preview_url = url_for("runtime_file", folder="uploads", filename=unique_name)

        return render_template("result.html",
                               pred_class=label,
                               confidence=round(conf, 1),
                               time_taken=duration,
                               features=feature_count,
                               engine=engine,
                               insights=insights,
                               preview_image=preview_url,
                               demo=False)
    except Exception as e:
        return render_template("result.html", error=f"❌ Prediction failed: {e}")



@app.route("/runtime/<folder>/<path:filename>")
def runtime_file(folder, filename):
    if folder not in {"uploads", "images"}:
        return "Invalid folder", 404
    target_dir = os.path.join(RUNTIME_DIR, folder)
    return send_from_directory(target_dir, filename)

@app.route("/demo")
def demo():
    demo_file = os.path.join(IMAGES_DIR, "demo_scan.png")
    try:
        generate_demo_scan(demo_file, seed=42)
    except Exception as e:
        print("Demo generation error:", e)
    preview_url = url_for("runtime_file", folder="images", filename="demo_scan.png")
    return render_template("result.html",
                           pred_class="Stroke",
                           confidence=91.3,
                           time_taken=0.42,
                           features=500,
                           preview_image=preview_url,
                           demo=True)

# For Vercel (expose 'app' only)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
