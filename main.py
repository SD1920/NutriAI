
import torch, timm, json, io
import torch.nn.functional as F
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ── Load config ────────────────────────────────────────────────────────────────
with open("classes.json") as f:
    config = json.load(f)
with open("nutrition_db.json") as f:
    nutrition_db = json.load(f)

CLASSES    = config["classes"]
N_CLASSES  = config["num_classes"]
IMG_SIZE   = config["input_size"]
THRESHOLD  = config["threshold"]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ─────────────────────────────────────────────────────────────────
model = timm.create_model("convnext_tiny", pretrained=False, num_classes=N_CLASSES)
ckpt  = torch.load("convnext_tiny_best.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval().to(DEVICE)
print(f"Model loaded: {N_CLASSES} classes on {DEVICE}")

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI(title="FitAI Food Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "model": "convnext_tiny",
            "classes": N_CLASSES, "threshold": THRESHOLD}

@app.post("/detect-food")
async def detect_food(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image")

    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    top_probs, top_idx = probs.topk(3)

    predictions = [
        {"dish": CLASSES[i], "confidence": round(p.item() * 100, 1)}
        for p, i in zip(top_probs, top_idx)
    ]
    top_dish  = predictions[0]["dish"]
    top_conf  = predictions[0]["confidence"]
    status    = "ok" if top_conf >= THRESHOLD else "low_confidence"
    nutrition = nutrition_db.get(top_dish, {})

    return {
        "success":     True,
        "status":      status,
        "top_dish":    top_dish,
        "confidence":  top_conf,
        "predictions": predictions,
        "nutrition":   nutrition,
        "message":     (
            f"Detected: {top_dish} ({top_conf}%)" if status == "ok"
            else f"Low confidence ({top_conf}%). Please type the food name or pick from suggestions."
        ),
    }
