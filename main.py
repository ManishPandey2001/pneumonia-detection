from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io

# ------------------ FastAPI Setup ------------------
app = FastAPI(
    title="Pneumonia Detection API",
    description="An API for detecting pneumonia from chest X-ray images.",
    version="1.0"
)

# Allow CORS if you want to test from a frontend or external client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Model Setup ------------------

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model structure (must match training)
model = models.resnet50()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load trained weights
model.load_state_dict(torch.load("resnet50_pneumonia.pth", map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ------------------ API Route ------------------

@app.post("/predict", summary="Predict Pneumonia", description="Upload an X-ray image and get pneumonia prediction.")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(image)
            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            label = "PNEUMONIA" if predicted.item() == 1 else "NORMAL"

        return JSONResponse(content={
            "prediction": label,
            "confidence": round(confidence.item(), 4),
            "probabilities": {
                "NORMAL": round(probs[0][0].item(), 4),
                "PNEUMONIA": round(probs[0][1].item(), 4)
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------ Root Endpoint ------------------

@app.get("/", summary="Root", description="Root endpoint to verify API is running.")
async def root():
    return {"message": "Pneumonia Detection API is running!"}
