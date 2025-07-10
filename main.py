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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ ResNet50 Model ------------------
resnet_model = models.resnet50()
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.load_state_dict(torch.load("resnet50_pneumonia.pth", map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()

resnet_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ------------------ InceptionV3 Model ------------------

inception_model = models.inception_v3(weights=None, aux_logits=True, init_weights=False)  # don't load ImageNet weights
inception_model.fc = nn.Linear(inception_model.fc.in_features, 2)     # match your saved model
inception_model.load_state_dict(torch.load("inception_pneumonia.pth", map_location=device))
inception_model = inception_model.to(device)
inception_model.eval()

inception_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ------------------ Original ResNet50 Endpoint ------------------
@app.post("/predict", summary="Predict Pneumonia (ResNet50)", description="Upload an X-ray image and get pneumonia prediction (ResNet50).")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = resnet_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = resnet_model(image)
            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            label = "PNEUMONIA" if predicted.item() == 1 else "NORMAL"

        return JSONResponse(content={
            "model": "ResNet50",
            "prediction": label,
            "confidence": round(confidence.item(), 4),
            "probabilities": {
                "NORMAL": round(probs[0][0].item(), 4),
                "PNEUMONIA": round(probs[0][1].item(), 4)
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------ New InceptionV3 Endpoint ------------------
@app.post("/predict-inception", summary="Predict Pneumonia (InceptionV3)", description="Upload an X-ray image and get pneumonia prediction (InceptionV3).")
async def predict_inception(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = inception_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = inception_model(image)
            if isinstance(output, tuple):  # due to aux_logits=True
                output = output[0]
            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            label = "PNEUMONIA" if predicted.item() == 1 else "NORMAL"

        return JSONResponse(content={
            "model": "InceptionV3",
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
    return {"message": "Pneumonia Detection API is running with ResNet50 and InceptionV3!"}
