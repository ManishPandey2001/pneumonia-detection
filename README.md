
# Pneumonia Detection API

This project uses a fine-tuned ResNet-50 model to classify chest X-ray images into **Pneumonia** or **Normal**. The trained model is deployed using a FastAPI web server.

## 📚 Model Details
- Model: ResNet-50
- Pretrained on: ImageNet
- Fine-tuned on: PneumoniaMNIST dataset
- Data Augmentation: Random rotation, affine transformation, color jitter, random resized crop
- Optimizer: Adam (lr=1e-5)
- Loss Function: CrossEntropyLoss with label smoothing and class weights
- Learning Rate Scheduler: ReduceLROnPlateau
- Early Stopping: Enabled (patience=5 epochs)

## ⚙️ API Endpoints

### `POST /predict`
- Upload a chest X-ray image.
- Returns prediction (`NORMAL` or `PNEUMONIA`) and probability confidence.

#### Request:
- `file`: Image file (JPG/PNG)

#### Response:
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9966,
  "probabilities": {
    "NORMAL": 0.0034,
    "PNEUMONIA": 0.9966
  }
}
