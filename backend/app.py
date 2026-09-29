import json
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="IRIS SVM Multi-Kernel & Evaluation API",
    description="API Phân loại hoa Iris kết hợp Endpoint đánh giá mô hình khoa học (Accuracy, Precision, Recall, F1, Confusion Matrix)",
    version="2.1.0"
)

# --- BẬT CORS ĐỂ TRANG WEB HTML KHÔNG BỊ CHẶN BỞI TRÌNH DUYỆT ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. TẢI CÁC MÔ HÌNH VÀ CÁC CHỈ SỐ ĐÁNH GIÁ VÀO BỘ NHỚ ---
model_files = {
    "linear": "model_linear.pkl",
    "rbf": "model_rbf.pkl",
    "poly": "model_poly.pkl",
    "sigmoid": "model_sigmoid.pkl"
}

models = {}

for kernel_name, file_path in model_files.items():
    if os.path.exists(file_path):
        models[kernel_name] = joblib.load(file_path)
        print(f"✅ Đã tải mô hình [{kernel_name}] từ {file_path}")
    elif os.path.exists("svm_model.pkl"):
        # Dự phòng nếu file kernel riêng chưa được tạo
        models[kernel_name] = joblib.load("svm_model.pkl")
        print(f"⚠️ Dùng tạm 'svm_model.pkl' cho kernel [{kernel_name}]")

# Đọc file chỉ số đánh giá metrics.json
metrics_data = {}
if os.path.exists("metrics.json"):
    with open("metrics.json", "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
        print("✅ Đã tải dữ liệu đánh giá mô hình 'metrics.json'")

# --- 2. CẤU HÌNH DỮ LIỆU ĐẦU VÀO ---
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    model_type: Optional[str] = "linear"

species = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

# --- 3. CÁC ENDPOINT API ---
@app.get("/")
def home():
    return {
        "message": "IRIS SVM Multi-Kernel & Evaluation API is running!",
        "available_models": list(models.keys()),
        "has_metrics": bool(metrics_data)
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "loaded_models_count": len(models)
    }

# ENDPOINT MỚI: Trả về toàn bộ dữ liệu đánh giá mô hình
@app.get("/metrics")
def get_metrics(kernel: Optional[str] = None):
    """
    Trả về chỉ số đánh giá (Accuracy, Precision, Recall, F1, Confusion Matrix)
    Nếu không truyền query parameter 'kernel', trả về dữ liệu của tất cả các mô hình.
    """
    if not metrics_data:
        raise HTTPException(
            status_code=404, 
            detail="Chưa có dữ liệu đánh giá 'metrics.json'. Vui lòng chạy train.py trước!"
        )
    
    if kernel:
        k_lower = kernel.lower()
        if k_lower in metrics_data:
            return {k_lower: metrics_data[k_lower]}
        raise HTTPException(status_code=400, detail=f"Không tìm thấy kernel '{kernel}'.")

    return metrics_data

# ENDPOINT DỰ ĐOÁN
@app.post("/predict")
def predict(data: IrisInput):
    selected_kernel = (data.model_type or "linear").lower()

    model = models.get(selected_kernel)
    if not model:
        model = next(iter(models.values())) if models else None
        selected_kernel = "default"

    if not model:
        raise HTTPException(status_code=500, detail="Chưa có mô hình nào được tải lên máy chủ!")

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    try:
        prediction = int(model.predict(features)[0])
        
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = round(float(max(probabilities)) * 100, 2)

        return {
            "class_id": prediction,
            "prediction": species.get(prediction, "unknown"),
            "kernel_used": selected_kernel,
            "confidence_percent": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi dự đoán: {str(e)}")