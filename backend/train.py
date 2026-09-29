import json
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC

# 1. Tải tập dữ liệu Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = list(iris.target_names)  # ['setosa', 'versicolor', 'virginica']

# Chia dữ liệu thành tập Train (80%) và Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40, stratify=y
)

# 2. Danh sách 4 Kernels của SVC muốn huấn luyện
kernels = ["linear", "rbf", "poly", "sigmoid"]
all_metrics = {}

print("--- ĐANG HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MODEL ---")

for k in kernels:
    # Khởi tạo và huấn luyện model với kernel tương ứng
    model = SVC(kernel=k, probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm thử (Test)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Báo cáo chi tiết Precision, Recall, F1-Score
    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Lưu riêng từng model ra file .pkl 
    file_name = f"model_{k}.pkl"
    joblib.dump(model, file_name)

    # Lưu file dự phòng svm_model.pkl cho kernel linear
    if k == "linear":
        joblib.dump(model, "svm_model.pkl")

    # Tổng hợp chỉ số đánh giá
    all_metrics[k] = {
        "accuracy": round(float(acc), 4),
        "macro_avg": {
            "precision": round(float(report["macro avg"]["precision"]), 4),
            "recall": round(float(report["macro avg"]["recall"]), 4),
            "f1_score": round(float(report["macro avg"]["f1-score"]), 4),
        },
        "per_class": {
            spec: {
                "precision": round(float(report[spec]["precision"]), 4),
                "recall": round(float(report[spec]["recall"]), 4),
                "f1_score": round(float(report[spec]["f1-score"]), 4),
                "support": int(report[spec]["support"]),
            }
            for spec in target_names
        },
        "confusion_matrix": cm,
        "classes": target_names,
    }

    print(f"-> Kernel [{k.upper()}]: Accuracy = {acc*100:.2f}% | Saved '{file_name}'")

# 3. Xuất toàn bộ chỉ số đánh giá ra file metrics.json
with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, ensure_ascii=False, indent=4)

print("\nAll Models and 'metrics.json' saved successfully!")
