# 🌿 Plant Disease Detection using CNN & TensorFlow

This project is a smart AI-powered plant disease detection system that classifies plant diseases using image data and updates the prediction to a Google Sheet in real-time. Ideal for agriculture, smart farming, or AIoT-based projects.

---

## 🧠 Features

- 🔍 Image classification of plant diseases using Convolutional Neural Networks (CNN)
- 📁 Automatically loads and preprocesses dataset from structured folders
- 🧪 Train/test split, normalization, and label encoding included
- 🧠 Model built using TensorFlow and Keras
- 💾 Model saved in `.h5` format
- ✅ Real-time disease prediction from a new image
- 📤 Integration with Google Sheets using `gspread` for live updates

---

## 📂 Dataset

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Structure**:
PlantVillage/ ├── Apple___Black_rot/ ├── Apple___healthy/ ├── Corn___Cercospora_leaf_spot Gray_leaf_spot/ └── ...


---

## 🛠️ Technologies Used

| Tool/Library    | Purpose                          |
|----------------|----------------------------------|
| Python          | Programming Language             |
| TensorFlow/Keras| Model Building & Training        |
| OpenCV          | Image Preprocessing              |
| NumPy           | Array Manipulation               |
| scikit-learn    | Train/Test Split & Label Encoding|
| gspread         | Google Sheets API Integration    |
| oauth2client    | Google Auth                      |

---

## 🧪 Model Architecture

A simple yet effective CNN model:
- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Flatten → Dense → Dropout
- Output layer: Softmax (multi-class)

---

## 🧬 Training

- Input shape: `(128, 128, 3)`
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Epochs: 25 (with EarlyStopping)
- Accuracy: ~depends on dataset & tuning

---

## 💾 Files

| File                     | Description                                  |
|--------------------------|----------------------------------------------|
| `plant_disease_model.h5` | Trained CNN model in HDF5 format             |
| `label_encoder.pkl`      | Encoded labels for disease classes           |
| `plant-426909-*.json`    | Google Service Account credentials file      |
| `main.py`                | Complete training + prediction script        |

---

## 🖼️ Predict New Images

```python
predicted_disease = predict_disease("test_image.jpg", model)
print(f"The predicted disease is: {predicted_disease}")


📤 Google Sheets Integration
After prediction, the result is pushed to Google Sheets:
update_google_sheet(predicted_disease, "live plant detection")
