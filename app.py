import os
import cv2
import numpy as np
import pickle
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# Load model and encoder
model = load_model("plant_disease_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Settings
img_size = 128

# Init Flask
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Disease to medicine and quantity mapping
disease_medicine_map = {
    "Pepper__bell___Bacterial_spot": ("Copper Oxychloride", "3 g/L"),
    "Pepper__bell___healthy": ("No medicine required", "-"),
    "Potato___Early_blight": ("Mancozeb", "2.5 g/L"),
    "Potato___healthy": ("No medicine required", "-"),
    "Potato___Late_blight": ("Metalaxyl", "2 ml/L"),
    "Tomato__Target_Spot": ("Azoxystrobin", "1 ml/L"),
    "Tomato__Tomato_mosaic_virus": ("Rogor or Neem Oil", "2 ml/L"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("Imidacloprid", "1 ml/L"),
    "Tomato_Bacterial_spot": ("Copper Hydroxide", "2.5 g/L"),
    "Tomato_Early_blight": ("Chlorothalonil", "2 g/L"),
    "Tomato_healthy": ("No medicine required", "-"),
    "Tomato_Late_blight": ("Metalaxyl + Mancozeb", "2 g/L"),
    "Tomato_Leaf_Mold": ("Mancozeb or Chlorothalonil", "2 g/L"),
    "Tomato_Septoria_leaf_spot": ("Thiophanate-methyl", "2 g/L"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("Abamectin", "0.5 ml/L")
}

# Prediction function
def predict_disease(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Image not valid", "", ""
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    disease = str(le.inverse_transform(predicted_class)[0])
    disease_display = disease.replace("_", " ")
    medicine, quantity = disease_medicine_map.get(disease, ("Not available", "-"))
    return disease_display, medicine, quantity

# Show register page on first load
@app.route("/")
def home():
    return redirect(url_for("register"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return "Username already exists"
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[0], password):
            session["user"] = username
            return redirect(url_for("predict"))
        else:
            return "Invalid Credentials"
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))
    disease = medicine = quantity = None
    image = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            disease, medicine, quantity = predict_disease(file_path)
            image = file.filename
    return render_template("index.html", prediction=disease, medicine=medicine, quantity=quantity, image=image)

if __name__ == "__main__":
    app.run(debug=True)
