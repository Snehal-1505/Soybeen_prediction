# app.py
import os
import json
from datetime import datetime

from flask import Flask, render_template, request, redirect, session, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image

from pymongo import MongoClient, errors

from tensorflow.keras.models import load_model

# ---------------------------
# Configuration
# ---------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace_this_with_a_secret")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB connection (change to your Atlas URI if needed)
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("DB_NAME", "soyabeen_prediction_db")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

users_collection = db["users"]
predictions_collection = db["predictions"]

# create unique index for username to avoid duplicates (safe to call repeatedly)
try:
    users_collection.create_index("username", unique=True)
except errors.PyMongoError:
    pass

# ---------------------------
# Load ML model and classes
# ---------------------------
MODEL_PATH = "model.h5"
CLASS_NAMES_JSON = "class_names.json"
DATASET_FALLBACK = r"D:\Major_pro\soyabeen_prediction\dataset"  # change if necessary

model = load_model(MODEL_PATH)

if os.path.exists(CLASS_NAMES_JSON):
    with open(CLASS_NAMES_JSON, "r") as f:
        class_names = json.load(f)
else:
    # fallback: read dataset folder (alphabetical)
    if os.path.exists(DATASET_FALLBACK):
        class_names = sorted(
            [d for d in os.listdir(DATASET_FALLBACK) if os.path.isdir(os.path.join(DATASET_FALLBACK, d))]
        )
    else:
        class_names = []
print("Loaded class names:", class_names)

# ---------------------------
# Helpers
# ---------------------------
def preprocess_image(path, target_size=(150, 150)):
    img = Image.open(path).convert("RGB").resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Gather form values
        fullname = request.form.get("fullname", "").strip()
        email = request.form.get("email", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        phone = request.form.get("phone", "").strip()
        dob = request.form.get("dob", "").strip()
        address = request.form.get("address", "").strip()

        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for("register"))

        # Prepare user document
        user_doc = {
            "fullname": fullname,
            "email": email,
            "username": username,
            "password_hash": generate_password_hash(password),
            "phone": phone,
            "dob": dob,
            "address": address,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            users_collection.insert_one(user_doc)
            flash("Registered successfully! You can now login.", "success")
            return redirect(url_for("login"))
        except errors.DuplicateKeyError:
            flash("Username already exists! Choose another.", "error")
            return redirect(url_for("register"))
        except Exception as e:
            flash(f"Database error: {e}", "error")
            return redirect(url_for("register"))

    # GET request
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = users_collection.find_one({"username": username})
        if user and check_password_hash(user.get("password_hash", ""), password):
            session["username"] = username
            flash("Logged in successfully!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "error")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    return render_template("dashboard.html", username=session["username"])


@app.route("/profile")
def profile():
    if "username" not in session:
        return redirect(url_for("login"))

    user = users_collection.find_one({"username": session["username"]}, {"_id": 0, "password_hash": 0})
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("login"))

    return render_template("profile.html", user=user)


@app.route("/past-report")
def past_report():
    if "username" not in session:
        return redirect(url_for("login"))

    user_predictions = list(
        predictions_collection.find({"username": session["username"]}, {"_id": 0}).sort("timestamp", -1)
    )
    return render_template("past_report.html", reports=user_predictions, username=session["username"])


@app.route("/upload_img", methods=["GET", "POST"])
def upload_img():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part", "error")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No selected file", "error")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Preprocess + predict
        try:
            img_array = preprocess_image(filepath, target_size=(150, 150))
            prediction = model.predict(img_array)
            idx = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
        except Exception as e:
            flash(f"Prediction error: {e}", "error")
            return redirect(request.url)

        if 0 <= idx < len(class_names):
            predicted_class = class_names[idx]
        else:
            predicted_class = "Unknown"

        # Store in MongoDB
        report_doc = {
            "username": session["username"],
            "image": filename,
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            predictions_collection.insert_one(report_doc)
        except Exception as e:
            flash(f"Failed to save report: {e}", "warning")

        return render_template("result.html", prediction=predicted_class, confidence=round(confidence, 2), image_path=filepath)

    return render_template("upload_img.html")


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        db["feedback"].insert_one({
            "name": name,
            "email": email,
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        flash("✅ Thank you for your feedback!", "success")
        return redirect(url_for('contact'))

    return render_template('contact.html')


@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("Logged out.", "success")
    return redirect(url_for("login"))


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
