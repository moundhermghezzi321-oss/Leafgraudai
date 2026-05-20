import os
from datetime import datetime
from uuid import uuid4

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify
)

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from PIL import Image

import torch
import torchvision.transforms as transforms
import gdown

# =========================================================
# FLASK APP
# =========================================================

app = Flask(
    __name__,
    template_folder="."
)

app.secret_key = "leafguard_secret_key"

# =========================================================
# DATABASE
# =========================================================

database_url = os.environ.get(
    "DATABASE_URL",
    "sqlite:////tmp/leafguard.db"
)

if database_url.startswith("postgres://"):
    database_url = database_url.replace(
        "postgres://",
        "postgresql://",
        1
    )

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# =========================================================
# UPLOAD FOLDER
# =========================================================

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================================================
# MODEL DOWNLOAD
# =========================================================

MODEL_PATH = "vit_mobile.pt"

FILE_ID = "1wjZXCNyiHg8864ZsKcntGulNWcDcHH_G"

if not os.path.exists(MODEL_PATH):

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )

# =========================================================
# LABELS
# =========================================================

LABELS = [
    "Black Rot",
    "ESCA",
    "Healthy",
    "Leaf Blight"
]

# =========================================================
# MODEL
# =========================================================

model = torch.jit.load(
    MODEL_PATH,
    map_location="cpu"
)

model.eval()

# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

# =========================================================
# USER MODEL
# =========================================================

class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(120),
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(300),
        nullable=False
    )

    role = db.Column(
        db.String(20),
        default="farmer"
    )

# =========================================================
# HISTORY MODEL
# =========================================================

class History(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    farmer_id = db.Column(
        db.Integer
    )

    image_path = db.Column(
        db.String(300)
    )

    disease = db.Column(
        db.String(120)
    )

    confidence = db.Column(
        db.Float
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

# =========================================================
# CREATE DATABASE
# =========================================================

with app.app_context():

    db.create_all()

# =========================================================
# HELPERS
# =========================================================

def current_user():

    if "user_id" not in session:
        return None

    return db.session.get(
        User,
        session["user_id"]
    )

# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    user = current_user()

    if not user:
        return redirect(url_for("login"))

    if user.role == "admin":
        return redirect(url_for("admin_dashboard"))

    return redirect(url_for("farmer_dashboard"))

# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"].strip()

        password = request.form["password"].strip()

        user = User.query.filter_by(
            email=email
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            session["user_id"] = user.id

            return redirect(url_for("index"))

        flash("Invalid credentials")

    return render_template("login.html")

# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"].strip()

        email = request.form["email"].strip()

        password = request.form["password"].strip()

        existing = User.query.filter_by(
            email=email
        ).first()

        if existing:

            flash("Email already exists")

            return redirect(url_for("register"))

        user = User(

            name=name,

            email=email,

            password=generate_password_hash(password),

            role="farmer"
        )

        db.session.add(user)

        db.session.commit()

        flash("Account created successfully")

        return redirect(url_for("login"))

    return render_template("register.html")

# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))

# =========================================================
# FARMER DASHBOARD
# =========================================================

@app.route("/farmer")
def farmer_dashboard():

    user = current_user()

    if not user:
        return redirect(url_for("login"))

    history = History.query.filter_by(
        farmer_id=user.id
    ).order_by(
        History.id.desc()
    ).all()

    return render_template(
        "farmer.html",
        history=history
    )

# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/admin")
def admin_dashboard():

    user = current_user()

    if not user:
        return redirect(url_for("login"))

    users = User.query.all()

    histories = History.query.all()

    return render_template(
        "admin.html",
        users=users,
        histories=histories
    )

# =========================================================
# EXPERT DASHBOARD
# =========================================================

@app.route("/expert")
def expert_dashboard():

    user = current_user()

    if not user:
        return redirect(url_for("login"))

    return render_template("expert.html")

# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    user = current_user()

    if not user:

        return jsonify({
            "error": "Unauthorized"
        }), 401

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    filename = secure_filename(file.filename)

    unique_name = f"{uuid4().hex}_{filename}"

    save_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_name
    )

    file.save(save_path)

    image = Image.open(save_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs[0],
            dim=0
        )

        confidence, predicted = torch.max(
            probabilities,
            0
        )

    disease = LABELS[predicted.item()]

    confidence_value = round(
        confidence.item() * 100,
        2
    )

    history = History(

        farmer_id=user.id,

        image_path=save_path,

        disease=disease,

        confidence=confidence_value
    )

    db.session.add(history)

    db.session.commit()

    return jsonify({

        "disease": disease,

        "confidence": confidence_value,

        "image": save_path

    })

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
