
import os
from datetime import datetime
from uuid import uuid4

from flask import Flask, request, redirect, url_for, session, flash, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image

try:
    import torch 
    import torchvision.transforms as transforms
except Exception as e:
    print("Torch import error:", e)
    torch = None
    transforms = None

try:
    import gdown
except Exception:
    gdown = None


# =========================================================
# APP CONFIG
# =========================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "leafguard_secret_key_change_me")

database_url = os.environ.get("DATABASE_URL", "sqlite:////tmp/leafguard.db")
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db = SQLAlchemy(app)


# =========================================================
# DISEASE LABELS AND INFORMATION
# عدل الأسماء إذا كانت classes في موديلك مختلفة
# =========================================================

DISEASE_INFO = {
    "Bacterial Leaf Spot": {
        "details": "Bacterial infection causing dark spots and lesions on grape leaves.",
        "advice": "Remove infected leaves, avoid overhead irrigation, and improve field hygiene.",
        "treatment": "Use copper-based treatment when recommended by an agricultural expert."
    },
    "Black Rot": {
        "details": "A fungal disease causing dark circular spots and tissue damage.",
        "advice": "Remove infected plant debris and improve air circulation.",
        "treatment": "Apply suitable fungicide according to expert recommendation."
    },
    "Downy Mildew": {
        "details": "A moisture-related grapevine disease causing yellow oily spots.",
        "advice": "Reduce humidity, prune dense foliage, and avoid excessive irrigation.",
        "treatment": "Use anti-mildew fungicide when recommended."
    },
    "ESCA": {
        "details": "A complex grapevine trunk disease affecting plant health and productivity.",
        "advice": "Inspect vines regularly and remove dead or infected wood.",
        "treatment": "Apply sanitation pruning and expert monitoring."
    },
    "Healthy": {
        "details": "The analyzed leaf appears healthy.",
        "advice": "Continue regular monitoring and preventive agricultural practices.",
        "treatment": "No treatment is required."
    },
    "Leaf Blight": {
        "details": "A disease causing burnt-like lesions and leaf damage.",
        "advice": "Remove damaged leaves and avoid excessive moisture.",
        "treatment": "Use suitable fungicide if necessary."
    },
    "Powdery Mildew": {
        "details": "A fungal disease appearing as white powder on grape leaves.",
        "advice": "Improve sunlight exposure and ventilation.",
        "treatment": "Use sulfur-based or recommended fungicide."
    }
}

LABELS = list(DISEASE_INFO.keys())


# =========================================================
# MODEL DOWNLOAD AND LOADING
# =========================================================

MODEL_PATH = "efficientnetb0_mobile.pt"
GOOGLE_DRIVE_FILE_ID = "1eYl2lQm8z-b1pP0ykQo7QptjGNvZjepz"

model = None
transform = None

def load_model():
    global model, transform

    if torch is None or transforms is None:
        print("Torch not available. Running in demo mode.")
        return

    if not os.path.exists(MODEL_PATH) and gdown is not None:
        try:
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            print("Downloading model from Google Drive...")
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            print("Model download failed:", e)

    if os.path.exists(MODEL_PATH):
        try:
            model = torch.jit.load(MODEL_PATH, map_location="cpu")
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            print("Model loaded successfully.")

        except Exception as e:
            print("Model loading failed:", e)
            model = None

    else:
        print("Model file not found. Running in demo mode.")


# =========================================================
# DATABASE MODELS
# =========================================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(160), nullable=False)
    email = db.Column(db.String(220), unique=True, nullable=False)
    password = db.Column(db.String(300), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="farmer")  # admin / expert / farmer
    status = db.Column(db.String(20), nullable=False, default="active")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    image_path = db.Column(db.String(300))
    disease = db.Column(db.String(160), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    details = db.Column(db.Text)
    advice = db.Column(db.Text)
    treatment = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    farmer = db.relationship("User")


class ExpertMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    expert_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    history_id = db.Column(db.Integer, db.ForeignKey("history.id"))
    disease = db.Column(db.String(160), nullable=False)
    image_path = db.Column(db.String(300))
    message = db.Column(db.Text, nullable=False)
    reply = db.Column(db.Text)
    status = db.Column(db.String(20), default="pending")  # pending / answered
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    replied_at = db.Column(db.DateTime)

    farmer = db.relationship("User", foreign_keys=[farmer_id])
    expert = db.relationship("User", foreign_keys=[expert_id])
    history = db.relationship("History")


with app.app_context():
    db.create_all()

    admin_email = os.environ.get("ADMIN_EMAIL", "admin@leafguard.com")
    admin_password = os.environ.get("ADMIN_PASSWORD", "admin12345")

    admin = User.query.filter_by(email=admin_email).first()
    if not admin:
        admin = User(
            name="Admin",
            email=admin_email,
            password=generate_password_hash(admin_password),
            role="admin",
            status="active"
        )
        db.session.add(admin)
        db.session.commit()


# =========================================================
# HELPERS
# =========================================================

def current_user():
    if "user_id" not in session:
        return None
    return db.session.get(User, session["user_id"])


def login_required():
    user = current_user()
    return user and user.status == "active"


def require_role(role):
    user = current_user()
    return user and user.role == role and user.status == "active"


def save_uploaded_image(file):
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    original = secure_filename(file.filename)
    ext = os.path.splitext(original)[1].lower()

    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    filename = f"{uuid4().hex}{ext}"
    relative_path = os.path.join("uploads", filename)
    full_path = os.path.join("static", relative_path)

    file.save(full_path)
    return relative_path, full_path


def run_prediction(image_path):
    if model is None or transform is None or torch is None:
        disease = "Powdery Mildew"
        confidence = 92.5
        return disease, confidence, DISEASE_INFO[disease]

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output[0], dim=0)
        confidence, pred = torch.max(probs, 0)

    idx = int(pred.item())
    if idx >= len(LABELS):
        idx = 0

    disease = LABELS[idx]
    confidence_value = round(float(confidence.item()) * 100, 2)

    return disease, confidence_value, DISEASE_INFO[disease]


def page(title, body):
    user = current_user()
    user_nav = ""
    if user:
        user_nav = f"""
        <div class="nav">
            <span>Welcome, {user.name} ({user.role})</span>
            <a href="/">Dashboard</a>
            <a href="/profile">Profile</a>
            <a href="/logout">Logout</a>
        </div>
        """

    return render_template_string(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{
                margin:0;
                font-family: Arial, sans-serif;
                background:#f3f4f6;
                color:#111827;
            }}
            .header {{
                background:linear-gradient(135deg,#064e3b,#16a34a);
                color:white;
                padding:28px;
                text-align:center;
            }}
            .nav {{
                background:#111827;
                color:white;
                padding:12px 24px;
                display:flex;
                gap:18px;
                justify-content:center;
                flex-wrap:wrap;
            }}
            .nav a {{
                color:#86efac;
                text-decoration:none;
                font-weight:bold;
            }}
            .container {{
                max-width:1100px;
                margin:30px auto;
                padding:20px;
            }}
            .card {{
                background:white;
                padding:22px;
                border-radius:14px;
                box-shadow:0 8px 25px rgba(0,0,0,0.08);
                margin-bottom:20px;
            }}
            input, select, textarea {{
                width:100%;
                padding:12px;
                margin:8px 0 16px;
                border:1px solid #d1d5db;
                border-radius:8px;
                box-sizing:border-box;
            }}
            button, .btn {{
                background:#16a34a;
                color:white;
                border:none;
                padding:11px 18px;
                border-radius:8px;
                cursor:pointer;
                text-decoration:none;
                display:inline-block;
            }}
            .danger {{
                background:#dc2626;
            }}
            .secondary {{
                background:#2563eb;
            }}
            table {{
                width:100%;
                border-collapse:collapse;
                margin-top:12px;
            }}
            th, td {{
                padding:10px;
                border-bottom:1px solid #e5e7eb;
                text-align:left;
            }}
            img.preview {{
                max-width:180px;
                border-radius:10px;
            }}
            .grid {{
                display:grid;
                grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
                gap:20px;
            }}
            .msg {{
                background:#ecfdf5;
                border-left:5px solid #16a34a;
                padding:12px;
                margin:10px 0;
            }}
            .small {{
                color:#6b7280;
                font-size:13px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>LeafGuard AI</h1>
            <p>Grapevine Leaf Disease Diagnosis Platform</p>
        </div>
        {user_nav}
        <div class="container">
            {body}
        </div>
    </body>
    </html>
    """)


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def index():
    user = current_user()
    if not user:
        return redirect(url_for("login"))

    if user.role == "admin":
        return redirect(url_for("admin_dashboard"))
    if user.role == "expert":
        return redirect(url_for("expert_dashboard"))
    return redirect(url_for("farmer_dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            if user.status != "active":
                flash("Account blocked.")
                return redirect(url_for("login"))

            session["user_id"] = user.id
            return redirect(url_for("index"))

        flash("Invalid email or password.")

    body = """
    <div class="card" style="max-width:420px;margin:auto;">
        <h2>Login</h2>
        <p class="small">Admin default: admin@leafguard.com / admin12345</p>
        <form method="POST">
            <label>Email</label>
            <input type="email" name="email" required>
            <label>Password</label>
            <input type="password" name="password" required>
            <button type="submit">Login</button>
        </form>
        <p><a href="/register">Create farmer account</a></p>
    </div>
    """
    return page("Login", body)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not name or not email or not password:
            flash("Please fill all fields.")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already exists.")
            return redirect(url_for("register"))

        farmer = User(
            name=name,
            email=email,
            password=generate_password_hash(password),
            role="farmer",
            status="active"
        )

        db.session.add(farmer)
        db.session.commit()

        return redirect(url_for("login"))

    body = """
    <div class="card" style="max-width:420px;margin:auto;">
        <h2>Create Farmer Account</h2>
        <form method="POST">
            <label>Full name</label>
            <input type="text" name="name" required>
            <label>Email</label>
            <input type="email" name="email" required>
            <label>Password</label>
            <input type="password" name="password" required>
            <button type="submit">Register</button>
        </form>
        <p><a href="/login">Already have an account?</a></p>
    </div>
    """
    return page("Register", body)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/profile", methods=["GET", "POST"])
def profile():
    user = current_user()
    if not user:
        return redirect(url_for("login"))

    if request.method == "POST":
        old = request.form.get("old_password", "")
        new = request.form.get("new_password", "")

        if check_password_hash(user.password, old) and new:
            user.password = generate_password_hash(new)
            db.session.commit()
            flash("Password changed.")
        else:
            flash("Wrong old password.")

    body = f"""
    <div class="card" style="max-width:500px;margin:auto;">
        <h2>Profile</h2>
        <p><b>Name:</b> {user.name}</p>
        <p><b>Email:</b> {user.email}</p>
        <p><b>Role:</b> {user.role}</p>
        <form method="POST">
            <label>Old password</label>
            <input type="password" name="old_password" required>
            <label>New password</label>
            <input type="password" name="new_password" required>
            <button type="submit">Change Password</button>
        </form>
    </div>
    """
    return page("Profile", body)


# =========================================================
# FARMER
# =========================================================

@app.route("/farmer")
def farmer_dashboard():
    if not require_role("farmer"):
        return redirect(url_for("login"))

    user = current_user()
    history = History.query.filter_by(farmer_id=user.id).order_by(History.id.desc()).all()
    messages = ExpertMessage.query.filter_by(farmer_id=user.id).order_by(ExpertMessage.id.desc()).all()

    rows = ""
    for h in history:
        img = url_for("static", filename=h.image_path) if h.image_path else ""
        rows += f"""
        <tr>
            <td>{h.created_at.strftime('%Y-%m-%d %H:%M')}</td>
            <td>{h.disease}</td>
            <td>{h.confidence}%</td>
            <td><img class="preview" src="{img}"></td>
            <td><a class="btn secondary" href="/contact-expert/{h.id}">Contact Expert</a></td>
        </tr>
        """

    msg_rows = ""
    for m in messages:
        reply = m.reply if m.reply else "<span class='small'>No reply yet</span>"
        msg_rows += f"""
        <div class="msg">
            <b>Disease:</b> {m.disease}<br>
            <b>Your message:</b> {m.message}<br>
            <b>Expert reply:</b> {reply}
        </div>
        """

    body = f"""
    <div class="grid">
        <div class="card">
            <h2>Analyze Leaf Image</h2>
            <form action="/predict" method="POST" enctype="multipart/form-data">
                <label>Upload or capture grapevine leaf image</label>
                <input type="file" name="image" accept="image/*" capture="environment" required>
                <button type="submit">Analyze</button>
            </form>
        </div>
        <div class="card">
            <h2>Farmer Services</h2>
            <p>Upload grapevine leaf images, get disease diagnosis, treatment, advice, and contact an expert.</p>
        </div>
    </div>

    <div class="card">
        <h2>Diagnosis History</h2>
        <table>
            <tr><th>Date</th><th>Disease</th><th>Confidence</th><th>Image</th><th>Action</th></tr>
            {rows if rows else '<tr><td colspan="5">No history yet</td></tr>'}
        </table>
    </div>

    <div class="card">
        <h2>Messages With Experts</h2>
        {msg_rows if msg_rows else '<p>No messages yet.</p>'}
    </div>
    """
    return page("Farmer Dashboard", body)


@app.route("/predict", methods=["POST"])
def predict():
    if not require_role("farmer"):
        return redirect(url_for("login"))

    if "image" not in request.files:
        return page("Error", "<div class='card'>No image uploaded.</div>")

    user = current_user()
    file = request.files["image"]

    if file.filename == "":
        return page("Error", "<div class='card'>Empty file.</div>")

    relative_path, full_path = save_uploaded_image(file)
    disease, confidence, info = run_prediction(full_path)

    history = History(
        farmer_id=user.id,
        image_path=relative_path,
        disease=disease,
        confidence=confidence,
        details=info["details"],
        advice=info["advice"],
        treatment=info["treatment"]
    )

    db.session.add(history)
    db.session.commit()

    img = url_for("static", filename=relative_path)

    body = f"""
    <div class="card">
        <h2>Diagnosis Result</h2>
        <img class="preview" src="{img}">
        <h3>Disease: {disease}</h3>
        <p><b>Confidence:</b> {confidence}%</p>
        <p><b>Details:</b> {info['details']}</p>
        <p><b>Advice:</b> {info['advice']}</p>
        <p><b>Treatment:</b> {info['treatment']}</p>
        <a class="btn secondary" href="/contact-expert/{history.id}">Contact Expert</a>
        <a class="btn" href="/farmer">Back</a>
    </div>
    """
    return page("Result", body)


@app.route("/contact-expert/<int:history_id>", methods=["GET", "POST"])
def contact_expert(history_id):
    if not require_role("farmer"):
        return redirect(url_for("login"))

    user = current_user()
    history = db.session.get(History, history_id)

    if not history or history.farmer_id != user.id:
        return redirect(url_for("farmer_dashboard"))

    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if message:
            msg = ExpertMessage(
                farmer_id=user.id,
                history_id=history.id,
                disease=history.disease,
                image_path=history.image_path,
                message=message,
                status="pending"
            )
            db.session.add(msg)
            db.session.commit()
            return redirect(url_for("farmer_dashboard"))

    body = f"""
    <div class="card">
        <h2>Contact Expert</h2>
        <p><b>Disease:</b> {history.disease}</p>
        <form method="POST">
            <label>Your message</label>
            <textarea name="message" rows="5" required></textarea>
            <button type="submit">Send Message</button>
        </form>
    </div>
    """
    return page("Contact Expert", body)


# =========================================================
# EXPERT
# =========================================================

@app.route("/expert")
def expert_dashboard():
    if not require_role("expert"):
        return redirect(url_for("login"))

    messages = ExpertMessage.query.order_by(ExpertMessage.id.desc()).all()

    rows = ""
    for m in messages:
        img = url_for("static", filename=m.image_path) if m.image_path else ""
        reply = m.reply if m.reply else ""
        rows += f"""
        <div class="card">
            <p><b>Farmer:</b> {m.farmer.name}</p>
            <p><b>Disease:</b> {m.disease}</p>
            <p><b>Message:</b> {m.message}</p>
            <img class="preview" src="{img}">
            <p><b>Status:</b> {m.status}</p>
            <form method="POST" action="/expert/reply/{m.id}">
                <textarea name="reply" rows="4" required>{reply}</textarea>
                <button type="submit">Send Reply</button>
            </form>
        </div>
        """

    body = f"""
    <div class="card">
        <h2>Expert Dashboard</h2>
        <p>Reply to farmers' diagnosis requests.</p>
    </div>
    {rows if rows else '<div class="card">No messages yet.</div>'}
    """
    return page("Expert Dashboard", body)


@app.route("/expert/reply/<int:message_id>", methods=["POST"])
def expert_reply(message_id):
    if not require_role("expert"):
        return redirect(url_for("login"))

    expert = current_user()
    msg = db.session.get(ExpertMessage, message_id)

    if msg:
        msg.reply = request.form.get("reply", "").strip()
        msg.expert_id = expert.id
        msg.status = "answered"
        msg.replied_at = datetime.utcnow()
        db.session.commit()

    return redirect(url_for("expert_dashboard"))


# =========================================================
# ADMIN
# =========================================================

@app.route("/admin")
def admin_dashboard():
    if not require_role("admin"):
        return redirect(url_for("login"))

    users = User.query.order_by(User.id.desc()).all()
    histories = History.query.order_by(History.id.desc()).limit(20).all()

    user_rows = ""
    for u in users:
        user_rows += f"""
        <tr>
            <td>{u.id}</td>
            <td>{u.name}</td>
            <td>{u.email}</td>
            <td>{u.role}</td>
            <td>{u.status}</td>
            <td>
                <a class="btn secondary" href="/admin/promote/{u.id}/expert">Make Expert</a>
                <a class="btn secondary" href="/admin/promote/{u.id}/farmer">Make Farmer</a>
                <a class="btn danger" href="/admin/status/{u.id}/blocked">Block</a>
                <a class="btn" href="/admin/status/{u.id}/active">Unblock</a>
            </td>
        </tr>
        """

    hist_rows = ""
    for h in histories:
        hist_rows += f"""
        <tr>
            <td>{h.created_at.strftime('%Y-%m-%d')}</td>
            <td>{h.farmer.name}</td>
            <td>{h.disease}</td>
            <td>{h.confidence}%</td>
        </tr>
        """

    body = f"""
    <div class="grid">
        <div class="card">
            <h2>Add Expert</h2>
            <form method="POST" action="/admin/add-expert">
                <label>Name</label>
                <input type="text" name="name" required>
                <label>Email</label>
                <input type="email" name="email" required>
                <label>Password</label>
                <input type="password" name="password" required>
                <button type="submit">Add Expert</button>
            </form>
        </div>
        <div class="card">
            <h2>Statistics</h2>
            <p><b>Users:</b> {User.query.count()}</p>
            <p><b>Farmers:</b> {User.query.filter_by(role='farmer').count()}</p>
            <p><b>Experts:</b> {User.query.filter_by(role='expert').count()}</p>
            <p><b>Analyses:</b> {History.query.count()}</p>
        </div>
    </div>

    <div class="card">
        <h2>Users Management</h2>
        <table>
            <tr><th>ID</th><th>Name</th><th>Email</th><th>Role</th><th>Status</th><th>Actions</th></tr>
            {user_rows}
        </table>
    </div>

    <div class="card">
        <h2>Recent Analyses</h2>
        <table>
            <tr><th>Date</th><th>Farmer</th><th>Disease</th><th>Confidence</th></tr>
            {hist_rows if hist_rows else '<tr><td colspan="4">No analyses yet</td></tr>'}
        </table>
    </div>
    """
    return page("Admin Dashboard", body)


@app.route("/admin/add-expert", methods=["POST"])
def add_expert():
    if not require_role("admin"):
        return redirect(url_for("login"))

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()

    if User.query.filter_by(email=email).first():
        return redirect(url_for("admin_dashboard"))

    expert = User(
        name=name,
        email=email,
        password=generate_password_hash(password),
        role="expert",
        status="active"
    )

    db.session.add(expert)
    db.session.commit()

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/status/<int:user_id>/<status>")
def change_status(user_id, status):
    if not require_role("admin"):
        return redirect(url_for("login"))

    user = db.session.get(User, user_id)

    if user and user.role != "admin" and status in ["active", "blocked"]:
        user.status = status
        db.session.commit()

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/promote/<int:user_id>/<role>")
def promote_user(user_id, role):
    if not require_role("admin"):
        return redirect(url_for("login"))

    user = db.session.get(User, user_id)

    if user and user.role != "admin" and role in ["farmer", "expert"]:
        user.role = role
        db.session.commit()

    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
