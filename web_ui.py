import os
import io
import base64
import sys
import subprocess
from datetime import datetime

from flask import Flask, request, redirect, url_for, render_template_string, flash
from PIL import Image
import cv2

app = Flask(__name__)
app.secret_key = "face-secret-key"

ROOT_DIR = os.path.dirname(__file__)
KNOWN_FACES_DIR = os.path.join(ROOT_DIR, "known_faces")
FASTER_SCRIPT = os.path.join(ROOT_DIR, "facerec_from_webcam_faster.py")

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Face Access</title>
  <style>
    body { font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #0f172a; color: #e5e7eb; }
    .container { max-width: 760px; margin: 0 auto; padding: 40px 24px; }
    h1 { font-size: 28px; margin-bottom: 24px; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 24px; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 20px; }
    .btn { display: inline-block; padding: 14px 16px; border-radius: 10px; text-decoration: none; text-align: center; font-weight: 600; }
    .btn-primary { background: #2563eb; color: white; }
    .btn-secondary { background: #374151; color: #e5e7eb; }
    .notice { color: #93c5fd; font-size: 14px; margin-top: 10px; }
    .flash { background: #1f2937; border: 1px solid #374151; padding: 12px; border-radius: 8px; margin-bottom: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Face Access</h1>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for msg in messages %}
          <div class="flash">{{ msg }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}
    <div class="card">
      <p>Choose an action:</p>
      <div class="actions">
        <a class="btn btn-primary" href="{{ url_for('register_capture') }}">Register New User</a>
        <a class="btn btn-secondary" href="{{ url_for('login_start') }}">Login with Facial Recognition</a>
      </div>
      <p class="notice">Registration captures a snapshot from your local webcam, then lets you save it to <code>known_faces</code>.</p>
    </div>
  </div>
</body>
</html>
"""

REGISTER_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Register User</title>
  <style>
    body { font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #0f172a; color: #e5e7eb; }
    .container { max-width: 860px; margin: 0 auto; padding: 40px 24px; }
    h1 { font-size: 26px; margin-bottom: 18px; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 24px; }
    .preview { display: flex; gap: 16px; align-items: flex-start; }
    img { max-width: 480px; border-radius: 12px; border: 1px solid #374151; }
    .form { flex: 1; }
    input[type=text] { width: 100%; padding: 12px; border-radius: 10px; border: 1px solid #374151; background: #0b1220; color: #e5e7eb; }
    .actions { margin-top: 14px; display: flex; gap: 10px; }
    .btn { padding: 12px 14px; border-radius: 10px; text-decoration: none; border: none; font-weight: 600; cursor: pointer; }
    .btn-primary { background: #2563eb; color: white; }
    .btn-secondary { background: #374151; color: #e5e7eb; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Register New User</h1>
    <div class="card">
      <div class="preview">
        <img src="data:image/jpeg;base64,{{ img_b64 }}" alt="Captured frame" />
        <form class="form" method="post" action="{{ url_for('register_save') }}">
          <label for="username">Username</label>
          <input type="text" id="username" name="username" placeholder="e.g. alice" required />
          <input type="hidden" name="snapshot_id" value="{{ snapshot_id }}" />
          <div class="actions">
            <a class="btn btn-secondary" href="{{ url_for('register_capture') }}">Retake</a>
            <button class="btn btn-primary" type="submit">Save</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</body>
</html>
"""

# Simple in-memory store for latest snapshot (id -> bytes)
SNAPSHOTS = {}


def _capture_frame() -> Image.Image:
    # Try camera 0, fallback to 1
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        cap = cv2.VideoCapture(1)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Unable to access webcam or read a frame.")
    # Convert BGR to RGB and return PIL Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return img


def _image_to_base64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/register")
def register_capture():
    try:
        img = _capture_frame()
        img_b64 = _image_to_base64_jpeg(img)
        snapshot_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        SNAPSHOTS[snapshot_id] = img
        return render_template_string(REGISTER_HTML, img_b64=img_b64, snapshot_id=snapshot_id)
    except Exception as e:
        flash(f"Capture failed: {e}")
        return redirect(url_for("index"))


@app.route("/register", methods=["POST"])
def register_save():
    username = request.form.get("username", "").strip()
    snapshot_id = request.form.get("snapshot_id", "")
    if not username:
        flash("Please enter a username.")
        return redirect(url_for("register_capture"))
    img = SNAPSHOTS.get(snapshot_id)
    if img is None:
        flash("Snapshot not found. Please retake.")
        return redirect(url_for("register_capture"))
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    safe_name = "".join(c for c in username if (c.isalnum() or c in ("-", "_")))
    if not safe_name:
        flash("Invalid username: use letters, digits, '-' or '_'.")
        return redirect(url_for("register_capture"))
    out_path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}.jpg")
    try:
        img.save(out_path, format="JPEG", quality=95)
        # Cleanup snapshot
        SNAPSHOTS.pop(snapshot_id, None)
        flash(f"User '{safe_name}' registered.")
    except Exception as e:
        flash(f"Save failed: {e}")
    return redirect(url_for("index"))


@app.route("/login")
def login_start():
    if not os.path.exists(FASTER_SCRIPT):
        flash("Face recognition script not found.")
        return redirect(url_for("index"))
    try:
        subprocess.Popen([sys.executable, FASTER_SCRIPT])
        flash("Facial recognition started in a separate window.")
    except Exception as e:
        flash(f"Unable to start recognition: {e}")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Run Flask app
    app.run(host="127.0.0.1", port=5000, debug=True)
