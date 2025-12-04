import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import face_recognition

KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")
FASTER_SCRIPT = os.path.join(os.path.dirname(__file__), "facerec_from_webcam_faster.py")


class FaceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Access")
        self.geometry("980x600")
        self.resizable(False, False)

        # State
        self.mode = "idle"  # idle | register | login
        self.camera_index = tk.IntVar(value=0)
        self.cap = None
        self.current_frame_bgr = None
        self.preview_imgtk = None
        self.process_this_frame = True  # skip every other frame for speed
        self.last_face_locations = []
        self.last_face_names = []

        # Cache known faces (loaded once)
        self.known_face_encodings = []
        self.known_face_names = []
        self._load_known_faces()

        # Main layout: left controls, right preview
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root, width=280)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Left column
        ttk.Label(left, text="Face Access", font=("SF Pro", 18, "bold")).pack(
            pady=(0, 16)
        )

        # Camera selector
        cam_row = ttk.Frame(left)
        cam_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(cam_row, text="Camera:").pack(side=tk.LEFT)
        cam_choices = [0, 1, 2, 3]
        cam_menu = ttk.OptionMenu(
            cam_row,
            self.camera_index,
            self.camera_index.get(),
            *cam_choices,
            command=lambda _: self.switch_camera(),
        )
        cam_menu.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Action buttons
        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Open Camera", command=self.switch_camera).pack(
            fill=tk.X, pady=6
        )
        ttk.Button(btn_row, text="Register User", command=self.start_register).pack(
            fill=tk.X, pady=6
        )
        ttk.Button(btn_row, text="Recognize User", command=self.start_login).pack(
            fill=tk.X, pady=6
        )

        # Status label (fixed width to avoid layout jumping)
        self.status = ttk.Label(
            left,
            text="Ready",
            foreground="#555",
            wraplength=240,
            anchor="w",
            justify="left",
        )
        self.status.pack(fill=tk.X, pady=(16, 0))

        # Right preview canvas + register panel (shown only in register mode)
        preview_area = ttk.Frame(right)
        preview_area.pack(fill=tk.BOTH, expand=True)

        self.preview_width = 680
        self.preview_height = 510
        self.preview_canvas = tk.Canvas(
            preview_area,
            width=self.preview_width,
            height=self.preview_height,
            bg="#111",
        )
        self.preview_canvas.pack(fill=tk.NONE, expand=False, padx=6, pady=(6, 0))

        # Register controls under webcam (username + capture button)
        self.register_panel = ttk.Frame(preview_area)
        self.register_panel.pack(fill=tk.X, padx=6, pady=(8, 6))

        self.name_var = tk.StringVar()
        ttk.Label(self.register_panel, text="Username").pack(side=tk.LEFT)
        self.entry_name = ttk.Entry(self.register_panel, textvariable=self.name_var)
        self.entry_name.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
        ttk.Button(
            self.register_panel, text="Capture & Save", command=self.save_image
        ).pack(side=tk.LEFT)

        # Hidden by default (only in register mode)
        self.register_panel.pack_forget()

        # Ensure known faces directory exists
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    def set_status(self, text: str):
        # Keep status short to avoid growing the sidebar
        if len(text) > 60:
            text = text[:57] + "..."
        self.status.config(text=text)
        self.update_idletasks()

    def open_register_window(self):
        # Deprecated in single-window UI
        pass

    def start_login(self):
        self.mode = "login"
        self.set_status("Recognition mode")
        # Hide register-only panel
        if hasattr(self, "register_panel"):
            self.register_panel.pack_forget()
        if self.cap is None:
            self.switch_camera()

    def switch_camera(self):
        # Open/reopen camera with selected index
        idx = int(self.camera_index.get())
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(idx)
        ok, _ = self.cap.read()
        if not ok:
            self.set_status(f"Camera {idx} failed")
        else:
            self.set_status(f"Camera {idx} opened")
        # Start preview loop
        self.after(50, self.update_preview_loop)

    def start_register(self):
        self.mode = "register"
        self.set_status("Register mode")
        # Show register-only panel
        if hasattr(self, "register_panel"):
            self.register_panel.pack(fill=tk.X, padx=6, pady=(8, 6))
        if self.cap is None:
            self.switch_camera()

    def update_preview_loop(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.current_frame_bgr = frame
            # For login mode, overlay recognition results
            display_bgr = frame
            if self.mode == "login":
                # Only process every other frame for speed
                if self.process_this_frame:
                    display_bgr = frame.copy()
                    self.overlay_recognition(display_bgr)
                else:
                    # Draw last known boxes to avoid blinking
                    self.draw_boxes(
                        display_bgr, self.last_face_locations, self.last_face_names
                    )
                self.process_this_frame = not self.process_this_frame

            # Convert to PIL Image
            rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            # Maintain aspect ratio in canvas
            img_w, img_h = img.size
            canvas_w, canvas_h = self.preview_width, self.preview_height
            scale = min(canvas_w / img_w, canvas_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)

            bg = Image.new("RGB", (canvas_w, canvas_h), (17, 17, 17))
            offset_x = (canvas_w - new_w) // 2
            offset_y = (canvas_h - new_h) // 2
            bg.paste(img_resized, (offset_x, offset_y))

            self.preview_imgtk = ImageTk.PhotoImage(bg)
            self.preview_canvas.create_image(
                0, 0, image=self.preview_imgtk, anchor=tk.NW
            )
        # Schedule next update
        self.after(60, self.update_preview_loop)

    def save_image(self):
        # Only relevant in register mode
        if self.mode != "register":
            self.set_status("Switch to Register mode to save")
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Missing Name", "Please enter a username.")
            return
        if self.current_frame_bgr is None:
            messagebox.showwarning("No Capture", "No frame available to save.")
            return
        rgb = cv2.cvtColor(self.current_frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        safe_name = "".join(c for c in name if (c.isalnum() or c in ("-", "_")))
        if not safe_name:
            messagebox.showwarning(
                "Invalid Name", "Username contains no valid characters."
            )
            return
        out_path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}.jpg")
        try:
            img.save(out_path, format="JPEG", quality=95)
            # Only show the username in the status to keep it compact
            self.set_status(f"User saved: {safe_name}")
            messagebox.showinfo("Saved", f"User registered: {safe_name}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save image: {e}")
            self.set_status("Save failed")

    def overlay_recognition(self, frame_bgr):
        # Inline recognition using cached known faces; best-effort
        try:
            # Downscale for speed
            small = cv2.resize(frame_bgr, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb_small = np.ascontiguousarray(rgb_small)

            # Use HOG model (faster) explicitly; set upsample to 1 for speed
            face_locations = face_recognition.face_locations(
                rgb_small, number_of_times_to_upsample=0, model="hog"
            )
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            names = []
            for fe in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, fe)
                name = "Unknown"
                if self.known_face_encodings:
                    dists = face_recognition.face_distance(
                        self.known_face_encodings, fe
                    )
                    best = int(np.argmin(dists))
                    if matches and matches[best]:
                        name = self.known_face_names[best]
                names.append(name)

            # Scale locations to original and store for reuse
            scaled_locations = []
            for top, right, bottom, left in face_locations:
                scaled_locations.append((top * 4, right * 4, bottom * 4, left * 4))
            self.last_face_locations = scaled_locations
            self.last_face_names = names
            # Draw boxes now
            self.draw_boxes(frame_bgr, self.last_face_locations, self.last_face_names)
        except Exception:
            pass

    def draw_boxes(self, frame_bgr, locations, names):
        try:
            for (top, right, bottom, left), name in zip(locations, names):
                color = (0, 150, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)
                cv2.rectangle(
                    frame_bgr, (left, bottom - 35), (right, bottom), color, cv2.FILLED
                )
                cv2.putText(
                    frame_bgr,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (255, 255, 255),
                    1,
                )
        except Exception:
            pass

    def _load_known_faces(self):
        try:
            self.known_face_encodings.clear()
            self.known_face_names.clear()
            for filename in os.listdir(KNOWN_FACES_DIR):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    img = face_recognition.load_image_file(
                        os.path.join(KNOWN_FACES_DIR, filename)
                    )
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        self.known_face_encodings.append(encs[0])
                        self.known_face_names.append(os.path.splitext(filename)[0])
        except Exception:
            # Ignore load errors
            pass


if __name__ == "__main__":
    app = FaceApp()
    app.mainloop()
