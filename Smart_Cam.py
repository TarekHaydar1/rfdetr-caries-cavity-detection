"""
Dental Camera AI - Simplified
Small GUI to stream camera to virtual camera with optional AI overlay
"""

import cv2
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
from pathlib import Path

CONFIG_FILE = Path.home() / '.dental_camera.json'

# ============================================================================
# HELPERS
# ============================================================================

def load_settings():
    """Load saved settings."""
    try:
        return json.load(open(CONFIG_FILE))
    except:
        return {'camera': 0, 'model': '', 'use_ai': False}

def save_settings(camera, model, use_ai):
    """Save settings."""
    json.dump({'camera': camera, 'model': model, 'use_ai': use_ai}, 
              open(CONFIG_FILE, 'w'))

def find_cameras():
    """Find all cameras."""
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened() and cap.read()[0]:
            cameras.append(i)
        cap.release()
    return cameras

# ============================================================================
# AI MODEL
# ============================================================================

class AIDetector:
    """Simple AI model wrapper."""
    
    def __init__(self, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(model_path, map_location=device)
        self.model = checkpoint.get('model', checkpoint)
        self.model.to(device).eval()
        self.device = device
    
    def process(self, frame):
        """Detect and draw boxes."""
        # Prepare image
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        
        # Detect
        with torch.no_grad():
            out = self.model(img)
        
        # Draw (adjust to your model output format)
        if isinstance(out, dict):
            boxes = out.get('pred_boxes', [[]])[0]
            scores = out.get('pred_logits', [[]])[0].softmax(-1).max(-1)[0]
            
            h, w = frame.shape[:2]
            for box, score in zip(boxes, scores):
                if score > 0.5:
                    cx, cy, bw, bh = box.cpu().numpy()
                    x1, y1 = int((cx - bw/2) * w), int((cy - bh/2) * h)
                    x2, y2 = int((cx + bw/2) * w), int((cy + bh/2) * h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{score:.2f}', (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

# ============================================================================
# CAMERA STREAM
# ============================================================================

class CameraStream:
    """Handles camera → virtual camera streaming."""
    
    def __init__(self, camera_idx, ai_model=None):
        self.running = False
        self.ai = ai_model
        
        # Open camera
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise Exception("Camera not available")
        
        # Get resolution
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 30)
        
        # Setup virtual camera
        import pyvirtualcam
        self.vcam = pyvirtualcam.Camera(w, h, fps, fmt=pyvirtualcam.PixelFormat.BGR)
    
    def start(self):
        """Start streaming."""
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
    
    def _loop(self):
        """Main loop: read → process → send."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Apply AI if enabled
                if self.ai:
                    frame = self.ai.process(frame)
                # Send to virtual camera
                self.vcam.send(frame)
    
    def stop(self):
        """Stop streaming."""
        self.running = False
        self.cap.release()

# ============================================================================
# GUI
# ============================================================================

class App:
    def __init__(self):
        # Window
        self.win = tk.Tk()
        self.win.title("Dental Camera AI")
        self.win.geometry("380x280")
        self.win.resizable(False, False)
        
        # Load settings
        settings = load_settings()
        self.cameras = find_cameras()
        self.stream = None
        
        # Variables
        self.cam_var = tk.StringVar()
        self.ai_var = tk.BooleanVar(value=settings['use_ai'])
        self.model_var = tk.StringVar(value=settings['model'])
        
        # Build UI
        self._build_ui()
        
        # Populate cameras
        if self.cameras:
            self.cam_combo['values'] = [f"Camera {i}" for i in self.cameras]
            idx = self.cameras.index(settings['camera']) if settings['camera'] in self.cameras else 0
            self.cam_combo.current(idx)
        else:
            self.cam_combo['values'] = ["No cameras"]
            self.cam_combo.current(0)
    
    def _build_ui(self):
        """Build interface."""
        pad = {'padx': 15, 'pady': 8}
        
        # Title
        ttk.Label(self.win, text="🦷 Dental Camera AI", 
                 font=("Arial", 14, "bold")).pack(pady=15)
        
        # Camera selection
        ttk.Label(self.win, text="Camera:").pack(anchor='w', **pad)
        self.cam_combo = ttk.Combobox(self.win, textvariable=self.cam_var, 
                                      state="readonly", width=40)
        self.cam_combo.pack(**pad)
        
        # AI toggle
        ttk.Checkbutton(self.win, text="Enable AI Detection", 
                       variable=self.ai_var).pack(anchor='w', **pad)
        
        # Model path
        model_frame = ttk.Frame(self.win)
        model_frame.pack(fill='x', **pad)
        ttk.Entry(model_frame, textvariable=self.model_var, 
                 state="readonly", width=28).pack(side='left', padx=(15, 5))
        ttk.Button(model_frame, text="Browse", 
                  command=self._browse, width=8).pack(side='left')
        
        # Status
        self.status = ttk.Label(self.win, text="⚪ Stopped", font=("Arial", 10))
        self.status.pack(pady=10)
        
        # Start button
        self.btn = ttk.Button(self.win, text="▶ Start", 
                             command=self._toggle, width=20)
        self.btn.pack(pady=10)
        
        self.win.protocol("WM_DELETE_WINDOW", self._close)
    
    def _browse(self):
        """Browse for model."""
        path = filedialog.askopenfilename(
            filetypes=[("Model", "*.pt *.pth"), ("All", "*.*")])
        if path:
            self.model_var.set(path)
    
    def _toggle(self):
        """Start/Stop stream."""
        if not self.stream:
            self._start()
        else:
            self._stop()
    
    def _start(self):
        """Start virtual camera."""
        try:
            # Get camera index
            if not self.cameras:
                messagebox.showerror("Error", "No cameras found")
                return
            cam_idx = self.cameras[self.cam_combo.current()]
            
            # Load AI model if enabled
            ai_model = None
            if self.ai_var.get():
                model_path = self.model_var.get()
                if not model_path:
                    messagebox.showerror("Error", "Select model file")
                    return
                self.status.config(text="🟡 Loading AI...")
                self.win.update()
                ai_model = AIDetector(model_path)
            
            # Start stream
            self.status.config(text="🟡 Starting...")
            self.win.update()
            self.stream = CameraStream(cam_idx, ai_model)
            self.stream.start()
            
            # Update UI
            self.btn.config(text="⏹ Stop")
            self.status.config(text="🟢 Active")
            
            # Save settings
            save_settings(cam_idx, self.model_var.get(), self.ai_var.get())
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text="⚪ Stopped")
            self.stream = None
    
    def _stop(self):
        """Stop stream."""
        if self.stream:
            self.stream.stop()
            self.stream = None
        self.btn.config(text="▶ Start")
        self.status.config(text="⚪ Stopped")
    
    def _close(self):
        """Clean exit."""
        self._stop()
        self.win.destroy()
    
    def run(self):
        """Run app."""
        self.win.mainloop()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    App().run()