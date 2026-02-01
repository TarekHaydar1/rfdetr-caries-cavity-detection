import cv2
import torch
import numpy as np
import threading
import time
import argparse
import logging
from queue import Queue

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

#==================

class Camera:
    
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        
    def start(self):
        self.running = True
        threading.Thread(target=self._capture, daemon=True).start()
        
    def _capture(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    
    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
            
    def stop(self):
        self.running = False
        self.cap.release()


# ============================================================================
# AI INFERENCE
# ============================================================================

class AIModel:
    """RF-DETR model wrapper."""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = checkpoint.get('model', checkpoint)
            self.model.to(self.device).eval()
            logger.info("Model loaded")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.model = None
            
    def infer(self, frame):
        """Run inference and draw boxes."""
        if self.model is None:
            return frame
            
        try:
            # Preprocess
            img = cv2.resize(frame, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img = img.unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                pred = self.model(img)
            
            # Postprocess - ADAPT THIS TO YOUR MODEL OUTPUT FORMAT
            if isinstance(pred, dict):
                boxes = pred.get('pred_boxes', [[]])[0]
                scores = pred.get('pred_logits', [[]])[0].softmax(-1).max(-1)[0]
            else:
                return frame  # Unknown format, return original
                
            # Draw boxes
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
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return frame


# ============================================================================
# VIRTUAL CAMERA
# ============================================================================

class VirtualCam:
    """Virtual camera output."""
    
    def __init__(self, width, height, fps):
        try:
            import pyvirtualcam
            self.cam = pyvirtualcam.Camera(width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR)
            logger.info(f"Virtual camera: {self.cam.device}")
        except Exception as e:
            logger.warning(f"Virtual camera unavailable: {e}")
            self.cam = None
            
    def send(self, frame):
        if self.cam:
            self.cam.send(frame)


# ============================================================================
# BUTTON DETECTION
# ============================================================================

class ButtonListener:
    """HID button detection."""
    
    def __init__(self, callback):
        self.callback = callback
        self.running = False
        
        try:
            import hid
            self.hid = hid
            self.available = True
        except ImportError:
            logger.warning("HID library not available - no button detection")
            self.available = False
            
    def start(self):
        if not self.available:
            return
            
        self.running = True
        threading.Thread(target=self._listen, daemon=True).start()
        
    def _listen(self):
        devices = []
        
        # Open all HID devices
        for dev_info in self.hid.enumerate():
            try:
                dev = self.hid.device()
                dev.open_path(dev_info['path'])
                dev.set_nonblocking(True)
                devices.append((dev, dev_info))
                logger.info(f"Monitoring: {dev_info.get('product_string', 'Unknown')}")
            except:
                pass
        
        last_states = {}
        
        while self.running:
            for dev, dev_info in devices:
                try:
                    data = dev.read(64)
                    if data:
                        path = dev_info['path']
                        # Detect any change in data as button press
                        if path not in last_states or last_states[path] != data:
                            last_states[path] = data
                            # Any non-zero byte = button press
                            if any(d != 0 for d in data):
                                logger.info(f"Button pressed: {bytes(data).hex()}")
                                self.callback()
                except:
                    pass
            time.sleep(0.01)
            
    def stop(self):
        self.running = False


# ============================================================================
# KEYSTROKE SENDER
# ============================================================================

class KeystrokeSender:
    """Send keystrokes."""
    
    def __init__(self):
        try:
            from pynput.keyboard import Controller, Key
            self.kb = Controller()
            self.Key = Key
            self.available = True
            logger.info("Keystroke sender ready")
        except ImportError:
            logger.warning("pynput not available - no keystroke sending")
            self.available = False
            
    def send_space(self):
        """Send space key (capture)."""
        if self.available:
            self.kb.press(self.Key.space)
            time.sleep(0.01)
            self.kb.release(self.Key.space)
            logger.info("Sent: SPACE")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dental Camera AI')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--model', type=str, default=None, help='Model path')
    args = parser.parse_args()
    
    logger.info("Starting Dental Camera AI...")
    
    # Initialize components
    camera = Camera(args.camera)
    camera.start()
    time.sleep(1)  # Wait for camera
    
    # Get frame properties
    test_frame = camera.read()
    if test_frame is None:
        logger.error("Camera not working")
        return
    h, w = test_frame.shape[:2]
    
    # AI model (optional)
    ai_model = AIModel(args.model) if args.model else None
    
    # Virtual camera
    vcam = VirtualCam(w, h, 30)
    
    # Keystroke sender
    keystroke = KeystrokeSender()
    
    # Button listener
    def on_button():
        keystroke.send_space()
    
    button_listener = ButtonListener(on_button)
    button_listener.start()
    
    logger.info(f"Running on camera {args.camera} ({w}x{h})")
    logger.info("Press Ctrl+C to stop")
    
    # Main loop
    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Apply AI
            if ai_model:
                frame = ai_model.infer(frame)
            
            # Send to virtual camera
            vcam.send(frame)
            
            # Show preview (optional)
            cv2.imshow('Dental Camera AI', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Stopping...")
    
    # Cleanup
    camera.stop()
    button_listener.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
