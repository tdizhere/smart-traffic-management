"""
app.py — Flask + SocketIO Master Backend
Serves both YOLO and Simulation sections.
"""

import os
import sys
import threading
import time
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import random


print("[APP] Loading engines...")
sys.stdout.flush()

from sim_engine import SimEngine
from yolo_engine import YoloEngine

print("[APP] Engines imported")
sys.stdout.flush()

# ------------------------------------------------------------------ #
#  App Setup                                                           #
# ------------------------------------------------------------------ #
app = Flask(__name__)
app.config["SECRET_KEY"] = "traffic-secret-key-2024"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    ping_timeout=60, ping_interval=25, logger=False, engineio_logger=False)

SAMPLE_VIDEO_DIR = os.path.join(os.path.dirname(__file__), "sample_videos")

# Pre-found YOLO model path (from system scan)
YOLO_MODEL_PATH = r"C:\Users\tanma\smart-traffic-management\yolov8n.pt"
if not os.path.exists(YOLO_MODEL_PATH):
    # Fallback: let YoloEngine auto-discover or download
    YOLO_MODEL_PATH = None

# ------------------------------------------------------------------ #
#  Engine Instances                                                    #
# ------------------------------------------------------------------ #
sim = SimEngine(socketio)
yolo_eng = YoloEngine(socketio, model_path=YOLO_MODEL_PATH)
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov')

def get_random_sample_video():
    """Helper to check the directory and return a random video path, or fallback to default."""
    if not os.path.exists(SAMPLE_VIDEO_DIR):
        return None
        
    # Get all matching video files in the directory
    videos = [
        f for f in os.listdir(SAMPLE_VIDEO_DIR) 
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]
    
    if videos:
        # Pick a completely random video from the list
        chosen_video = random.choice(videos)
        return os.path.join(SAMPLE_VIDEO_DIR, chosen_video)
        
    # Fallback to downloading a default if the directory is empty
    fallback_path = os.path.join(SAMPLE_VIDEO_DIR, "traffic_sample.mp4")
    if not os.path.exists(fallback_path):
        _download_sample_video(fallback_path)
        
    return fallback_path if os.path.exists(fallback_path) else None
# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of annotated YOLO frames."""
    def generate():
        while True:
            with yolo_eng.lock:
                frame_bytes = yolo_eng.current_frame
            if frame_bytes:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       frame_bytes + b"\r\n")
            else:
                # Send placeholder when no video loaded
                time.sleep(0.05)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/upload_video/<int:lane_idx>", methods=["POST"])
def upload_video(lane_idx):
    """Handle video file upload for a specific lane."""
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    os.makedirs(SAMPLE_VIDEO_DIR, exist_ok=True)
    save_path = os.path.join(SAMPLE_VIDEO_DIR, f"lane_{lane_idx}_uploaded.mp4")
    f.save(save_path)
    yolo_eng.set_lane_video(lane_idx, save_path)
    return jsonify({"status": "ok", "lane": lane_idx})


@app.route("/use_sample/<int:lane_idx>")
def use_sample(lane_idx):
    """Use a random sample video for a specific lane."""
    sample_path = get_random_sample_video()
    
    if sample_path:
        yolo_eng.set_lane_video(lane_idx, sample_path)
        # Included video filename in response for better front-end debugging
        return jsonify({
            "status": "ok", 
            "lane": lane_idx, 
            "file_mounted": os.path.basename(sample_path)
        })
        
    return jsonify({"error": "No sample videos available. Please upload a video."}), 404


@app.route("/use_sample_all")
def use_sample_all():
    """Use a unique random sample video for each active lane."""
    # Ensure there are lanes configured to process
    if yolo_eng.n_lanes <= 0:
        return jsonify({"error": "No lanes initialized."}), 400

    for i in range(yolo_eng.n_lanes):
        # Calling this inside the loop ensures each lane gets its own unique random selection!
        sample_path = get_random_sample_video()
        if sample_path:
            yolo_eng.set_lane_video(i, sample_path)
        else:
            return jsonify({"error": "Sample video asset processing failed during layout initialization."}), 404
            
    return jsonify({"status": "ok"})

@app.route("/clear_lane/<int:lane_idx>")
def clear_lane(lane_idx):
    """Clear video feed for a specific lane."""
    yolo_eng.clear_lane_video(lane_idx)
    return jsonify({"status": "ok", "lane": lane_idx})


def _download_sample_video(save_path):
    """Download a small public traffic video for demo."""
    try:
        import requests as req_lib
        url = "https://www.pexels.com/download/video/855282/"  # small royalty-free traffic video
        headers = {"User-Agent": "Mozilla/5.0"}
        r = req_lib.get(url, headers=headers, timeout=15, stream=True)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[APP] Sample video downloaded to {save_path}")
    except Exception as e:
        print(f"[APP] Sample video download failed: {e}")


@app.route("/set_quads/<int:n>")
def set_quads(n):
    yolo_eng.set_n_lanes(n)
    return jsonify({"status": "ok", "n_lanes": n})


# ------------------------------------------------------------------ #
#  SocketIO Events                                                     #
# ------------------------------------------------------------------ #
@socketio.on("connect")
def on_connect():
    print(f"[SOCKET] Client connected: {request.sid}")


@socketio.on("disconnect")
def on_disconnect():
    print(f"[SOCKET] Client disconnected: {request.sid}")


@socketio.on("toggle_emergency")
def on_toggle_emergency(data=None):
    """Simulation section emergency corridor trigger."""
    lane = None
    if isinstance(data, dict):
        lane = data.get("lane")
    elif isinstance(data, str):
        lane = data
    sim.trigger_emergency(lane=lane)
    emit("sim_state", sim.tick(), broadcast=True)


@socketio.on("cancel_emergency")
def on_cancel_emergency():
    """Cancel simulation emergency corridor override."""
    sim.cancel_emergency()
    emit("sim_state", sim.tick(), broadcast=True)


@socketio.on("yolo_set_quads")
def on_yolo_set_quads(data):
    n = data.get("n", 4)
    yolo_eng.set_n_lanes(n)


@socketio.on("yolo_emergency_override")
def on_yolo_emergency():
    """Force emergency mode on YOLO engine."""
    yolo_eng.emergency = True
    yolo_eng._cycle_timer = 15


# ------------------------------------------------------------------ #
#  Background Threads                                                  #
# ------------------------------------------------------------------ #
def start_sim_loop():
    time.sleep(1)
    sim.run_loop()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 60)
    print("  AI Traffic Management System")
    print("  http://localhost:5000")
    print("=" * 60)
    sys.stdout.flush()

    # Start simulation loop
    sim_thread = threading.Thread(target=start_sim_loop, daemon=True)
    sim_thread.start()

    # Start YOLO processing loop immediately
    yolo_eng.start()

    print("[APP] Starting Flask server on http://0.0.0.0:5000")
    sys.stdout.flush()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True,
                 use_reloader=True, allow_unsafe_werkzeug=True)
