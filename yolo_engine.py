"""
yolo_engine.py — Per-Lane YOLO Video Processing Engine

Each lane has its OWN independent video file.
YOLO runs on each lane's current frame separately.
All annotated frames are stitched into a grid and streamed as MJPEG.
No quadrant splitting — no violations.
"""

import cv2
import numpy as np
import threading
import time
import os
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# COCO vehicle classes
VEHICLE_CLASSES = {
    2: ("car",        1.0),
    3: ("motorcycle", 0.8),
    5: ("bus",        2.5),
    7: ("truck",      1.5),
}

# Per-lane accent colors (BGR)
LANE_COLORS = [
    (255, 140,  60),   # Lane 0 — orange
    ( 60, 210, 255),   # Lane 1 — cyan
    ( 60, 220, 100),   # Lane 2 — green
    (200,  80, 255),   # Lane 3 — purple
]

LANE_DEFAULT_NAMES = ["North", "East", "South", "West"]

# Target frame size per lane tile in the output grid
TILE_W = 450
TILE_H = 300


def compute_weight(counts):
    return round(
        counts.get("car", 0)        * 1.0 +
        counts.get("motorcycle", 0) * 0.8 +
        counts.get("truck", 0)      * 1.5 +
        counts.get("bus", 0)        * 2.5,
        1
    )


def compute_green_time(weight):
    return max(6, min(60, int(weight * 1.2)))


def detect_emergency_color(frame):
    """Detect flashing red/blue (emergency lights) via HSV color masking."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_r1 = cv2.inRange(hsv, np.array([0,   150, 150]), np.array([ 10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([160, 150, 150]), np.array([180, 255, 255]))
    mask_b  = cv2.inRange(hsv, np.array([100, 150, 100]), np.array([130, 255, 255]))
    combined = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), mask_b)
    return cv2.countNonZero(combined) > 800


class LaneCapture:
    """Holds state for a single lane's video feed."""
    __slots__ = ["cap", "path", "name", "loaded", "frame", "counts",
                 "weight", "green_time", "light", "emergency", "lock"]

    def __init__(self, name):
        self.cap        = None
        self.path       = None
        self.name       = name
        self.loaded     = False
        self.frame      = None          # latest annotated BGR frame
        self.counts     = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0}
        self.weight     = 0.0
        self.green_time = 6
        self.light      = "red"
        self.emergency  = False
        self.lock       = threading.Lock()

    def set_video(self, path):
        with self.lock:
            if self.cap:
                self.cap.release()
            self.cap    = cv2.VideoCapture(path)
            self.path   = path
            self.loaded = self.cap.isOpened()

    def read_frame(self):
        """Read next frame, loop video at end. Returns BGR frame or None."""
        if not self.cap or not self.cap.isOpened():
            return None
        with self.lock:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()

    def clear_video(self):
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.path = None
            self.loaded = False
            self.frame = None
            self.counts = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0}
            self.weight = 0.0
            self.green_time = 6
            self.light = "red"
            self.emergency = False


class YoloEngine:
    def __init__(self, socketio, model_path=None, n_lanes=4):
        self.socketio   = socketio
        self.model      = None
        self.model_path = model_path
        self.model_ready = False

        self.n_lanes = n_lanes
        self.lanes   = [LaneCapture(LANE_DEFAULT_NAMES[i]) for i in range(4)]

        self.running       = False
        self.lock          = threading.Lock()
        self.current_frame = None      # stitched JPEG bytes for MJPEG
        self.active_lane   = 0
        self._cycle_timer  = 10
        self.emergency     = False
        self.frame_count   = 0

        # YOLO Frame skip cache
        self.cached_boxes  = [[] for _ in range(4)]
        self.cached_counts = [{"car": 0, "motorcycle": 0, "truck": 0, "bus": 0} for _ in range(4)]
        self.cached_emerg  = [False for _ in range(4)]

        # Start async model load
        threading.Thread(target=self._load_model_bg, daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Model Loading (non-blocking)                                         #
    # ------------------------------------------------------------------ #
    def _load_model_bg(self):
        if not YOLO_AVAILABLE:
            print("[YOLO] ultralytics not available")
            return

        candidates = []
        if self.model_path and os.path.exists(self.model_path):
            candidates.append(self.model_path)

        for p in [
            Path(r"C:\Users\tanma\smart-traffic-management\yolov8n.pt"),
            Path(r"C:\Users\tanma\Desktop\FINALPROJECT (1)\yolov8n.pt"),
            Path.home() / ".config" / "Ultralytics" / "yolov8n.pt",
        ]:
            if p.exists():
                candidates.append(str(p))

        for path in candidates:
            try:
                print(f"[YOLO] Loading model: {path}")
                self.model = YOLO(path)
                self.model_ready = True
                print("[YOLO] ✓ Model ready")
                return
            except Exception as e:
                print(f"[YOLO] Load failed ({path}): {e}")

        try:
            print("[YOLO] Downloading yolov8n.pt ...")
            self.model = YOLO("yolov8n.pt")
            self.model_ready = True
            print("[YOLO] ✓ Downloaded and ready")
        except Exception as e:
            print(f"[YOLO] Could not load model: {e}")

    # ------------------------------------------------------------------ #
    #  Lane Video Management                                               #
    # ------------------------------------------------------------------ #
    def set_lane_video(self, lane_idx, path):
        if 0 <= lane_idx < 4:
            self.lanes[lane_idx].set_video(path)
            print(f"[YOLO] Lane {lane_idx} ({self.lanes[lane_idx].name}) → {path}")
            if not self.running:
                self.start()

    def clear_lane_video(self, lane_idx):
        if 0 <= lane_idx < 4:
            self.lanes[lane_idx].clear_video()
            self.cached_boxes[lane_idx] = []
            self.cached_counts[lane_idx] = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0}
            self.cached_emerg[lane_idx] = False
            print(f"[YOLO] Lane {lane_idx} video cleared")

    def set_n_lanes(self, n):
        self.n_lanes = max(2, min(4, int(n)))
        for i in range(4):
            self.lanes[i].name = LANE_DEFAULT_NAMES[i]

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.run_loop, daemon=True).start()

    def stop(self):
        self.running = False
        for lane in self.lanes:
            lane.release()

    # ------------------------------------------------------------------ #
    #  Main Processing Loop                                                #
    # ------------------------------------------------------------------ #
    def run_loop(self):
        while self.running:
            self.frame_count += 1
            lane_frames = []
            any_active = False

            for i in range(self.n_lanes):
                lane = self.lanes[i]
                raw = lane.read_frame()

                if raw is None:
                    # Lane has no video yet — draw placeholder tile
                    tile = self._placeholder_tile(i)
                    lane.counts = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0}
                    lane.weight = 0.0
                    lane.green_time = 6
                    lane.emergency = False
                    self.cached_boxes[i] = []
                    self.cached_counts[i] = lane.counts.copy()
                    self.cached_emerg[i] = False
                else:
                    any_active = True
                    # Resize for speed
                    raw = cv2.resize(raw, (TILE_W, TILE_H))
                    # Run YOLO every 3 frames per lane, or if cache is empty
                    if self.frame_count % 10 == 0 or not self.cached_counts[i] or (lane.frame is None):
                        counts, boxes, emerg = self._detect(raw)
                        lane.counts    = counts
                        lane.weight    = compute_weight(counts)
                        lane.green_time = compute_green_time(lane.weight)
                        lane.emergency = emerg
                        # Cache
                        self.cached_boxes[i]  = boxes
                        self.cached_counts[i] = counts
                        self.cached_emerg[i]  = emerg
                    else:
                        # Use cache
                        counts = self.cached_counts[i]
                        boxes  = self.cached_boxes[i]
                        emerg  = self.cached_emerg[i]
                        lane.counts = counts
                        lane.weight = compute_weight(counts)
                        lane.green_time = compute_green_time(lane.weight)
                        lane.emergency = emerg

                    # Annotate tile
                    tile = self._annotate_tile(raw, i, counts, boxes, emerg)
                    lane.frame = tile

                lane_frames.append(tile)

            # Update cycle (pick active lane)
            self._update_cycle()

            # Apply light states on tiles (overlay border glow)
            annotated_tiles = []
            for i, tile in enumerate(lane_frames):
                light = "green" if i == self.active_lane else "red"
                self.lanes[i].light = light
                annotated_tiles.append(self._apply_light_border(tile, i, light))

            # Stitch grid
            grid = self._stitch_grid(annotated_tiles)

            # JPEG encode
            _, jpeg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
            with self.lock:
                self.current_frame = jpeg.tobytes()

            # Push stats every 5 frames
            if self.frame_count % 5 == 0:
                self.socketio.emit("yolo_state", self._build_state())

            # Prevent CPU hogging
            if not any_active:
                time.sleep(0.1)
            else:
                time.sleep(0.01)

    # ------------------------------------------------------------------ #
    #  YOLO Detection                                                       #
    # ------------------------------------------------------------------ #
    def _detect(self, frame):
        counts  = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0}
        boxes   = []
        emerg   = detect_emergency_color(frame)

        if not self.model_ready or self.model is None:
            return counts, boxes, emerg

        results = self.model(frame, verbose=False, conf=0.35, iou=0.45, device="cpu")[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            label, _ = VEHICLE_CLASSES[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            counts[label] += 1
            boxes.append({"box": (x1, y1, x2, y2), "label": label, "conf": conf})

        return counts, boxes, emerg

    # ------------------------------------------------------------------ #
    #  Frame Annotation                                                     #
    # ------------------------------------------------------------------ #
    def _annotate_tile(self, frame, lane_idx, counts, boxes, emerg):
        tile  = frame.copy()
        color = LANE_COLORS[lane_idx % len(LANE_COLORS)]
        name  = self.lanes[lane_idx].name

        # Draw bounding boxes
        for b in boxes:
            x1, y1, x2, y2 = b["box"]
            cv2.rectangle(tile, (x1, y1), (x2, y2), color, 2)
            lbl = f"{b['label']} {b['conf']:.2f}"
            cv2.putText(tile, lbl, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # Top banner
        cv2.rectangle(tile, (0, 0), (TILE_W, 44), (8, 10, 20), -1)
        cv2.rectangle(tile, (0, 0), (TILE_W, 44), color, 1)

        # Lane name
        cv2.putText(tile, name, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 244, 255), 1)

        # Vehicle counts in banner
        total = sum(counts.values())
        info  = f"C:{counts['car']}  T:{counts['truck']}  B:{counts['bus']}  M:{counts['motorcycle']}  [{total}]"
        cv2.putText(tile, info, (8, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 200), 1)

        # Emergency banner
        if emerg:
            cv2.rectangle(tile, (0, TILE_H - 34), (TILE_W, TILE_H), (0, 0, 160), -1)
            cv2.putText(tile, "! EMERGENCY VEHICLE !", (10, TILE_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        return tile

    def _apply_light_border(self, tile, lane_idx, light):
        """Draw a thick colored border indicating GREEN/RED state."""
        out   = tile.copy()
        bw    = 5
        color = (0, 220, 82) if light == "green" else (50, 59, 255)
        cv2.rectangle(out, (0, 0), (TILE_W - 1, TILE_H - 1), color, bw)

        # Light circle indicator top-right
        cx, cy = TILE_W - 22, 22
        cv2.circle(out, (cx, cy), 12, color, -1)
        cv2.circle(out, (cx, cy), 12, (240, 244, 255), 1)

        # Green time label bottom-right
        lane    = self.lanes[lane_idx]
        label   = f"W:{lane.weight}  {lane.green_time}s"
        cv2.putText(out, label, (TILE_W - 105, TILE_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 230), 1)

        return out

    def _placeholder_tile(self, lane_idx):
        """Black tile with upload prompt when no video is loaded."""
        tile  = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        tile[:] = (12, 14, 24)
        color = LANE_COLORS[lane_idx % len(LANE_COLORS)]
        name  = self.lanes[lane_idx].name
        cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), color, 2)
        cv2.putText(tile, name, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)
        cv2.putText(tile, "No video loaded", (TILE_W // 2 - 70, TILE_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (100, 110, 140), 1)
        cv2.putText(tile, "Upload a video for this lane",
                    (TILE_W // 2 - 100, TILE_H // 2 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 80, 110), 1)
        return tile

    # ------------------------------------------------------------------ #
    #  Grid Stitching                                                       #
    # ------------------------------------------------------------------ #
    def _stitch_grid(self, tiles):
        """Stitch N tiles into a 2-column grid."""
        n = len(tiles)
        if n == 0:
            return np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)

        # Pad to even count for 2-column layout
        while len(tiles) % 2 != 0:
            tiles.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))

        rows = []
        for i in range(0, len(tiles), 2):
            sep = np.zeros((TILE_H, 3, 3), dtype=np.uint8)  # thin separator
            rows.append(np.hstack([tiles[i], sep, tiles[i + 1]]))

        h_sep = np.zeros((3, rows[0].shape[1], 3), dtype=np.uint8)
        grid  = rows[0]
        for row in rows[1:]:
            grid = np.vstack([grid, h_sep, row])

        return grid

    # ------------------------------------------------------------------ #
    #  Cycle Logic                                                         #
    # ------------------------------------------------------------------ #
    def _update_cycle(self):
        self._cycle_timer -= 1
        if self._cycle_timer <= 0:
            # Check for emergency on any lane
            emerg_lanes = [i for i in range(self.n_lanes) if self.lanes[i].emergency]
            if emerg_lanes:
                self.active_lane  = emerg_lanes[0]
                self.emergency    = True
                self._cycle_timer = 15
            else:
                self.emergency = False
                weights = [self.lanes[i].weight for i in range(self.n_lanes)]
                self.active_lane  = int(np.argmax(weights)) if any(w > 0 for w in weights) else 0
                best_w            = weights[self.active_lane] if weights else 5.0
                self._cycle_timer = compute_green_time(best_w)

    def force_emergency(self):
        self.emergency    = True
        self._cycle_timer = 0
        weights = [self.lanes[i].weight for i in range(self.n_lanes)]
        if any(w > 0 for w in weights):
            self.active_lane = int(np.argmax(weights))

    # ------------------------------------------------------------------ #
    #  State Payload                                                        #
    # ------------------------------------------------------------------ #
    def _build_state(self):
        quads = []
        for i in range(self.n_lanes):
            l = self.lanes[i]
            quads.append({
                "name":       l.name,
                "idx":        i,
                "loaded":     l.loaded,
                "counts":     l.counts.copy(),
                "weight":     l.weight,
                "green_time": l.green_time,
                "light":      l.light,
                "emergency":  l.emergency,
            })
        return {
            "quads":       quads,
            "active_quad": self.active_lane,
            "timer":       self._cycle_timer,
            "emergency":   self.emergency,
            "n_lanes":     self.n_lanes,
            "model_ready": self.model_ready,
        }
