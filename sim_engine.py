"""
sim_engine.py — Simulation State Machine
4-lane intersection: North, South, East, West
Weight-based dynamic timing, emergency corridor override.
"""

import random
import time
import threading

LANES = ["N", "S", "E", "W"]
LANE_NAMES = {"N": "North", "S": "South", "E": "East", "W": "West"}


def compute_weight(lane_data):
    return round(
        lane_data["cars"] * 1.0 +
        lane_data["trucks"] * 1.5 +
        lane_data["buses"] * 2.5,
        1
    )


def compute_timer(weight):
    return max(6, min(18, int(weight * 1.2)))


class SimEngine:
    def __init__(self, socketio=None, mode="dynamic", initial_seed=None):
        self.socketio = socketio
        self.lock = threading.Lock()
        self.running = False
        self.mode = mode  # "dynamic" or "fixed"
        
        if initial_seed is not None:
            self.rng = random.Random(initial_seed)
        else:
            self.rng = random.Random()

        # Per-lane vehicle queues
        self.lanes = {
            lane: {
                "cars":   self.rng.randint(2, 8),
                "trucks": self.rng.randint(0, 3),
                "buses":  self.rng.randint(0, 2)
            }
            for lane in LANES
        }

        self.active_lane = "N"
        self.timer = self._get_timer_duration(self.active_lane)
        self.emergency = False
        self.emergency_lane = None   # which lane has the corridor open
        self.emergency_countdown = 0
        self.stats = {"total_vehicles": 0, "cycles": 0, "vehicles_passed": 0}
        self.weights = {}
        self._recalc_weights()
        # Ambulance tracking: {lane: count}
        self.ambulances = {lane: 0 for lane in LANES}
        self.AMBULANCE_SPAWN_RATE = 0.015  # 1.5% per tick per lane
        self.AMBULANCE_CORRIDOR_DURATION = 20  # seconds

    def _get_timer_duration(self, lane):
        if self.mode == "fixed":
            return 30
        else:
            return compute_timer(compute_weight(self.lanes[lane]))

    def _recalc_weights(self):
        for lane in LANES:
            self.weights[lane] = compute_weight(self.lanes[lane])

    def _pick_next_lane(self):
        if self.mode == "fixed":
            # Round robin for fixed mode
            idx = LANES.index(self.active_lane)
            return LANES[(idx + 1) % len(LANES)]
        else:
            # Heaviest lane for dynamic mode
            return max(self.weights, key=lambda l: self.weights[l])

    def _arrival_tick(self):
        """Simulate random vehicle arrivals and departures each tick."""
        for lane in LANES:
            # Random ambulance spawn — very low probability
            if self.rng.random() < self.AMBULANCE_SPAWN_RATE:
                self.ambulances[lane] = self.ambulances.get(lane, 0) + 1
                # Auto-trigger emergency corridor for this lane
                if not self.emergency:
                    self._do_trigger_emergency(lane)

            # Arrivals on all lanes
            if self.rng.random() < 0.6:
                vtype = self.rng.choices(
                    ["cars", "trucks", "buses"], weights=[70, 20, 10]
                )[0]
                self.lanes[lane][vtype] = min(
                    self.lanes[lane][vtype] + self.rng.randint(1, 3), 20
                )
                self.stats["total_vehicles"] += 1

            # Departures: active green lane OR the emergency corridor lane
            if lane == self.active_lane:
                for vtype in ["cars", "trucks", "buses"]:
                    depart = self.rng.randint(1, 4)
                    actual_depart = min(self.lanes[lane][vtype], depart)
                    self.lanes[lane][vtype] = max(
                        0, self.lanes[lane][vtype] - depart
                    )
                    self.stats["vehicles_passed"] += actual_depart
                # Clear ambulances on the corridor/green lane
                if self.ambulances.get(lane, 0) > 0:
                    self.ambulances[lane] = 0

    def tick(self):
        with self.lock:
            if self.emergency:
                self._arrival_tick()   # corridor lane keeps moving / arrivals happen
                self._recalc_weights()
                self.emergency_countdown -= 1
                if self.emergency_countdown <= 0:
                    self.emergency = False
                    self.emergency_lane = None
            else:
                self.timer -= 1
                if self.timer <= 0:
                    self._arrival_tick()
                    self._recalc_weights()
                    self.active_lane = self._pick_next_lane()
                    self.timer = self._get_timer_duration(self.active_lane)
                    self.stats["cycles"] += 1
                else:
                    self._arrival_tick()
                    self._recalc_weights()

            # Build state payload
            state = {
                "lanes": {},
                "active_lane": self.active_lane,
                "timer": self.timer if not self.emergency else self.emergency_countdown,
                "emergency": self.emergency,
                "emergency_lane": self.emergency_lane,
                "stats": self.stats.copy(),
                "weights": self.weights.copy()
            }
            for lane in LANES:
                w = self.weights[lane]
                state["lanes"][lane] = {
                    **self.lanes[lane],
                    "ambulances": self.ambulances.get(lane, 0),
                    "weight": w,
                    "green_time": self._get_timer_duration(lane),
                    "name": LANE_NAMES[lane],
                    "light": "green" if lane == self.active_lane else "red"
                }

        return state

    def _do_trigger_emergency(self, lane=None):
        """Internal — must be called with lock held or before lock needed."""
        self.emergency = True
        self.emergency_countdown = self.AMBULANCE_CORRIDOR_DURATION
        self._recalc_weights()
        if lane in LANES:
            self.active_lane = lane
            self.emergency_lane = lane
        else:
            self.active_lane = self._pick_next_lane()
            self.emergency_lane = self.active_lane

    def trigger_emergency(self, lane=None):
        with self.lock:
            self._do_trigger_emergency(lane)

    def cancel_emergency(self):
        with self.lock:
            self.emergency = False
            self.emergency_lane = None
            self.emergency_countdown = 0
            self.timer = self._get_timer_duration(self.active_lane)

    def run_loop(self):
        self.running = True
        while self.running:
            state = self.tick()
            if self.socketio:
                self.socketio.emit("sim_state", state)
            time.sleep(1.0)

    def stop(self):
        self.running = False


class ComparisonRunner:
    def __init__(self, socketio):
        self.socketio = socketio
        self.running = False
        self.lock = threading.Lock()
        
    def start_comparison(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            
        seed = int(time.time() * 1000)
        self.sim_fixed = SimEngine(mode="fixed", initial_seed=seed)
        self.sim_dynamic = SimEngine(mode="dynamic", initial_seed=seed)
        
        threading.Thread(target=self._run_loop, daemon=True).start()
        
    def _run_loop(self):
        ticks = 0
        max_ticks = 60
        while self.running and ticks < max_ticks:
            state_fixed = self.sim_fixed.tick()
            state_dynamic = self.sim_dynamic.tick()
            
            payload = {
                "fixed": state_fixed,
                "dynamic": state_dynamic,
                "progress": int((ticks / max_ticks) * 100),
                "remaining": max_ticks - ticks,
                "completed": False
            }
            
            if self.socketio:
                self.socketio.emit("comparison_state", payload)
            
            ticks += 1
            # Run at accelerated speed (5 ticks per second) for a quick 12-second test
            time.sleep(0.2)
            
        # Final payload
        if self.running:
            state_fixed = self.sim_fixed.tick()
            state_dynamic = self.sim_dynamic.tick()
            payload = {
                "fixed": state_fixed,
                "dynamic": state_dynamic,
                "progress": 100,
                "remaining": 0,
                "completed": True
            }
            if self.socketio:
                self.socketio.emit("comparison_state", payload)
                
        self.running = False

