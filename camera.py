import cv2
import cv2.aruco as aruco
import numpy as np
import time
import threading

# ============================================================
#  KeiganMotor SDK
# ============================================================
try:
    from pykeigan import usbcontroller
    KEIGAN_AVAILABLE = True
except ImportError:
    KEIGAN_AVAILABLE = False
    print("KeiganMotor SDK が見つかりません。")

# ============================================================
#  ArUco Tracker
# ============================================================
class ArUcoTracker:
    def __init__(self, camera_id=0, width=1280, height=720, fps=30,
                 port="COM3", max_vel=5.0, kp=0.005, deadband_px=10):

        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.port = port
        self.max_vel = max_vel
        self.kp = kp
        self.deadband_px = deadband_px
        self.fov_scale = 0.5   # 画角を50%にする（数値を小さくするとより狭く）

        self.running = True

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = (
            aruco.DetectorParameters()
            if hasattr(aruco, "DetectorParameters")
            else aruco.DetectorParameters_create()
        )

        self.detector = (
            aruco.ArucoDetector(self.aruco_dict, self.parameters)
            if hasattr(aruco, "ArucoDetector")
            else None
        )

        self.cap = None
        self.dev = None
        self.init_motor()

    # --------------------------------------------------------
    def init_motor(self):
        if KEIGAN_AVAILABLE:
            try:
                self.dev = usbcontroller.USBController(self.port)
                self.dev.enable_action()
                print("Motor Ready")
            except Exception as e:
                print("Motor Error:", e)

    # --------------------------------------------------------
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera Ready ({w}x{h}@{fps:.1f}fps)")

    # --------------------------------------------------------
    def command_velocity(self, v):
        if not self.dev:
            return

        if abs(v) < 0.001:
            self.dev.stop_motor()
            return

        v = max(min(v, self.max_vel), -self.max_vel)
        self.dev.set_speed(abs(v))

        if v > 0:
            self.dev.run_forward()
        else:
            self.dev.run_reverse()

    # --------------------------------------------------------
    def detect_markers(self, gray):
        if self.detector:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters
            )
        return corners, ids

    # --------------------------------------------------------
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids = self.detect_markers(gray)

        # センターライン
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            c = corners[0][0]
            cx = int(c[:, 0].mean())
            err = cx - (w // 2)

            if abs(err) <= self.deadband_px:
                self.command_velocity(0)
            else:
                self.command_velocity(-self.kp * err)
        else:
            self.command_velocity(0)

        return frame
    
    def crop_fov(self, frame):
        h, w = frame.shape[:2]

        cw = int(w * self.fov_scale)
        ch = int(h * self.fov_scale)

        x0 = (w - cw) // 2
        y0 = (h - ch) // 2

        return frame[y0:y0 + ch, x0:x0 + cw]


    # --------------------------------------------------------
    def run(self):
        self.initialize_camera()
        print("Tracking Started")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.crop_fov(frame)

            frame = self.process_frame(frame)
            cv2.imshow("ArUco Tracker", frame)

            # qキーで終了
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break

        self.cleanup()

    # --------------------------------------------------------
    def cleanup(self):
        if self.dev:
            self.dev.stop_motor()
            self.dev.disable_action()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Tracker Stopped")


# ============================================================
#  Enter Key Control
# ============================================================
tracker = None
tracker_thread = None
tracker_started = False

def start_tracker():
    global tracker, tracker_thread, tracker_started

    if tracker_started:
        print("Tracker already running")
        return

    tracker = ArUcoTracker()
    tracker.running = True
    tracker_thread = threading.Thread(target=tracker.run)
    tracker_thread.start()
    tracker_started = True
    print(">>> Tracker STARTED")

def stop_tracker():
    global tracker, tracker_started

    if tracker_started and tracker:
        print(">>> Stopping tracker...")
        tracker.running = False
        tracker_thread.join()
        tracker_started = False
        print(">>> Tracker STOPPED")


# ============================================================
#  Main
# ============================================================
print("=================================")
print(" ENTER : Start / Stop tracker")
print(" q     : Quit (while running)")
print(" Ctrl+C: Exit program")
print("=================================")

try:
    while True:
        input()
        if not tracker_started:
            start_tracker()
        else:
            stop_tracker()

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    if tracker_started:
        stop_tracker()

