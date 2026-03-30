import cv2
import cv2.aruco as aruco
import RPi.GPIO as GPIO
import time
import threading
import subprocess

# KeiganMotor SDK の読み込み確認
try:
    from pykeigan import usbcontroller
    KEIGAN_AVAILABLE = True
except ImportError:
    KEIGAN_AVAILABLE = False
    print("KeiganMotor SDK が見つかりません。")

# ============================================================
#  ArUcoトラッカークラス
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
        self.running = True  # トラッカー動作フラグ

        # ArUco検出器設定
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.cap = None
        self.dev = None
        self.init_motor()

    # KeiganMotor初期化
    def init_motor(self):
        if KEIGAN_AVAILABLE:
            try:
                self.dev = usbcontroller.USBController(self.port)
                self.dev.enable_action()
                print("Motor Ready")
            except Exception as e:
                print("Motor Error:", e)

    # カメラ初期化
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        print("Camera Ready")

    # モーター速度指令
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
    
    def detect_markers(self, gray):
        if self.detector:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters
            )
        return corners, ids

            
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

    # トラッカー実行
    def run(self):
        self.initialize_camera()
        print("Tracking Started")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.crop_fov(frame)
            
            frame = self.process_frame(frame)

        self.cleanup()

    # クリーンアップ
    def cleanup(self):
        if self.dev:
            self.dev.stop_motor()
            self.dev.disable_action()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        print("Tracker Stopped")


# ============================================================
#  ボタン操作：短押し＝開始 / 長押し＝停止 / ダブル押し＝シャットダウン
# ============================================================
BUTTON_PIN = 12
LONG_PRESS_TIME = 2.0           # 長押し判定（秒）
DOUBLE_PRESS_INTERVAL = 0.5     # ダブル押し判定（秒）

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

tracker = None
tracker_thread = None
tracker_started = False
press_start_time = None
last_release_time = 0
click_count = 0

# トラッカー開始
def start_tracker():
    global tracker, tracker_thread, tracker_started
    tracker = ArUcoTracker()
    tracker_thread = threading.Thread(target=tracker.run)
    tracker_thread.start()
    tracker_started = True

# トラッカー停止
def stop_tracker():
    global tracker, tracker_started
    if tracker_started and tracker:
        print("Stopping tracker...")
        tracker.running = False
        tracker_thread.join()
        tracker_started = False

# Raspberry Pi シャットダウン
def shutdown_pi():
    print("Shutting down Raspberry Pi...")
    subprocess.Popen(["sudo", "shutdown", "-h", "now"])

# ボタン状態チェック
def check_button():
    global press_start_time, click_count, last_release_time

    state = GPIO.input(BUTTON_PIN)

    # ボタン押下開始
    if state == GPIO.LOW:
        if press_start_time is None:
            press_start_time = time.time()

    # ボタン離上
    else:
        if press_start_time is not None:
            duration = time.time() - press_start_time

            # 長押し判定 → トラッカー停止
            if duration >= LONG_PRESS_TIME:
                print("Long press detected → Stop tracker")
                stop_tracker()
                press_start_time = None
                return True

            # 短押し → クリックカウント
            click_count += 1
            last_release_time = time.time()
            press_start_time = None

    # ダブル押し判定 → シャットダウン
    if click_count == 1:
        if time.time() - last_release_time > DOUBLE_PRESS_INTERVAL:
            print("Single short press → Start tracker")
            start_tracker()
            click_count = 0
    elif click_count >= 2:
        print("Double press detected → Shutdown Pi")
        shutdown_pi()
        return False

    return True


print("====== 準備完了 ======")
print("短押し  = トラッカー開始")
print("長押し  = トラッカー停止")
print("ダブル押し = Raspberry Pi シャットダウン")
print("===================")

try:
    while True:
        if not check_button():
            break
        time.sleep(0.05)
finally:
    GPIO.cleanup()
    if tracker_started:
        stop_tracker()

