import sys
import cv2
import rclpy
import os
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import OccupancyGrid  # 지도 데이터 타입 추가
from cv_bridge import CvBridge
from ultralytics import YOLO
from datetime import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# ---------------------------------------------------------
# 0. ROS Worker (지도 구독 및 시그널 추가)
# ---------------------------------------------------------
class RosWorker(QThread):
    yolo_signal = pyqtSignal(str)
    teleop_log_signal = pyqtSignal(str)
    image_signal = pyqtSignal(QPixmap)
    map_signal = pyqtSignal(QPixmap)  # 지도 전송용 시그널 추가

    def __init__(self):
        super().__init__()
        if not rclpy.ok(): rclpy.init()
        self.node = Node('turtlebot_client_final')
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.node.create_publisher(PoseStamped, '/goal_pose', 10)

        self.model = None
        self.model_path = '/home/raheonseok/robot_ws/best.pt'
        self.yolo_cam = self.node.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.camera_callback, 10)

        # --- [지도 구독 추가] ---
        self.map_sub = self.node.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        self.result_pub = self.node.create_publisher(String, 'yolo_result', 10)
        self.current_speed = 0.0
        self.running = True

    def run(self):
        if self.model is None:
            self.teleop_log_signal.emit("🔄 YOLO 및 Nav2 고성능 모드 로딩 중...")
            try:
                self.model = YOLO(self.model_path)
                os.system("ros2 param set /local_costmap/local_costmap inflation_layer.inflation_radius 0.13")
                os.system("ros2 param set /global_costmap/global_costmap inflation_layer.inflation_radius 0.13")
                os.system("ros2 param set /local_costmap/local_costmap inflation_layer.cost_scaling_factor 30.0")
                os.system("ros2 param set /bt_navigator bt_loop_duration 10")
                os.system("ros2 param set /controller_server failure_tolerance 2.0")
                os.system("ros2 param set /dwb_controller follow_path.PathAlign.scale 1.0")
                os.system("ros2 param set /dwb_controller follow_path.PathDist.scale 40.0")
                os.system("ros2 param set /dwb_controller follow_path.min_vel_theta 0.2")
                os.system("ros2 param set /behavior_server local_frame map")
                self.teleop_log_signal.emit("✅ 'Aborted' 방지 및 코너 탈출 모드 가동!")
            except Exception as e:
                self.teleop_log_signal.emit(f"❌ 설정 실패: {e}")

        while rclpy.ok() and self.running:
            rclpy.spin_once(self.node, timeout_sec=0.01)
            self.msleep(5)

    def camera_callback(self, msg):
        if self.model is None: return
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            results = self.model.predict(frame, conf=0.5, verbose=False)
            for box in results[0].boxes:
                name = self.model.names[int(box.cls[0])]
                self.yolo_signal.emit(f"[{datetime.now().strftime('%H:%M:%S')}] {name}")
            ann_frame = results[0].plot()
            rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data, w, h, w*c, QImage.Format_RGB888).copy()
            self.image_signal.emit(QPixmap.fromImage(qimg))

    # --- [지도 데이터 처리 콜백 추가] ---
    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data).reshape((height, width))

        # 지도 데이터 시각화 (0: 빈공간, 100: 벽, -1: 미탐사)
        map_img = np.zeros((height, width, 3), dtype=np.uint8)
        map_img[data == 0] = [255, 255, 255]    # 빈공간: 흰색
        map_img[data == 100] = [0, 0, 0]        # 벽: 검은색
        map_img[data == -1] = [127, 127, 127]   # 미탐사: 회색

        # 위아래 반전 방지 및 이미지 변환
        map_img = cv2.flip(map_img, 0)
        h, w, c = map_img.shape
        qimg = QImage(map_img.data, w, h, w*c, QImage.Format_RGB888).copy()
        self.map_signal.emit(QPixmap.fromImage(qimg))

    def send_cmd(self, lin, ang):
        if lin == 0.0: self.current_speed = 0.0
        else: self.current_speed = max(-0.5, min(self.current_speed + lin, 2.0))
        t = Twist(); t.linear.x = float(self.current_speed); t.angular.z = float(ang)
        self.cmd_vel_pub.publish(t)
        self.teleop_log_signal.emit(f"CMD: L:{self.current_speed:.1f}, A:{ang:.1f}")

    def send_goal(self, x, y):
        try:
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = self.node.get_clock().now().to_msg()
            goal.pose.position.x = float(x)
            goal.pose.position.y = float(y)
            goal.pose.position.z = 0.0
            goal.pose.orientation.x = 0.0
            goal.pose.orientation.y = 0.0
            goal.pose.orientation.z = 0.0
            goal.pose.orientation.w = 1.0
            self.goal_pub.publish(goal)
            self.teleop_log_signal.emit(f"🎯 Goal Sent: X={x}, Y={y}")
        except Exception as e:
            self.teleop_log_signal.emit(f"❌ 목표 전송 실패: {e}")

    def stop(self):
        self.running = False
        self.wait()

# ---------------------------------------------------------
# UI (변경사항 없음, 단 SLAM 뷰어 라벨 변수화)
# ---------------------------------------------------------
class CameraWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("border: 1px solid #444; background: #25282b; border-radius: 10px;")
        l = QVBoxLayout(self)
        self.lbl = QLabel("CAM FEED VIEW"); self.lbl.setStyleSheet("background: #000;")
        self.lbl.setAlignment(Qt.AlignCenter); self.lbl.setScaledContents(True)
        l.addWidget(QLabel(" ● TURTLEBOT CAM STREAM"))
        l.addWidget(self.lbl, stretch=1)

class ControlWidget(QFrame):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.setStyleSheet("border: 1px solid #444; background: #25282b; border-radius: 10px;")
        l = QVBoxLayout(self); gl = QGridLayout()
        bs = "QPushButton { background: #3a3f44; color: white; font-size: 20px; min-height: 50px; border-radius: 5px; }"
        ss = "QPushButton { background: #a93226; color: white; font-weight: bold; min-height: 50px; border-radius: 5px; }"
        btns = { '↑': (0,1, 0.1, 0), '↓': (2,1, -0.1, 0), '←': (1,0, 0, 0.5), '→': (1,2, 0, -0.5), 'STOP': (1,1, 0, 0) }
        for k, v in btns.items():
            b = QPushButton(k); b.setStyleSheet(ss if k == 'STOP' else bs)
            b.clicked.connect(lambda ch, x=v[2], y=v[3]: self.worker.send_cmd(x, y))
            gl.addWidget(b, v[0], v[1])
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setStyleSheet("background: #000; color: #0f0; font-family: Consolas; font-size: 10px;")
        l.addWidget(QLabel(" MANUAL CONTROL")); l.addLayout(gl); l.addWidget(self.log)

class NavWidget(QFrame):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.setStyleSheet("border: 1px solid #444; background: #25282b; color: white; border-radius: 10px;")
        l = QVBoxLayout(self)
        h = QHBoxLayout()
        h.addWidget(QLabel("x:")); self.ex = QLineEdit(); h.addWidget(self.ex)
        h.addWidget(QLabel("y:")); self.ey = QLineEdit(); h.addWidget(self.ey)
        presets = { 'A': (4.4, 0.0), 'B': (4.4, -3.4), 'C': (0.0, 0.0), 'D': (-0.4, -1.5) }
        for txt, pos in presets.items():
            btn = QPushButton(txt); btn.setFixedWidth(30); btn.setStyleSheet("background: #555;")
            btn.clicked.connect(lambda ch, x=pos[0], y=pos[1]: self.worker.send_goal(x, y))
            h.addWidget(btn)
        l.addWidget(QLabel(" NAV Control")); l.addLayout(h)
        self.btn_go = QPushButton("좌표로 이동")
        self.btn_go.clicked.connect(lambda: self.worker.send_goal(self.ex.text(), self.ey.text()))
        l.addWidget(self.btn_go)

class YoloWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("border: 1px solid #444; background: #25282b; border-radius: 10px;")
        l = QVBoxLayout(self); self.lst = QListWidget()
        self.lst.setStyleSheet("background: #000; color: #3498db; border: none;")
        l.addWidget(QLabel(" YOLO DETECTION")); l.addWidget(self.lst)

class TurtlebotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = RosWorker()
        self.cw = CameraWidget(); self.ctrl = ControlWidget(self.worker)
        self.mw = QFrame(); self.mw.setStyleSheet("border: 1px solid #444; background: #25282b; border-radius: 10px;")
        ml = QVBoxLayout(self.mw)
        ml.addWidget(QLabel(" 🗺️ REAL-TIME SLAM MAP"))

        # --- [SLAM 뷰어용 라벨 추가] ---
        self.map_lbl = QLabel("WAITING FOR MAP..."); self.map_lbl.setStyleSheet("background: #000;")
        self.map_lbl.setAlignment(Qt.AlignCenter); self.map_lbl.setScaledContents(True)
        ml.addWidget(self.map_lbl, stretch=1)
        # ----------------------------

        self.nav = NavWidget(self.worker); self.yolo = YoloWidget()
        self.initUI()
        self.worker.image_signal.connect(self.cw.lbl.setPixmap)
        self.worker.map_signal.connect(self.map_lbl.setPixmap) # 지도 시그널 연결
        self.worker.teleop_log_signal.connect(self.ctrl.log.append)
        self.worker.yolo_signal.connect(lambda t: self.yolo.lst.addItem(t))
        self.worker.start()

    def initUI(self):
        self.setWindowTitle("Turtlebot Manual Control Client")
        self.setFixedSize(1100, 800)
        self.setStyleSheet("background: #1a1c1e; color: #dcdcdc; font-family: 'Consolas';")
        main_w = QWidget(); self.setCentralWidget(main_w); layout = QGridLayout(main_w)
        layout.addWidget(self.cw, 0, 0); layout.addWidget(self.ctrl, 1, 0)
        layout.addWidget(self.mw, 0, 1)
        right_b = QWidget(); rbl = QVBoxLayout(right_b); rbl.setContentsMargins(0,0,0,0)
        rbl.addWidget(self.nav, 1); rbl.addWidget(self.yolo, 2)
        layout.addWidget(right_b, 1, 1)

    def closeEvent(self, event):
        self.worker.stop(); rclpy.shutdown(); event.accept()

def main():
    app = QApplication(sys.argv); ex = TurtlebotApp(); ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__': main()
