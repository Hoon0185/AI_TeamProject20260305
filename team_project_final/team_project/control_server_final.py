import sys
import cv2
import rclpy
import os
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, qos_profile_sensor_data
from ultralytics import YOLO
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class RosWorker(QThread):
    yolo_signal = pyqtSignal(str)
    teleop_log_signal = pyqtSignal(str)
    image_signal = pyqtSignal(QPixmap)
    map_signal = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()
        if not rclpy.ok(): rclpy.init()
        self.node = Node('turtlebot_client_final')
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_client = ActionClient(self.node, NavigateToPose, 'navigate_to_pose')

        self.model = None
        self.model_path = '../robot_ws/best.pt'

        # 실시간 동기화를 위한 맵 데이터 저장소
        self.raw_map_data = None
        self.map_info = None
        self.robot_pose = (0.0, 0.0)

        # 구독 설정
        map_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, depth=1)

        self.map_sub = self.node.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile=map_qos)
        self.odom_sub = self.node.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile=qos_profile_sensor_data)
        self.yolo_cam = self.node.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.camera_callback, qos_profile=qos_profile_sensor_data)

        self.current_angular = 0.0
        self.current_speed = 0.0
        self.running = True

        # 최종 결과를 출력하기 위한 변수 초기화
        self.left    = "not_found"
        self.right   = "not_found"
        self.o_stop  = "not_found"
        self.e_stop  = "not_found"

    def run(self):
        if self.model is None:
            try: self.model = YOLO(self.model_path)
            except: pass
        while rclpy.ok() and self.running:
            rclpy.spin_once(self.node, timeout_sec=0.01)
            self.msleep(10)

    def odom_callback(self, msg):
        # 로봇이 움직일 때마다 좌표 업데이트 및 맵 다시 그리기
        self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.raw_map_data is not None:
            self.update_map_view()

    def map_callback(self, msg):
        # 새로운 맵 데이터가 들어오면 저장
        self.map_info = msg.info
        self.raw_map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.update_map_view()

    def update_map_view(self):
        if self.raw_map_data is None or self.map_info is None: return

        # 지도는 5번에 한 번만 전송하여 통신 부하 감소
        if hasattr(self, 'map_skip_count'):
            self.map_skip_count += 1
        else:
            self.map_skip_count = 0

        if self.map_skip_count % 5 != 0: return

        h, w = self.raw_map_data.shape

        map_img = np.zeros((h, w, 3), dtype=np.uint8)
        map_img[self.raw_map_data == 0] = [255, 255, 255]
        map_img[self.raw_map_data == 100] = [0, 0, 0]
        map_img[self.raw_map_data == -1] = [100, 100, 100]

        rx = int((self.robot_pose[0] - self.map_info.origin.position.x) / self.map_info.resolution)
        ry = int((self.robot_pose[1] - self.map_info.origin.position.y) / self.map_info.resolution)

        if 0 <= rx < w and 0 <= ry < h:
            cv2.circle(map_img, (rx, ry), 5, (255, 0, 0), -1)

        map_img = cv2.flip(map_img, 0)
        display_map = cv2.resize(map_img, (600, 600), interpolation=cv2.INTER_NEAREST)
        h_d, w_d, c_d = display_map.shape
        qimg = QImage(display_map.data, w_d, h_d, w_d*c_d, QImage.Format_RGB888).copy()
        self.map_signal.emit(QPixmap.fromImage(qimg))

    def camera_callback(self, msg):
        if self.model is None: return
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            results = self.model.predict(frame, conf=0.5, verbose=False)
            for box in results[0].boxes:
                name = self.model.names[int(box.cls[0])]
                self.yolo_signal.emit(f"[{datetime.now().strftime('%H:%M:%S')}] {name}")

                if name == "e_stop" :
                    self.e_stop = "found"
                elif name == "o_stop" :
                    self.o_stop = "found"
                elif name == "left" :
                    self.left = "found"
                elif name == "right" :
                    self.right = "found"

            ann_frame = results[0].plot()
            rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data, w, h, w*c, QImage.Format_RGB888).copy()
            self.image_signal.emit(QPixmap.fromImage(qimg))

    def send_cmd(self, lin, ang):
        # 선속도 로직 (기존 유지)
        if lin == 0.0 and ang == 0.0: # STOP 버튼 대응
            self.current_speed = 0.0
            self.current_angular = 0.0
        elif lin != 0.0:
            self.current_speed = max(-0.5, min(self.current_speed + lin, 2.0))

        # 각속도 로직 추가 (누를 때마다 ang만큼 증가/감소)
        if ang != 0.0:
            self.current_angular = max(-2.0, min(self.current_angular + ang, 2.0))

        # 메시지 발행
        t = Twist()
        t.linear.x = float(self.current_speed)
        t.angular.z = float(self.current_angular)
        self.cmd_vel_pub.publish(t)

        self.teleop_log_signal.emit(f"CMD: L:{self.current_speed:.1f}, A:{self.current_angular:.1f}")

    def send_goal(self, x, y):
        try:
            target_x = float(str(x).strip())
            target_y = float(str(y).strip())
        except:
            self.teleop_log_signal.emit("좌표 입력 오류!")
            return

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.teleop_log_signal.emit("Nav2 서버 연결 실패!")
            return

        # 새 목표를 보낼 때 이전 핸들을 확실히 취소하고 잠시 대기
        if hasattr(self, 'goal_handle') and self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            QThread.msleep(100)

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x = target_x
        goal.pose.pose.position.y = target_y
        goal.pose.pose.orientation.w = 1.0

        self.teleop_log_signal.emit(f"[경로 탐색] 목적지 전송: ({target_x}, {target_y})")
        self.nav_client.send_goal_async(goal).add_done_callback(self.goal_res)

    def goal_res(self, future):
        handle = future.result()
        if handle.accepted:
            self.goal_handle = handle # 핸들 저장
            handle.get_result_async().add_done_callback(self.goal_fin)

    def goal_fin(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.teleop_log_signal.emit("목적지에 성공적으로 도착했습니다!")
            self.teleop_log_signal.emit(f"left = {self.left}, right = {self.right}, stop = {self.o_stop}, e_stop = {self.e_stop}")

            # 출력 후 초기화
            self.e_stop = "not_found"
            self.o_stop = "not_found"
            self.left   = "not_found"
            self.right  = "not_found"

        elif status == GoalStatus.STATUS_ABORTED:
            # 끼임 발생 시 처리
            self.teleop_log_signal.emit("경로 차단됨, 자동 탈출 로직 실행")
            self.escape_behavior()
        elif status == GoalStatus.STATUS_CANCELED:
            self.teleop_log_signal.emit("주행이 취소되었습니다.")

    def escape_behavior(self):
        """
        좁은 길목 끼임 발생 시 로봇이 더 영리하게 빠져나오도록 수정
        """
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        QThread.msleep(1000) # 완전히 멈출 때까지 대기

        # 1단계: 저속 후진으로 벽과의 거리 확보
        self.teleop_log_signal.emit("벽에서 떨어지기 위해 저속 후진 중")
        escape_msg = Twist()
        escape_msg.linear.x = -0.06  # 아주 천천히 (0.1은 실제 로봇에겐 빠를 수 있음)
        self.cmd_vel_pub.publish(escape_msg)
        QThread.msleep(2500) # 충분히 뒤로 뺌

        # 2단계: 제자리 회전으로 각도 틀기
        escape_msg.linear.x = 0.0
        escape_msg.angular.z = 0.4
        self.cmd_vel_pub.publish(escape_msg)
        QThread.msleep(1500)

        self.cmd_vel_pub.publish(stop_msg)
        self.teleop_log_signal.emit("공간 확보 완료. 다시 목적지를 설정해 주세요.")

    def stop(self):
        self.running = False
        self.wait()

    def cancel_nav(self):
        self.send_cmd(0.0, 0.0) # 즉시 정지 명령

        # 결과 출력
        self.teleop_log_signal.emit(f"left = {self.left}, right = {self.right}, stop = {self.o_stop}, e_stop = {self.e_stop}")
        
        # 강제 초기화
        self.left   = "not_found"
        self.right  = "not_found"
        self.o_stop = "not_found"
        self.e_stop = "not_found"

        if hasattr(self, 'goal_handle') and self.goal_handle is not None:
            self.teleop_log_signal.emit("자율 주행 취소 요청 중")
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None


# GUI
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
    def __init__(self, ros_worker):
        super().__init__()
        self.ros_worker = ros_worker
        self.setStyleSheet("border: 1px solid #444; border-radius: 10px; background-color: #25282b;")
        layout = QVBoxLayout(self)

        # 버튼 영역 레이아웃
        pad_layout = QGridLayout()
        pad_layout.setSpacing(15)

        # 일반 방향키 버튼 
        btn_style = """
        QPushButton {
            background-color: #3a3f44;
            border-radius: 10px;
            color: white;
            font-size: 24px;
            min-width: 80px;
            min-height: 60px;
            border: 1px solid #555;
        }
        QPushButton:pressed {
            background-color: #555c63;
            border: 2px solid #3498db;
        }
        """

        # STOP 버튼 
        stop_btn_style = """
        QPushButton {
            background-color: #a93226;
            color: white;
            border-radius: 10px;
            font-weight: bold;
            font-size: 16px;
            min-width: 80px;
            min-height: 60px;
        }
        QPushButton:pressed {
            background-color: #e74c3c;
            border: 2px solid white;
        }
        """

        btn_up = QPushButton("↑"); btn_down = QPushButton("↓")
        btn_left = QPushButton("←"); btn_right = QPushButton("→")
        btn_stop = QPushButton("STOP")

        # 스타일 적용
        for btn in [btn_up, btn_down, btn_left, btn_right]:
            btn.setStyleSheet(btn_style)
        btn_stop.setStyleSheet(stop_btn_style)

        # 시그널 연결
        btn_up.clicked.connect(lambda: self.ros_worker.send_cmd(0.1, 0.0))
        btn_down.clicked.connect(lambda: self.ros_worker.send_cmd(-0.1, 0.0))
        btn_left.clicked.connect(lambda: self.ros_worker.send_cmd(0.0, 0.1))
        btn_right.clicked.connect(lambda: self.ros_worker.send_cmd(0.0, -0.1))
        btn_stop.clicked.connect(self.ros_worker.cancel_nav)

        # 배치
        pad_layout.addWidget(btn_up, 0, 1)
        pad_layout.addWidget(btn_left, 1, 0)
        pad_layout.addWidget(btn_stop, 1, 1)
        pad_layout.addWidget(btn_right, 1, 2)
        pad_layout.addWidget(btn_down, 2, 1)

        pad_layout.setColumnStretch(0, 1)
        pad_layout.setColumnStretch(1, 1)
        pad_layout.setColumnStretch(2, 1)

        # 로그 영역
        self.teleop_log = QTextEdit()
        self.teleop_log.setReadOnly(True)
        self.teleop_log.setStyleSheet("background-color: #121212; color: #00ff00; font-family: Consolas; border: none; font-size: 11px;")

        layout.addWidget(QLabel("MANUAL CONTROL"))
        layout.addLayout(pad_layout)
        layout.addWidget(self.teleop_log)

    @pyqtSlot(str)
    def add_log(self, text):
        self.teleop_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")


class NavWidget(QFrame):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        nav_btn_style = """
        QPushButton { background-color: #555; color: white; border-radius: 5px; font-weight: bold; min-height: 30px; }
        QPushButton:hover { background-color: #666; }
        QPushButton:pressed { background-color: #3498db; border: 1px solid white; padding-top: 2px; padding-left: 2px; }
        """
        self.setStyleSheet("border: 1px solid #444; background: #25282b; color: white; border-radius: 10px;")
        l = QVBoxLayout(self); h = QHBoxLayout()

        h.addWidget(QLabel("x:")); self.ex = QLineEdit(); h.addWidget(self.ex)
        h.addWidget(QLabel("y:")); self.ey = QLineEdit(); h.addWidget(self.ey)

        # 프리셋 좌표 설정
        presets = { 'A': (4.4, 0.0), 'B': (4.4, -3.4), 'C': (0.0, 0.0), 'D': (-0.4, -1.5) }

        for txt, pos in presets.items():
            btn = QPushButton(txt)
            btn.setFixedWidth(40)
            btn.setStyleSheet(nav_btn_style)
            
            btn.clicked.connect(lambda ch, x_val=pos[0], y_val=pos[1]: self.worker.send_goal(x_val, y_val))
            h.addWidget(btn)

        l.addWidget(QLabel(" NAV Control")); l.addLayout(h)

        self.btn_go = QPushButton("좌표로 이동")
        self.btn_go.setStyleSheet(nav_btn_style.replace("min-height: 30px;", "min-height: 40px; background-color: #2e86c1;"))
        # 텍스트 박스의 값을 실시간으로 읽어 전달
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
        ml = QVBoxLayout(self.mw); ml.addWidget(QLabel("REAL-TIME SLAM MAP"))
        self.map_lbl = QLabel("WAITING FOR MAP..."); self.map_lbl.setStyleSheet("background: #000;")
        self.map_lbl.setAlignment(Qt.AlignCenter); self.map_lbl.setScaledContents(True)
        ml.addWidget(self.map_lbl, stretch=1)

        self.nav = NavWidget(self.worker); self.yolo = YoloWidget()
        self.initUI()
        self.worker.image_signal.connect(self.cw.lbl.setPixmap)
        self.worker.map_signal.connect(self.map_lbl.setPixmap)
        self.worker.teleop_log_signal.connect(self.ctrl.add_log)
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
