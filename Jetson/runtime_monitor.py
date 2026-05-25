#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runtime_monitor.py — DonkeyCar 运行时数据采集独立脚本

完全独立于 manage.py，通过 monkey-patch Vehicle.start 在启动前
自动注入数据采集 Part，无需修改 manage.py 任何代码。

用法:
  # 采集 V8 自动驾驶数据
  python runtime_monitor.py drive --model ~/mycar/models/v8_140000_steps_policy.pth --type v8

  # 手动模式采集（不加载模型，只记录用户操作 + 系统状态）
  python runtime_monitor.py drive

  # 自定义日志间隔和文件
  python runtime_monitor.py drive --model xxx --type v8 --log-interval 0.2 --log-dir ~/mycar/monitor_logs

  # 使用手柄
  python runtime_monitor.py drive --model xxx --type v8 --js

参数与 manage.py drive 完全一致，额外参数:
  --log-interval  日志写入间隔(秒), 默认 0.5
  --log-dir       日志输出目录, 默认 ~/mycar/monitor_logs

输出:
  monitor_logs/
    run_20260226_153000.csv      ← 每次运行生成一个带时间戳的 CSV
"""

import os
import sys
import time
import csv
import json
import math
import signal
import struct
import shlex
import threading
import queue
import subprocess
import numpy as np
import textwrap
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ============================================================
# RP2040 串口传感器读取器
# ============================================================
class RP2040SerialReader:
    """
    后台线程读取 RP2040 扩展板的串口数据帧。

    协议格式 (jetracer.cpp 逆向):
      [0xAA, 0x55, frame_size, type, payload..., checksum]

    数据帧 payload (38 字节):
      gyro_xyz(6) + accel_xyz(6) + euler_rpy(6) +
      odom_xy_yaw(6) + delta_xy_yaw(6) +
      motor_lvel(2) + motor_rvel(2) + motor_lset(2) + motor_rset(2)
    """

    HEAD1 = 0xAA
    HEAD2 = 0x55

    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._serial = None
        self._connected = False

        # 最新传感器数据（线程安全读取）
        self._data = {
            # IMU
            'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0,       # rad/s
            'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0,    # m/s^2
            'euler_roll': 0.0, 'euler_pitch': 0.0, 'euler_yaw': 0.0,  # degrees
            # 里程计
            'odom_x': 0.0, 'odom_y': 0.0, 'odom_yaw': 0.0,     # meters / rad
            'delta_x': 0.0, 'delta_y': 0.0, 'delta_yaw': 0.0,  # meters / rad
            # 电机
            'motor_lvel': 0, 'motor_rvel': 0,                   # 实际编码速度
            'motor_lset': 0, 'motor_rset': 0,                   # 设定编码速度
            # 元数据
            'frame_count': 0,
            'last_update': 0.0,
            'parse_errors': 0,
        }

    def start(self):
        """启动后台读取线程"""
        try:
            import serial
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            self._connected = True
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            print(f"   RP2040 串口已连接: {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"   RP2040 串口连接失败: {e}")
            self._connected = False
            return False

    def stop(self):
        """停止读取线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._connected = False

    def get_data(self):
        """获取最新传感器数据的快照（线程安全）"""
        with self._lock:
            return dict(self._data)

    @property
    def is_connected(self):
        return self._connected

    def _read_loop(self):
        """后台线程主循环 —— 状态机解析串口帧"""
        buf = bytearray(64)
        state = 'HEAD1'
        frame_size = 0

        while self._running:
            try:
                if state == 'HEAD1':
                    b = self._serial.read(1)
                    if len(b) == 0:
                        continue
                    if b[0] == self.HEAD1:
                        buf[0] = b[0]
                        state = 'HEAD2'

                elif state == 'HEAD2':
                    b = self._serial.read(1)
                    if len(b) == 0:
                        state = 'HEAD1'
                        continue
                    if b[0] == self.HEAD2:
                        buf[1] = b[0]
                        state = 'SIZE'
                    else:
                        state = 'HEAD1'

                elif state == 'SIZE':
                    b = self._serial.read(1)
                    if len(b) == 0:
                        state = 'HEAD1'
                        continue
                    frame_size = b[0]
                    buf[2] = b[0]
                    if frame_size < 5 or frame_size > 60:
                        state = 'HEAD1'
                        continue
                    state = 'DATA'

                elif state == 'DATA':
                    # 读取 type + payload + checksum = frame_size - 3 字节
                    remaining = frame_size - 3
                    chunk = self._serial.read(remaining)
                    if len(chunk) != remaining:
                        state = 'HEAD1'
                        continue
                    buf[3:3 + remaining] = chunk

                    # 校验和: sum(buf[0:frame_size-1]) == buf[frame_size-1]
                    calc_sum = sum(buf[:frame_size - 1]) & 0xFF
                    if calc_sum != buf[frame_size - 1]:
                        with self._lock:
                            self._data['parse_errors'] += 1
                        state = 'HEAD1'
                        continue

                    # 解析成功
                    self._parse_frame(buf, frame_size)
                    state = 'HEAD1'

            except Exception as e:
                # 串口断开等异常
                time.sleep(0.1)
                state = 'HEAD1'

    def _parse_frame(self, buf, size):
        """解析 RP2040 数据帧，与 jetracer.cpp State_Handle 一致"""
        if size < 42:  # 至少需要到 motor 数据
            return

        d = buf

        # 陀螺仪 (rad/s): raw/32768 * 2000 deg/s -> rad/s
        gyro_x = (struct.unpack('>h', bytes(d[4:6]))[0]) / 32768.0 * 2000.0 / 180.0 * 3.1415926
        gyro_y = (struct.unpack('>h', bytes(d[6:8]))[0]) / 32768.0 * 2000.0 / 180.0 * 3.1415926
        gyro_z = (struct.unpack('>h', bytes(d[8:10]))[0]) / 32768.0 * 2000.0 / 180.0 * 3.1415926

        # 加速度 (m/s^2): raw/32768 * 2g * 9.8
        accel_x = (struct.unpack('>h', bytes(d[10:12]))[0]) / 32768.0 * 2.0 * 9.8
        accel_y = (struct.unpack('>h', bytes(d[12:14]))[0]) / 32768.0 * 2.0 * 9.8
        accel_z = (struct.unpack('>h', bytes(d[14:16]))[0]) / 32768.0 * 2.0 * 9.8

        # 欧拉角 (度): raw / 10.0
        euler_roll  = (struct.unpack('>h', bytes(d[16:18]))[0]) / 10.0
        euler_pitch = (struct.unpack('>h', bytes(d[18:20]))[0]) / 10.0
        euler_yaw   = (struct.unpack('>h', bytes(d[20:22]))[0]) / 10.0

        # 里程计位置 (m): raw / 1000
        odom_x   = (struct.unpack('>h', bytes(d[22:24]))[0]) / 1000.0
        odom_y   = (struct.unpack('>h', bytes(d[24:26]))[0]) / 1000.0
        odom_yaw = (struct.unpack('>h', bytes(d[26:28]))[0]) / 1000.0

        # 增量 (m): raw / 1000
        delta_x   = (struct.unpack('>h', bytes(d[28:30]))[0]) / 1000.0
        delta_y   = (struct.unpack('>h', bytes(d[30:32]))[0]) / 1000.0
        delta_yaw = (struct.unpack('>h', bytes(d[32:34]))[0]) / 1000.0

        # 电机编码速度
        motor_lvel = struct.unpack('>h', bytes(d[34:36]))[0]
        motor_rvel = struct.unpack('>h', bytes(d[36:38]))[0]
        motor_lset = struct.unpack('>h', bytes(d[38:40]))[0]
        motor_rset = struct.unpack('>h', bytes(d[40:42]))[0]

        with self._lock:
            self._data.update({
                'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
                'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z,
                'euler_roll': euler_roll, 'euler_pitch': euler_pitch, 'euler_yaw': euler_yaw,
                'odom_x': odom_x, 'odom_y': odom_y, 'odom_yaw': odom_yaw,
                'delta_x': delta_x, 'delta_y': delta_y, 'delta_yaw': delta_yaw,
                'motor_lvel': motor_lvel, 'motor_rvel': motor_rvel,
                'motor_lset': motor_lset, 'motor_rset': motor_rset,
                'last_update': time.time(),
            })
            self._data['frame_count'] += 1


# ============================================================
# ROS LiDAR 读取器
# ============================================================
def sectorize_lidar_scan(ranges, angle_min=0.0, angle_increment=0.0,
                         range_min=0.18, range_max=20.0, sectors=72,
                         fov_deg=360.0):
    """Convert a LaserScan range vector into fixed sector range/valid arrays."""
    def quantile(values, frac):
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * float(frac)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(ordered) - 1)
        weight = pos - lo
        return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

    sector_count = max(1, int(sectors or 1))
    max_r = float(range_max or 20.0)
    min_r = float(range_min or 0.18)
    sector_values = [[] for _ in range(sector_count)]
    raw = list(ranges or [])
    n = len(raw)
    angle_inc = float(angle_increment or 0.0)

    if n > 0:
        if abs(angle_inc) > 1e-12:
            half_fov = 0.5 * float(fov_deg or 360.0)
            edges = [
                half_fov - idx * float(fov_deg or 360.0) / float(sector_count)
                for idx in range(sector_count + 1)
            ]
            base_angle = float(angle_min or 0.0)
            for beam_idx, value in enumerate(raw):
                try:
                    v = float(value)
                except Exception:
                    continue
                if math.isnan(v) or math.isinf(v) or v < min_r:
                    continue
                angle_deg = math.degrees(base_angle + beam_idx * angle_inc)
                angle_deg = ((angle_deg + 180.0) % 360.0) - 180.0
                for idx in range(sector_count):
                    hi = edges[idx]
                    lo = edges[idx + 1]
                    if idx == 0:
                        in_sector = angle_deg <= hi and angle_deg >= lo
                    else:
                        in_sector = angle_deg < hi and angle_deg >= lo
                    if in_sector:
                        sector_values[idx].append(max(min(v, max_r), min_r))
                        break
        else:
            for idx in range(sector_count):
                start = int(round(idx * n / float(sector_count)))
                end = int(round((idx + 1) * n / float(sector_count)))
                if end <= start:
                    end = start + 1
                for value in raw[start:end]:
                    try:
                        v = float(value)
                    except Exception:
                        continue
                    if math.isnan(v) or math.isinf(v) or v < min_r:
                        continue
                    sector_values[idx].append(max(min(v, max_r), min_r))

    sector_ranges = []
    sector_valid = []
    for values in sector_values:
        if values:
            sector_ranges.append(round(float(quantile(values, 0.20)), 4))
            sector_valid.append(1.0)
        else:
            sector_ranges.append(round(max_r, 4))
            sector_valid.append(0.0)
    return {
        'sector_ranges': sector_ranges,
        'sector_valid': sector_valid,
    }


class RosLidarReader:
    """
    通过 ROS /scan 订阅 LaserScan，并将原始 scan 桥接回 Python3 主进程。

    运行时使用 system python + rospy 子进程，避免在 runtime_monitor.py
    里直接依赖 Python3 的 ROS 包环境。
    """

    def __init__(self, topic='/scan',
                 ros_setup='/opt/ros/melodic/setup.bash',
                 workspace_setup='~/catkin_ws/devel/setup.bash',
                 python_cmd='/usr/bin/python',
                 auto_start_driver=True,
                 driver_launch='jetracer lidar.launch',
                 driver_ready_timeout=12.0,
                 driver_log_dir='~/mycar/monitor_logs'):
        self.topic = topic
        self.ros_setup = ros_setup
        self.workspace_setup = workspace_setup
        self.python_cmd = python_cmd
        self.auto_start_driver = bool(auto_start_driver)
        self.driver_launch = str(driver_launch or '').strip()
        self.driver_ready_timeout = float(driver_ready_timeout)
        self.driver_log_dir = os.path.expanduser(driver_log_dir)
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._process = None
        self._driver_process = None
        self._driver_log_file = None
        self._driver_log_path = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._data = self._make_default_data()

    def _make_default_data(self):
        return {
            'frame_count': 0,
            'valid_points': 0,
            'points_total': 0,
            'last_update': 0.0,
            'scan_age_ms': -1.0,
            'nearest_min': -1.0,
            'parse_errors': 0,
            'angle_min': 0.0,
            'angle_max': 0.0,
            'angle_increment': 0.0,
            'range_min': 0.0,
            'range_max': 0.0,
            'ranges': [],
            'sector_ranges': [20.0] * 72,
            'sector_valid': [0.0] * 72,
            'intensities': [],
        }

    def _build_helper_script(self):
        return textwrap.dedent(f"""
            import json
            import math
            import sys
            import time
            import rospy
            from sensor_msgs.msg import LaserScan
            from std_srvs.srv import Empty

            TOPIC = {self.topic!r}

            def encode_array(values):
                encoded = []
                for value in values:
                    if math.isnan(value) or math.isinf(value):
                        encoded.append(None)
                    else:
                        encoded.append(round(float(value), 4))
                return encoded

            def sectorize_lidar_scan(ranges, angle_min=0.0, angle_increment=0.0,
                                     range_min=0.18, range_max=20.0, sectors=72,
                                     fov_deg=360.0):
                def quantile(values, frac):
                    ordered = sorted(values)
                    if len(ordered) == 1:
                        return ordered[0]
                    pos = (len(ordered) - 1) * float(frac)
                    lo = int(math.floor(pos))
                    hi = min(lo + 1, len(ordered) - 1)
                    weight = pos - lo
                    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

                sector_count = max(1, int(sectors or 1))
                max_r = float(range_max or 20.0)
                min_r = float(range_min or 0.18)
                sector_values = [[] for _ in range(sector_count)]
                raw = list(ranges or [])
                n = len(raw)
                angle_inc = float(angle_increment or 0.0)

                if n > 0:
                    if abs(angle_inc) > 1e-12:
                        half_fov = 0.5 * float(fov_deg or 360.0)
                        edges = [
                            half_fov - idx * float(fov_deg or 360.0) / float(sector_count)
                            for idx in range(sector_count + 1)
                        ]
                        base_angle = float(angle_min or 0.0)
                        for beam_idx, value in enumerate(raw):
                            try:
                                v = float(value)
                            except Exception:
                                continue
                            if math.isnan(v) or math.isinf(v) or v < min_r:
                                continue
                            angle_deg = math.degrees(base_angle + beam_idx * angle_inc)
                            angle_deg = ((angle_deg + 180.0) % 360.0) - 180.0
                            for idx in range(sector_count):
                                hi = edges[idx]
                                lo = edges[idx + 1]
                                if idx == 0:
                                    in_sector = angle_deg <= hi and angle_deg >= lo
                                else:
                                    in_sector = angle_deg < hi and angle_deg >= lo
                                if in_sector:
                                    sector_values[idx].append(max(min(v, max_r), min_r))
                                    break
                    else:
                        for idx in range(sector_count):
                            start = int(round(idx * n / float(sector_count)))
                            end = int(round((idx + 1) * n / float(sector_count)))
                            if end <= start:
                                end = start + 1
                            for value in raw[start:end]:
                                try:
                                    v = float(value)
                                except Exception:
                                    continue
                                if math.isnan(v) or math.isinf(v) or v < min_r:
                                    continue
                                sector_values[idx].append(max(min(v, max_r), min_r))

                sector_ranges = []
                sector_valid = []
                for values in sector_values:
                    if values:
                        sector_ranges.append(round(float(quantile(values, 0.20)), 4))
                        sector_valid.append(1.0)
                    else:
                        sector_ranges.append(round(max_r, 4))
                        sector_valid.append(0.0)
                return {{
                    'sector_ranges': sector_ranges,
                    'sector_valid': sector_valid,
                }}

            def emit_message(msg, frame_count):
                valid_ranges = [
                    float(r) for r in msg.ranges
                    if (not math.isnan(r)) and (not math.isinf(r)) and r >= msg.range_min and r <= msg.range_max
                ]

                stamp = msg.header.stamp.to_sec()
                if stamp <= 0:
                    stamp = time.time()

                frame_count += 1
                sectors = sectorize_lidar_scan(
                    ranges=msg.ranges,
                    angle_min=msg.angle_min,
                    angle_increment=msg.angle_increment,
                    range_min=msg.range_min,
                    range_max=msg.range_max,
                    sectors=72,
                )
                payload = {{
                    'frame_count': frame_count,
                    'stamp': stamp,
                    'valid_points': len(valid_ranges),
                    'points_total': len(msg.ranges),
                    'nearest_min': -1.0 if not valid_ranges else round(min(valid_ranges), 4),
                    'angle_min': round(float(msg.angle_min), 6),
                    'angle_max': round(float(msg.angle_max), 6),
                    'angle_increment': round(float(msg.angle_increment), 6),
                    'range_min': round(float(msg.range_min), 4),
                    'range_max': round(float(msg.range_max), 4),
                    'ranges': encode_array(msg.ranges),
                    'sector_ranges': sectors['sector_ranges'],
                    'sector_valid': sectors['sector_valid'],
                    'intensities': encode_array(msg.intensities) if msg.intensities else [],
                }}

                sys.stdout.write(json.dumps(payload, separators=(',', ':')) + '\\n')
                sys.stdout.flush()
                return frame_count

            rospy.init_node('runtime_monitor_lidar_bridge', anonymous=True, disable_signals=True)
            try:
                rospy.wait_for_service('/start_motor', timeout=2.0)
                start_motor = rospy.ServiceProxy('/start_motor', Empty)
                start_motor()
            except Exception:
                pass
            frame_count = 0
            while not rospy.is_shutdown():
                try:
                    msg = rospy.wait_for_message(TOPIC, LaserScan, timeout=2.0)
                    frame_count = emit_message(msg, frame_count)
                except rospy.ROSException:
                    continue
        """).strip()

    def _ros_command(self, body):
        return (
            f"source {shlex.quote(self.ros_setup)} >/dev/null 2>&1 && "
            f"if [ -f {self.workspace_setup} ]; then source {self.workspace_setup} >/dev/null 2>&1; fi && "
            f"{body}"
        )

    def _topic_available(self):
        command = self._ros_command("timeout 3 rostopic list")
        try:
            output = subprocess.check_output(
                ['/bin/bash', '-lc', command],
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                timeout=5.0,
            )
        except Exception:
            return False
        return any(line.strip() == self.topic for line in output.splitlines())

    def _start_driver_if_needed(self):
        if self._topic_available():
            print(f"   LiDAR topic 已可用: {self.topic}")
            return True
        if not self.auto_start_driver or not self.driver_launch:
            print(f"   LiDAR topic 暂不可用: {self.topic}")
            return True

        os.makedirs(self.driver_log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._driver_log_path = os.path.join(self.driver_log_dir, f'lidar_driver_{ts}.log')
        launch_parts = shlex.split(self.driver_launch)
        launch_cmd = "roslaunch " + " ".join(shlex.quote(part) for part in launch_parts)
        command = self._ros_command(launch_cmd)
        try:
            self._driver_log_file = open(self._driver_log_path, 'a')
            self._driver_process = subprocess.Popen(
                ['/bin/bash', '-lc', command],
                stdout=self._driver_log_file,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            print(f"   LiDAR driver 启动中: {self.driver_launch}")
            print(f"   LiDAR driver 日志: {self._driver_log_path}")
        except Exception as e:
            print(f"   LiDAR driver 启动失败: {e}")
            return False

        deadline = time.time() + max(0.0, self.driver_ready_timeout)
        while time.time() < deadline:
            if self._driver_process.poll() is not None:
                print(f"   LiDAR driver 已退出，查看日志: {self._driver_log_path}")
                return False
            if self._topic_available():
                print(f"   LiDAR topic ready: {self.topic}")
                return True
            time.sleep(0.5)
        print(f"   LiDAR topic 等待超时: {self.topic} ({self.driver_ready_timeout:.1f}s)")
        return False

    def start(self):
        if self._running:
            return True

        self._start_driver_if_needed()

        helper_script = self._build_helper_script()
        command = self._ros_command(f"{self.python_cmd} -u -c {shlex.quote(helper_script)}")

        try:
            self._process = subprocess.Popen(
                ['/bin/bash', '-lc', command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )
            self._running = True
            self._stdout_thread = threading.Thread(target=self._stdout_loop, daemon=True)
            self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
            self._stdout_thread.start()
            self._stderr_thread.start()
            self._connected = True
            print(f"   LiDAR ROS 桥接已启动: topic={self.topic} (raw /scan)")
            return True
        except Exception as e:
            print(f"   LiDAR ROS 桥接启动失败: {e}")
            self._connected = False
            self._running = False
            return False

    def stop(self):
        self._running = False
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except Exception:
                self._process.kill()
        if self._stdout_thread:
            self._stdout_thread.join(timeout=1.0)
        if self._stderr_thread:
            self._stderr_thread.join(timeout=1.0)
        if self._driver_process and self._driver_process.poll() is None:
            self._driver_process.terminate()
            try:
                self._driver_process.wait(timeout=3.0)
            except Exception:
                self._driver_process.kill()
        if self._driver_log_file:
            try:
                self._driver_log_file.close()
            except Exception:
                pass
        self._connected = False

    def _stdout_loop(self):
        try:
            for line in self._process.stdout:
                if not self._running:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    with self._lock:
                        self._data['parse_errors'] += 1
                    continue

                with self._lock:
                    self._data['frame_count'] = payload.get('frame_count', self._data['frame_count'])
                    self._data['valid_points'] = payload.get('valid_points', 0)
                    self._data['points_total'] = payload.get('points_total', 0)
                    self._data['nearest_min'] = payload.get('nearest_min', -1.0)
                    self._data['last_update'] = payload.get('stamp', time.time())
                    self._data['scan_age_ms'] = max(0.0, (time.time() - self._data['last_update']) * 1000.0)
                    self._data['angle_min'] = payload.get('angle_min', 0.0)
                    self._data['angle_max'] = payload.get('angle_max', 0.0)
                    self._data['angle_increment'] = payload.get('angle_increment', 0.0)
                    self._data['range_min'] = payload.get('range_min', 0.0)
                    self._data['range_max'] = payload.get('range_max', 0.0)
                    self._data['ranges'] = payload.get('ranges', [])
                    self._data['sector_ranges'] = payload.get('sector_ranges', [20.0] * 72)
                    self._data['sector_valid'] = payload.get('sector_valid', [0.0] * 72)
                    self._data['intensities'] = payload.get('intensities', [])
                    self._connected = True
        finally:
            self._connected = False

    def _stderr_loop(self):
        try:
            for line in self._process.stderr:
                if not self._running:
                    break
                text = line.strip()
                if text:
                    print(f"   [LiDAR ROS] {text}")
        finally:
            pass

    def get_data(self):
        with self._lock:
            data = dict(self._data)
            data['ranges'] = list(self._data['ranges'])
            data['sector_ranges'] = list(self._data['sector_ranges'])
            data['sector_valid'] = list(self._data['sector_valid'])
            data['intensities'] = list(self._data['intensities'])
            if data['last_update'] > 0:
                data['scan_age_ms'] = max(0.0, (time.time() - data['last_update']) * 1000.0)
            return data

    @property
    def is_connected(self):
        return self._connected and self._process is not None and self._process.poll() is None


# ============================================================
# 数据采集 Part
# ============================================================
class AsyncLogWriter:
    """Background CSV/JSONL writer for runtime monitor samples."""

    def __init__(self, csv_writer, csv_file, raw_file=None,
                 flush_every=5, flush_interval=2.0, max_queue=256):
        self.csv_writer = csv_writer
        self.csv_file = csv_file
        self.raw_file = raw_file
        self.flush_every = int(max(1, flush_every))
        self.flush_interval = float(max(0.1, flush_interval))
        self._queue = queue.Queue(maxsize=int(max(1, max_queue)))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._pending = 0
        self._last_flush = time.time()
        self.max_queue_size = int(max(1, max_queue))
        self.max_queue_depth = 0
        self.dropped_records = 0
        self.records_written = 0
        self.raw_records_written = 0
        self._stats_lock = threading.Lock()
        self._thread.start()

    def _note_queue_depth(self, depth):
        with self._stats_lock:
            if int(depth) > self.max_queue_depth:
                self.max_queue_depth = int(depth)

    def stats(self):
        with self._stats_lock:
            return {
                'queue_depth': int(self._queue.qsize()),
                'max_queue_depth': int(self.max_queue_depth),
                'max_queue_size': int(self.max_queue_size),
                'dropped_records': int(self.dropped_records),
                'records_written': int(self.records_written),
                'raw_records_written': int(self.raw_records_written),
            }

    def write(self, row, raw_record=None):
        item = (row, raw_record)
        try:
            self._queue.put_nowait(item)
            self._note_queue_depth(self._queue.qsize())
        except queue.Full:
            # Preserve control-loop liveness under logging pressure. Dropping a
            # debug sample is preferable to blocking the vehicle loop.
            with self._stats_lock:
                self.dropped_records += 1
                self.max_queue_depth = max(self.max_queue_depth, self.max_queue_size)

    def _write_item(self, row, raw_record):
        self.csv_writer.writerow(row)
        if self.raw_file is not None and raw_record is not None:
            self.raw_file.write(json.dumps(raw_record, separators=(',', ':')) + '\n')
        with self._stats_lock:
            self.records_written += 1
            if self.raw_file is not None and raw_record is not None:
                self.raw_records_written += 1
        self._pending += 1

    def _flush_if_needed(self, force=False):
        now = time.time()
        if not force and self._pending < self.flush_every and now - self._last_flush < self.flush_interval:
            return
        self.csv_file.flush()
        if self.raw_file is not None:
            self.raw_file.flush()
        self._pending = 0
        self._last_flush = now

    def _run(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                row, raw_record = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._flush_if_needed()
                continue
            try:
                self._write_item(row, raw_record)
                self._flush_if_needed()
            finally:
                self._queue.task_done()
        self._flush_if_needed(force=True)

    def close(self):
        self._queue.join()
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._flush_if_needed(force=True)


class AsyncTelemetryCache:
    """Refresh slow Jetson telemetry in the background."""

    def __init__(self, read_fn, default_data=None, interval=1.0):
        self.read_fn = read_fn
        self.interval = float(max(0.1, interval))
        self._data = dict(default_data or {})
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.last_error = None

    def start(self):
        self._thread.start()

    def get(self):
        with self._lock:
            return dict(self._data)

    def _refresh_once(self):
        try:
            data = self.read_fn()
            if data:
                with self._lock:
                    self._data = dict(data)
            self.last_error = None
        except Exception as exc:
            self.last_error = exc

    def _run(self):
        while not self._stop.is_set():
            self._refresh_once()
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


class DataCollector:
    """
    DonkeyCar Vehicle Part —— 运行时数据采集器

    挂载到 Vehicle 主循环，每帧读取 Memory 中的数据，
    按间隔写入 CSV 文件。
    """

    # 定义要采集的 DonkeyCar 内存通道（按顺序）
    # 注: IMU/编码器数据来自 RP2040 串口，不通过 DonkeyCar 内存
    INPUT_KEYS = [
        # 核心数据（必需）
        'cam/image_array',      # 摄像头图像
        'user/angle',           # 用户转向
        'user/throttle',        # 用户油门
        'pilot/angle',          # AI 转向
        'pilot/throttle',       # AI 油门
        'pilot/inference_latency_ms',  # shadow pilot end-to-end latency
        'pilot/preprocess_latency_ms', # shadow observation preprocessing latency
        'pilot/raw_angle',      # raw high-level policy action[0]
        'pilot/raw_throttle',   # raw high-level policy action[1]
        'angle',                # 最终执行转向
        'throttle',             # 最终执行油门
        'user/mode',            # 驾驶模式
        'recording',            # 是否录制
        'run_pilot',            # AI 是否激活
        'tub/num_records',      # 已录制条数
        'safety/blocked',
        'safety/block_reason',
        'safety/inference_timeout_count',
        'safety/lidar_missing_count',
        'safety/lidar_stale_count',
        'safety/rp2040_missing_count',
        'safety/last_lidar_age_ms',
        'safety/last_rp2040_age_ms',
    ]

    BASE_CSV_FIELDS = [
        # 基础信息
        'timestamp', 'sample_id', 'frame', 'elapsed_sec', 'loop_dt_ms', 'effective_fps',
        # deployment metadata
        'control_mode', 'model_name', 'model_path', 'model_size_MB',
        'backend', 'input_resolution', 'input_modalities', 'policy_chain',
        'track_condition', 'run_label',
        # 驾驶模式 & 状态
        'mode', 'recording', 'run_pilot', 'tub_records',
        # deployment safety monitor
        'safety_blocked', 'safety_block_reason',
        'safety_inference_timeout_count',
        'safety_lidar_missing_count', 'safety_lidar_stale_count',
        'safety_rp2040_missing_count',
        'safety_last_lidar_age_ms', 'safety_last_rp2040_age_ms',
        # shadow actuator route evidence
        'actual_actuator_source', 'v17_output_route',
        'shadow_action_angle', 'shadow_action_throttle',
        'vehicle_action_angle', 'vehicle_action_throttle',
        'shadow_non_takeover',
        # 用户输入
        'user_angle', 'user_throttle',
        # AI 输出
        'pilot_angle', 'pilot_throttle',
        'pilot_inference_latency_ms', 'pilot_preprocess_latency_ms',
        'pilot_raw_angle', 'pilot_raw_throttle',
        # 最终执行
        'final_angle', 'final_throttle',
        # 转向差异（AI vs 用户）
        'angle_diff',
        # 图像统计
        'img_mean', 'img_std', 'img_brightness', 'img_contrast',
        # ── RP2040 传感器 (串口直读) ──
        # IMU 陀螺仪 (rad/s)
        'gyro_x', 'gyro_y', 'gyro_z',
        # IMU 加速度 (m/s^2)
        'accel_x', 'accel_y', 'accel_z',
        # IMU 欧拉角 (度)
        'euler_roll', 'euler_pitch', 'euler_yaw',
        # 里程计位置 (m)
        'odom_x', 'odom_y', 'odom_yaw',
        # 里程计增量
        'delta_x', 'delta_y', 'delta_yaw',
        # 电机编码速度 (脉冲/20ms)
        'motor_lvel', 'motor_rvel', 'motor_lset', 'motor_rset',
        # RP2040 帧计数 & 解析错误
        'rp2040_frames', 'rp2040_errors',
        # ── Jetson 系统 ──
        # 温度传感器 (7个温区)
        'cpu_temp', 'gpu_temp', 'ao_temp', 'pll_temp', 'fan_temp',
        'pmic_temp', 'wifi_temp',
        # 负载
        'gpu_load_pct', 'cpu_load_pct',
        # CPU/GPU 频率
        'cpu_freq_mhz', 'gpu_freq_mhz',
        # 内存 & 交换
        'mem_used_mb', 'mem_total_mb', 'mem_used_pct',
        'swap_used_mb', 'swap_total_mb',
        'process_rss_mb',
        # Async DataCollector writer health
        'async_queue_depth', 'async_queue_max_depth',
        'async_writer_backlog', 'async_writer_max_backlog',
        'async_writer_dropped_records',
        'async_writer_records_written', 'async_writer_raw_records_written',
        # 磁盘
        'disk_used_pct',
        # 风扇
        'fan_pwm',
        # WiFi 信号
        'wifi_rssi_dbm', 'wifi_link_quality',
        # ── 电源监控 ──
        # INA219 电池 (I2C 0x41)
        'battery_voltage_v', 'battery_current_ma',
        # INA3221 Jetson 电源轨 (mW)
        'power_in_mw', 'power_gpu_mw', 'power_cpu_mw',
    ]

    def __init__(self, log_path, log_interval=0.5, serial_reader=None,
                 lidar_reader=None, metadata=None):
        self.log_path = log_path
        self.log_interval = log_interval
        self.last_log_time = 0
        self.frame_count = 0
        self.sample_count = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.serial_reader = serial_reader  # RP2040SerialReader 实例
        self.lidar_reader = lidar_reader
        self.metadata = dict(metadata or {})
        self.process_rss_mb_start = self._read_process_rss_mb()
        self.process_rss_mb_max = self.process_rss_mb_start
        self.async_writer_stats_path = None
        self.lidar_fields = []
        self.lidar_raw_path = None
        self.lidar_raw_file = None
        if lidar_reader:
            self.lidar_fields = [
                'lidar_frames', 'lidar_valid_points', 'lidar_scan_age_ms',
                'lidar_points_total', 'lidar_nearest_min',
                'lidar_angle_min_rad', 'lidar_angle_max_rad',
                'lidar_angle_increment_rad',
                'lidar_range_min', 'lidar_range_max',
            ]
        self.csv_fields = list(self.BASE_CSV_FIELDS) + self.lidar_fields

        # 打开 CSV
        self.csv_file = open(self.log_path, 'w', newline='')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fields)
        self.writer.writeheader()
        self.csv_file.flush()
        if lidar_reader:
            log_stem, _ = os.path.splitext(self.log_path)
            self.lidar_raw_path = f'{log_stem}_lidar_raw.jsonl'
            self.lidar_raw_file = open(self.lidar_raw_path, 'w')
        self.async_writer = AsyncLogWriter(
            csv_writer=self.writer,
            csv_file=self.csv_file,
            raw_file=self.lidar_raw_file,
            flush_every=5,
            flush_interval=2.0,
            max_queue=512,
        )

        # 用于计算 CPU 负载的上一次 /proc/stat 快照
        self._prev_cpu_stat = None
        self._telemetry_cache = AsyncTelemetryCache(
            read_fn=self._read_thermal_zones_sync,
            default_data=self._default_thermal_zones(),
            interval=1.0,
        )
        self._telemetry_cache.start()
        # Vehicle 引用（稍后由 monkey-patch 设置）
        self.vehicle = None

        print(f"\n{'='*60}")
        print(f"DataCollector 已启动")
        print(f"   日志文件: {self.log_path}")
        print(f"   采样间隔: {self.log_interval}s")
        print(f"   DonkeyCar 通道: {len(self.INPUT_KEYS)} 个")
        print(f"   RP2040 串口: {'已连接' if serial_reader and serial_reader.is_connected else '未连接'}")
        print(f"   LiDAR ROS: {'已连接' if lidar_reader and lidar_reader.is_connected else '未连接'}")
        if self.lidar_raw_path:
            print(f"   LiDAR Raw: {self.lidar_raw_path}")
        print(f"   CSV 字段: {len(self.csv_fields)} 列")
        print(f"{'='*60}\n")

    # ---- Jetson 系统传感器 ----

    def _read_cpu_load(self):
        """通过两次 /proc/stat 采样计算 CPU 综合负载率（%）"""
        try:
            with open('/proc/stat') as f:
                line = f.readline()   # cpu  user nice system idle iowait ...
            vals = list(map(int, line.split()[1:]))
            total = sum(vals)
            idle  = vals[3]
            if self._prev_cpu_stat is None:
                self._prev_cpu_stat = (total, idle)
                return -1.0
            prev_total, prev_idle = self._prev_cpu_stat
            self._prev_cpu_stat = (total, idle)
            d_total = total - prev_total
            d_idle  = idle  - prev_idle
            return round((1.0 - d_idle / d_total) * 100.0, 1) if d_total > 0 else -1.0
        except Exception:
            return -1.0

    def _read_mem(self):
        """读取 /proc/meminfo 返回 (used_mb, total_mb, used_pct, swap_used_mb, swap_total_mb)"""
        try:
            info = {}
            with open('/proc/meminfo') as f:
                for line in f:
                    k, v = line.split(':', 1)
                    info[k.strip()] = int(v.split()[0])   # kB
            total = info.get('MemTotal', 0)
            avail = info.get('MemAvailable', info.get('MemFree', 0))
            used  = total - avail
            pct   = round(used / total * 100.0, 1) if total > 0 else -1.0
            swap_total = info.get('SwapTotal', 0)
            swap_free  = info.get('SwapFree', 0)
            swap_used  = swap_total - swap_free
            return (round(used / 1024, 1), round(total / 1024, 1), pct,
                    round(swap_used / 1024, 1), round(swap_total / 1024, 1))
        except Exception:
            return -1.0, -1.0, -1.0, -1.0, -1.0

    def _read_process_rss_mb(self):
        """Read this process RSS from /proc/self/status without psutil."""
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        parts = line.split()
                        if len(parts) >= 2:
                            return round(int(parts[1]) / 1024.0, 3)
        except Exception:
            pass
        return -1.0

    def _read_wifi(self):
        """读取 WiFi RSSI 和链路质量，返回 (rssi_dbm, link_quality)"""
        try:
            with open('/proc/net/wireless') as f:
                lines = f.readlines()
            if len(lines) >= 3:
                parts = lines[2].split()
                link_quality = float(parts[2].rstrip('.'))
                rssi_dbm = float(parts[3].rstrip('.'))
                return rssi_dbm, link_quality
        except Exception:
            pass
        return -999.0, -1.0

    def _read_cpu_freq(self):
        """读取 CPU 平均频率 (MHz)"""
        try:
            freqs = []
            for i in range(4):
                with open(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq') as f:
                    freqs.append(int(f.read().strip()) / 1000.0)  # kHz -> MHz
            return round(sum(freqs) / len(freqs), 0) if freqs else -1.0
        except Exception:
            return -1.0

    def _read_gpu_freq(self):
        """读取 GPU 当前频率 (MHz)"""
        try:
            with open('/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq') as f:
                return round(int(f.read().strip()) / 1e6, 0)  # Hz -> MHz
        except Exception:
            return -1.0

    def _read_fan_pwm(self):
        """读取风扇 PWM 值 (0-255)"""
        try:
            with open('/sys/devices/pwm-fan/target_pwm') as f:
                return int(f.read().strip())
        except Exception:
            return -1

    def _read_disk_usage(self):
        """读取根分区磁盘使用率 (%)"""
        try:
            st = os.statvfs('/')
            total = st.f_blocks * st.f_frsize
            free  = st.f_bavail * st.f_frsize
            used_pct = round((1.0 - free / total) * 100.0, 1) if total > 0 else -1.0
            return used_pct
        except Exception:
            return -1.0

    def _read_ina219(self):
        """读取 INA219 电池电压和电流 (I2C bus 1, addr 0x41)"""
        try:
            if not hasattr(self, '_smbus'):
                import smbus
                self._smbus = smbus.SMBus(1)
            bus = self._smbus
            # Bus voltage register (0x02) - big-endian
            raw = bus.read_word_data(0x41, 0x02)
            raw = ((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF)
            voltage = (raw >> 3) * 0.004  # LSB = 4mV
            # Shunt voltage register (0x01)
            raw = bus.read_word_data(0x41, 0x01)
            raw = ((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF)
            if raw & 0x8000:
                raw -= 65536
            shunt_mv = raw * 0.01  # LSB = 10uV
            # 假设 0.1 ohm 分流电阻 (常见配置)
            current_ma = shunt_mv / 0.1
            return round(voltage, 3), round(current_ma, 1)
        except Exception:
            return -1.0, -1.0

    def _read_ina3221(self):
        """读取 INA3221 Jetson 电源轨功率 (mW)，需要 sudo 权限"""
        try:
            if not hasattr(self, '_ina3221_ok'):
                self._ina3221_ok = os.access(
                    '/sys/bus/iio/devices/iio:device0/in_power0_input', os.R_OK)
            if not self._ina3221_ok:
                return -1.0, -1.0, -1.0
            powers = []
            for i in range(3):
                with open(f'/sys/bus/iio/devices/iio:device0/in_power{i}_input') as f:
                    powers.append(int(f.read().strip()))
            return powers[0], powers[1], powers[2]  # IN, GPU, CPU
        except Exception:
            return -1.0, -1.0, -1.0

    def _default_thermal_zones(self):
        """Default telemetry values used before the first background refresh."""
        return {
            'ao_temp': -1.0,
            'cpu_temp': -1.0,
            'gpu_temp': -1.0,
            'pll_temp': -1.0,
            'pmic_temp': -1.0,
            'fan_temp': -1.0,
            'wifi_temp': -1.0,
            'gpu_load_pct': -1.0,
            'cpu_load_pct': -1.0,
            'cpu_freq_mhz': -1.0,
            'gpu_freq_mhz': -1.0,
            'mem_used_mb': -1.0,
            'mem_total_mb': -1.0,
            'mem_used_pct': -1.0,
            'swap_used_mb': -1.0,
            'swap_total_mb': -1.0,
            'disk_used_pct': -1.0,
            'fan_pwm': -1,
            'wifi_rssi_dbm': -999.0,
            'wifi_link_quality': -1.0,
            'battery_voltage_v': -1.0,
            'battery_current_ma': -1.0,
            'power_in_mw': -1.0,
            'power_gpu_mw': -1.0,
            'power_cpu_mw': -1.0,
        }

    def _read_thermal_zones_sync(self):
        """批量读取 Jetson 温度传感器、GPU/CPU 负载、内存、电源等。"""
        zones = {
            'ao_temp':   0,   # AO-therm
            'cpu_temp':  1,   # CPU-therm
            'gpu_temp':  2,   # GPU-therm
            'pll_temp':  3,   # PLL-therm
            'pmic_temp': 4,   # PMIC-Die
            'fan_temp':  5,   # thermal-fan-est
            'wifi_temp': 6,   # iwlwifi
        }
        result = {}
        for name, zone_id in zones.items():
            try:
                with open(f'/sys/class/thermal/thermal_zone{zone_id}/temp') as f:
                    result[name] = int(f.read().strip()) / 1000.0
            except Exception:
                result[name] = -1.0

        # GPU load (%)
        try:
            with open('/sys/devices/gpu.0/load') as f:
                result['gpu_load_pct'] = int(f.read().strip()) / 10.0
        except Exception:
            result['gpu_load_pct'] = -1.0

        # CPU load (%)
        result['cpu_load_pct'] = self._read_cpu_load()

        # CPU/GPU 频率
        result['cpu_freq_mhz'] = self._read_cpu_freq()
        result['gpu_freq_mhz'] = self._read_gpu_freq()

        # 内存 + 交换
        (result['mem_used_mb'], result['mem_total_mb'], result['mem_used_pct'],
         result['swap_used_mb'], result['swap_total_mb']) = self._read_mem()

        # 磁盘
        result['disk_used_pct'] = self._read_disk_usage()

        # 风扇
        result['fan_pwm'] = self._read_fan_pwm()

        # WiFi
        result['wifi_rssi_dbm'], result['wifi_link_quality'] = self._read_wifi()

        # INA219 电池
        result['battery_voltage_v'], result['battery_current_ma'] = self._read_ina219()

        # INA3221 电源轨
        result['power_in_mw'], result['power_gpu_mw'], result['power_cpu_mw'] = self._read_ina3221()

        return result

    def _read_thermal_zones(self):
        """Return the latest background telemetry snapshot without blocking the loop."""
        if hasattr(self, '_telemetry_cache') and self._telemetry_cache:
            return self._telemetry_cache.get()
        return self._read_thermal_zones_sync()

    # ---- DonkeyCar Part 接口 ----

    def run(self, img_arr, user_angle, user_throttle,
            pilot_angle, pilot_throttle,
            pilot_inference_latency_ms, pilot_preprocess_latency_ms,
            pilot_raw_angle, pilot_raw_throttle,
            final_angle, final_throttle,
            mode, recording, run_pilot, tub_records,
            safety_blocked, safety_block_reason,
            safety_inference_timeout_count,
            safety_lidar_missing_count, safety_lidar_stale_count,
            safety_rp2040_missing_count,
            safety_last_lidar_age_ms, safety_last_rp2040_age_ms):
        """Vehicle 每帧调用一次

        参数对应 INPUT_KEYS 的顺序（11个核心数据）。
        IMU/编码器/电机数据从 RP2040 串口线程获取，不经过 DonkeyCar 内存。
        """
        
        now = time.time()
        self.frame_count += 1
        loop_dt = now - self.prev_time
        self.prev_time = now
        elapsed = now - self.start_time

        # 按间隔采样写入
        if now - self.last_log_time < self.log_interval:
            return
        self.last_log_time = now
        self.sample_count += 1
        
        # -- 图像统计 --
        img_mean = img_std = img_brightness = img_contrast = 0.0
        if img_arr is not None:
            flat = img_arr.astype(np.float32)
            img_mean = float(np.mean(flat))
            img_std = float(np.std(flat))
            # 亮度 = RGB 平均（感知亮度的近似）
            img_brightness = float(np.mean(flat[:, :, :3]))
            # 对比度 = 亮度通道标准差
            gray = 0.299 * flat[:, :, 0] + 0.587 * flat[:, :, 1] + 0.114 * flat[:, :, 2]
            img_contrast = float(np.std(gray))
        
        # -- RP2040 串口传感器数据 --
        if self.serial_reader and self.serial_reader.is_connected:
            sd = self.serial_reader.get_data()
        else:
            sd = {}  # 串口未连接时全部为 0

        # -- LiDAR ROS 数据 --
        if self.lidar_reader and self.lidar_reader.is_connected:
            ld = self.lidar_reader.get_data()
        else:
            ld = {}

        # -- 系统数据 --
        temps = self._read_thermal_zones()
        process_rss_mb = self._read_process_rss_mb()
        if process_rss_mb >= 0:
            if self.process_rss_mb_max is None or self.process_rss_mb_max < 0:
                self.process_rss_mb_max = process_rss_mb
            else:
                self.process_rss_mb_max = max(self.process_rss_mb_max, process_rss_mb)

        # -- 计算值 --
        angle_diff = 0.0
        if pilot_angle is not None and user_angle is not None:
            angle_diff = (pilot_angle or 0) - (user_angle or 0)

        effective_fps = self.frame_count / elapsed if elapsed > 0 else 0
        control_mode = self.metadata.get('control_mode', 'normal')
        mode_text = mode or 'unknown'
        run_pilot_bool = bool(run_pilot)
        shadow_action_angle = float(pilot_angle or 0.0)
        shadow_action_throttle = float(pilot_throttle or 0.0)
        vehicle_action_angle = float(final_angle or 0.0)
        vehicle_action_throttle = float(final_throttle or 0.0)
        if control_mode == 'shadow':
            actual_actuator_source = 'user/manual'
            v17_output_route = 'shadow_only'
            shadow_non_takeover = not run_pilot_bool
        elif control_mode == 'active':
            actual_actuator_source = 'v17_active'
            v17_output_route = 'active_actuator'
            shadow_non_takeover = False
        else:
            actual_actuator_source = 'donkeycar_default'
            v17_output_route = 'normal'
            shadow_non_takeover = False

        # -- 写入 CSV --
        row = {
            'timestamp':       f'{now:.3f}',
            'sample_id':       self.sample_count,
            'frame':           self.frame_count,
            'elapsed_sec':     f'{elapsed:.2f}',
            'loop_dt_ms':      f'{loop_dt * 1000:.1f}',
            'effective_fps':   f'{effective_fps:.1f}',
            'control_mode':    control_mode,
            'model_name':      self.metadata.get('model_name', ''),
            'model_path':      self.metadata.get('model_path', ''),
            'model_size_MB':   self.metadata.get('model_size_MB', ''),
            'backend':         self.metadata.get('backend', ''),
            'input_resolution': self.metadata.get('input_resolution', ''),
            'input_modalities': self.metadata.get('input_modalities', ''),
            'policy_chain':    self.metadata.get('policy_chain', ''),
            'track_condition': self.metadata.get('track_condition', ''),
            'run_label':       self.metadata.get('run_label', ''),
            'mode':            mode_text,
            'recording':       bool(recording),
            'run_pilot':       run_pilot_bool,
            'tub_records':     tub_records or 0,
            'safety_blocked':   bool(safety_blocked),
            'safety_block_reason': safety_block_reason or '',
            'safety_inference_timeout_count': int(safety_inference_timeout_count or 0),
            'safety_lidar_missing_count': int(safety_lidar_missing_count or 0),
            'safety_lidar_stale_count': int(safety_lidar_stale_count or 0),
            'safety_rp2040_missing_count': int(safety_rp2040_missing_count or 0),
            'safety_last_lidar_age_ms': f'{safety_last_lidar_age_ms if safety_last_lidar_age_ms is not None else -1:.1f}',
            'safety_last_rp2040_age_ms': f'{safety_last_rp2040_age_ms if safety_last_rp2040_age_ms is not None else -1:.1f}',
            'actual_actuator_source': actual_actuator_source,
            'v17_output_route': v17_output_route,
            'shadow_action_angle': f'{shadow_action_angle:.4f}',
            'shadow_action_throttle': f'{shadow_action_throttle:.4f}',
            'vehicle_action_angle': f'{vehicle_action_angle:.4f}',
            'vehicle_action_throttle': f'{vehicle_action_throttle:.4f}',
            'shadow_non_takeover': bool(shadow_non_takeover),
            'user_angle':      f'{user_angle or 0:.4f}',
            'user_throttle':   f'{user_throttle or 0:.4f}',
            'pilot_angle':     f'{pilot_angle or 0:.4f}',
            'pilot_throttle':  f'{pilot_throttle or 0:.4f}',
            'pilot_inference_latency_ms': f'{pilot_inference_latency_ms if pilot_inference_latency_ms is not None else -1:.3f}',
            'pilot_preprocess_latency_ms': f'{pilot_preprocess_latency_ms if pilot_preprocess_latency_ms is not None else -1:.3f}',
            'pilot_raw_angle': f'{pilot_raw_angle if pilot_raw_angle is not None else 0:.4f}',
            'pilot_raw_throttle': f'{pilot_raw_throttle if pilot_raw_throttle is not None else 0:.4f}',
            'final_angle':     f'{final_angle or 0:.4f}',
            'final_throttle':  f'{final_throttle or 0:.4f}',
            'angle_diff':      f'{angle_diff:.4f}',
            'img_mean':        f'{img_mean:.2f}',
            'img_std':         f'{img_std:.2f}',
            'img_brightness':  f'{img_brightness:.2f}',
            'img_contrast':    f'{img_contrast:.2f}',
            # RP2040 传感器
            'gyro_x':          f"{sd.get('gyro_x', 0):.4f}",
            'gyro_y':          f"{sd.get('gyro_y', 0):.4f}",
            'gyro_z':          f"{sd.get('gyro_z', 0):.4f}",
            'accel_x':         f"{sd.get('accel_x', 0):.4f}",
            'accel_y':         f"{sd.get('accel_y', 0):.4f}",
            'accel_z':         f"{sd.get('accel_z', 0):.4f}",
            'euler_roll':      f"{sd.get('euler_roll', 0):.2f}",
            'euler_pitch':     f"{sd.get('euler_pitch', 0):.2f}",
            'euler_yaw':       f"{sd.get('euler_yaw', 0):.2f}",
            'odom_x':          f"{sd.get('odom_x', 0):.4f}",
            'odom_y':          f"{sd.get('odom_y', 0):.4f}",
            'odom_yaw':        f"{sd.get('odom_yaw', 0):.4f}",
            'delta_x':         f"{sd.get('delta_x', 0):.4f}",
            'delta_y':         f"{sd.get('delta_y', 0):.4f}",
            'delta_yaw':       f"{sd.get('delta_yaw', 0):.4f}",
            'motor_lvel':      sd.get('motor_lvel', 0),
            'motor_rvel':      sd.get('motor_rvel', 0),
            'motor_lset':      sd.get('motor_lset', 0),
            'motor_rset':      sd.get('motor_rset', 0),
            'rp2040_frames':   sd.get('frame_count', 0),
            'rp2040_errors':   sd.get('parse_errors', 0),
            'cpu_temp':        f'{temps["cpu_temp"]:.1f}',
            'gpu_temp':        f'{temps["gpu_temp"]:.1f}',
            'ao_temp':         f'{temps["ao_temp"]:.1f}',
            'pll_temp':        f'{temps["pll_temp"]:.1f}',
            'fan_temp':        f'{temps["fan_temp"]:.1f}',
            'pmic_temp':       f'{temps["pmic_temp"]:.1f}',
            'wifi_temp':       f'{temps["wifi_temp"]:.1f}',
            'gpu_load_pct':    f'{temps["gpu_load_pct"]:.1f}',
            'cpu_load_pct':    f'{temps["cpu_load_pct"]:.1f}',
            'cpu_freq_mhz':    f'{temps["cpu_freq_mhz"]:.0f}',
            'gpu_freq_mhz':    f'{temps["gpu_freq_mhz"]:.0f}',
            'mem_used_mb':     f'{temps["mem_used_mb"]:.0f}',
            'mem_total_mb':    f'{temps["mem_total_mb"]:.0f}',
            'mem_used_pct':    f'{temps["mem_used_pct"]:.1f}',
            'swap_used_mb':    f'{temps["swap_used_mb"]:.0f}',
            'swap_total_mb':   f'{temps["swap_total_mb"]:.0f}',
            'process_rss_mb':  f'{process_rss_mb:.3f}',
            'async_queue_depth': 0,
            'async_queue_max_depth': 0,
            'async_writer_backlog': 0,
            'async_writer_max_backlog': 0,
            'async_writer_dropped_records': 0,
            'async_writer_records_written': 0,
            'async_writer_raw_records_written': 0,
            'disk_used_pct':   f'{temps["disk_used_pct"]:.1f}',
            'fan_pwm':         temps['fan_pwm'],
            'wifi_rssi_dbm':   f'{temps["wifi_rssi_dbm"]:.0f}',
            'wifi_link_quality': f'{temps["wifi_link_quality"]:.0f}',
            'battery_voltage_v':  f'{temps["battery_voltage_v"]:.3f}',
            'battery_current_ma': f'{temps["battery_current_ma"]:.1f}',
            'power_in_mw':     f'{temps["power_in_mw"]:.0f}',
            'power_gpu_mw':    f'{temps["power_gpu_mw"]:.0f}',
            'power_cpu_mw':    f'{temps["power_cpu_mw"]:.0f}',
        }
        if self.lidar_reader:
            row.update({
                'lidar_frames':        ld.get('frame_count', 0),
                'lidar_valid_points':  ld.get('valid_points', 0),
                'lidar_points_total':  ld.get('points_total', 0),
                'lidar_scan_age_ms':   f"{ld.get('scan_age_ms', -1.0):.1f}",
                'lidar_nearest_min':   f"{ld.get('nearest_min', -1.0):.4f}",
                'lidar_angle_min_rad': f"{ld.get('angle_min', 0.0):.6f}",
                'lidar_angle_max_rad': f"{ld.get('angle_max', 0.0):.6f}",
                'lidar_angle_increment_rad': f"{ld.get('angle_increment', 0.0):.6f}",
                'lidar_range_min':     f"{ld.get('range_min', 0.0):.4f}",
                'lidar_range_max':     f"{ld.get('range_max', 0.0):.4f}",
            })
        raw_record = None
        if self.lidar_raw_file and ld.get('frame_count', 0) > 0:
            tub_record_index_est = -1
            if recording and tub_records:
                tub_record_index_est = max(int(tub_records) - 1, 0)
            raw_record = {
                'timestamp': round(now, 3),
                'sample_id': self.sample_count,
                'frame': self.frame_count,
                'elapsed_sec': round(elapsed, 3),
                'control_mode': self.metadata.get('control_mode', 'normal'),
                'track_condition': self.metadata.get('track_condition', ''),
                'run_label': self.metadata.get('run_label', ''),
                'mode': mode or 'unknown',
                'recording': bool(recording),
                'run_pilot': bool(run_pilot),
                'tub_records': int(tub_records or 0),
                'safety': {
                    'blocked': bool(safety_blocked),
                    'block_reason': safety_block_reason or '',
                    'inference_timeout_count': int(safety_inference_timeout_count or 0),
                    'lidar_missing_count': int(safety_lidar_missing_count or 0),
                    'lidar_stale_count': int(safety_lidar_stale_count or 0),
                    'rp2040_missing_count': int(safety_rp2040_missing_count or 0),
                    'last_lidar_age_ms': float(safety_last_lidar_age_ms if safety_last_lidar_age_ms is not None else -1.0),
                    'last_rp2040_age_ms': float(safety_last_rp2040_age_ms if safety_last_rp2040_age_ms is not None else -1.0),
                },
                'tub_record_index_est': tub_record_index_est,
                'user_angle': float(user_angle or 0.0),
                'user_throttle': float(user_throttle or 0.0),
                'pilot_angle': float(pilot_angle or 0.0),
                'pilot_throttle': float(pilot_throttle or 0.0),
                'pilot_inference_latency_ms': float(pilot_inference_latency_ms or -1.0),
                'pilot_preprocess_latency_ms': float(pilot_preprocess_latency_ms or -1.0),
                'pilot_raw_angle': float(pilot_raw_angle or 0.0),
                'pilot_raw_throttle': float(pilot_raw_throttle or 0.0),
                'final_angle': float(final_angle or 0.0),
                'final_throttle': float(final_throttle or 0.0),
                'actuator_route': {
                    'actual_actuator_source': actual_actuator_source,
                    'v17_output_route': v17_output_route,
                    'shadow_action_angle': shadow_action_angle,
                    'shadow_action_throttle': shadow_action_throttle,
                    'vehicle_action_angle': vehicle_action_angle,
                    'vehicle_action_throttle': vehicle_action_throttle,
                    'shadow_non_takeover': bool(shadow_non_takeover),
                },
                'lidar': {
                    'frame_count': ld.get('frame_count', 0),
                    'stamp': ld.get('last_update', 0.0),
                    'scan_age_ms': round(ld.get('scan_age_ms', -1.0), 1),
                    'valid_points': ld.get('valid_points', 0),
                    'points_total': ld.get('points_total', 0),
                    'nearest_min': ld.get('nearest_min', -1.0),
                    'angle_min': ld.get('angle_min', 0.0),
                    'angle_max': ld.get('angle_max', 0.0),
                    'angle_increment': ld.get('angle_increment', 0.0),
                    'range_min': ld.get('range_min', 0.0),
                    'range_max': ld.get('range_max', 0.0),
                    'ranges': ld.get('ranges', []),
                    'intensities': ld.get('intensities', []),
                },
            }
        async_stats = self.async_writer.stats()
        predicted_queue_depth = min(
            int(async_stats.get('max_queue_size', 0) or 0),
            int(async_stats.get('queue_depth', 0) or 0) + 1,
        )
        predicted_max_queue_depth = max(
            int(async_stats.get('max_queue_depth', 0) or 0),
            predicted_queue_depth,
        )
        row.update({
            'async_queue_depth': predicted_queue_depth,
            'async_queue_max_depth': predicted_max_queue_depth,
            'async_writer_backlog': predicted_queue_depth,
            'async_writer_max_backlog': predicted_max_queue_depth,
            'async_writer_dropped_records': int(async_stats.get('dropped_records', 0) or 0),
            'async_writer_records_written': int(async_stats.get('records_written', 0) or 0),
            'async_writer_raw_records_written': int(async_stats.get('raw_records_written', 0) or 0),
        })
        if raw_record is not None:
            raw_record['async_writer'] = {
                'queue_depth': predicted_queue_depth,
                'max_queue_depth': predicted_max_queue_depth,
                'dropped_records': int(async_stats.get('dropped_records', 0) or 0),
                'records_written': int(async_stats.get('records_written', 0) or 0),
                'raw_records_written': int(async_stats.get('raw_records_written', 0) or 0),
            }
            raw_record['process_rss_mb'] = process_rss_mb
        self.async_writer.write(row, raw_record)

        # 终端摘要（每60条打印一次）
        if self.frame_count % int(60 / max(self.log_interval, 0.01)) == 0:
            print(f"\n[{elapsed:.0f}s] frame={self.frame_count} fps={effective_fps:.1f} mode={mode}")
            print(f"   steer={final_angle or 0:+.3f}  thr={final_throttle or 0:.3f}  "
                  f"(pilot steer={pilot_angle or 0:+.3f} thr={pilot_throttle or 0:.3f})  "
                  f"diff={angle_diff:+.3f}")
            print(f"   CPU={temps['cpu_temp']:.0f}C  GPU={temps['gpu_temp']:.0f}C  "
                  f"AO={temps['ao_temp']:.0f}C  PLL={temps['pll_temp']:.0f}C  "
                  f"PMIC={temps['pmic_temp']:.0f}C  Fan={temps['fan_temp']:.0f}C")
            print(f"   CPU负载={temps['cpu_load_pct']:.0f}%@{temps['cpu_freq_mhz']:.0f}MHz  "
                  f"GPU负载={temps['gpu_load_pct']:.0f}%@{temps['gpu_freq_mhz']:.0f}MHz  "
                  f"FanPWM={temps['fan_pwm']}")
            print(f"   内存={temps['mem_used_mb']:.0f}/{temps['mem_total_mb']:.0f}MB "
                  f"({temps['mem_used_pct']:.0f}%)  "
                  f"SWAP={temps['swap_used_mb']:.0f}/{temps['swap_total_mb']:.0f}MB  "
                  f"磁盘={temps['disk_used_pct']:.0f}%")
            bv = temps['battery_voltage_v']
            bi = temps['battery_current_ma']
            pw = temps['power_in_mw']
            print(f"   电池={bv:.2f}V/{bi:.0f}mA  "
                  f"功率: IN={pw:.0f}mW GPU={temps['power_gpu_mw']:.0f}mW CPU={temps['power_cpu_mw']:.0f}mW  "
                  f"WiFi={temps['wifi_rssi_dbm']:.0f}dBm")
            print(f"   亮度={img_brightness:.0f}  对比度={img_contrast:.0f}  "
                  f"均值={img_mean:.0f}  \u03c3={img_std:.0f}")
            if sd.get('frame_count', 0) > 0:
                print(f"   RP2040: gyro=({sd.get('gyro_x',0):.2f},{sd.get('gyro_y',0):.2f},{sd.get('gyro_z',0):.2f})  "
                      f"accel=({sd.get('accel_x',0):.1f},{sd.get('accel_y',0):.1f},{sd.get('accel_z',0):.1f})  "
                      f"yaw={sd.get('euler_yaw',0):.1f}deg")
                print(f"   odom=({sd.get('odom_x',0):.3f},{sd.get('odom_y',0):.3f})  "
                      f"motor L={sd.get('motor_lvel',0)}/R={sd.get('motor_rvel',0)}  "
                      f"frames={sd.get('frame_count',0)} errs={sd.get('parse_errors',0)}")
            if ld.get('frame_count', 0) > 0:
                print(f"   LiDAR: points={ld.get('valid_points', 0)}/{ld.get('points_total', 0)}  "
                      f"near={ld.get('nearest_min', -1.0):.2f}m  "
                      f"angle=[{ld.get('angle_min', 0.0):.2f},{ld.get('angle_max', 0.0):.2f}]  "
                      f"inc={ld.get('angle_increment', 0.0):.4f}rad  "
                      f"age={ld.get('scan_age_ms', -1.0):.0f}ms")

    def shutdown(self):
        if hasattr(self, 'async_writer') and self.async_writer:
            self.async_writer.close()
            writer_stats = self.async_writer.stats()
        else:
            writer_stats = {}
        if hasattr(self, '_telemetry_cache') and self._telemetry_cache:
            self._telemetry_cache.stop()
        process_rss_mb_end = self._read_process_rss_mb()
        if process_rss_mb_end >= 0:
            if self.process_rss_mb_max is None or self.process_rss_mb_max < 0:
                self.process_rss_mb_max = process_rss_mb_end
            else:
                self.process_rss_mb_max = max(self.process_rss_mb_max, process_rss_mb_end)
        try:
            stats_payload = {
                'created_at': datetime.now().isoformat(timespec='seconds'),
                'log_path': self.log_path,
                'process_rss_mb_start': self.process_rss_mb_start,
                'process_rss_mb_end': process_rss_mb_end,
                'process_rss_mb_max': self.process_rss_mb_max,
                'async_writer': writer_stats,
            }
            log_stem, _ = os.path.splitext(self.log_path)
            self.async_writer_stats_path = f'{log_stem}_async_writer_stats.json'
            with open(self.async_writer_stats_path, 'w') as f:
                json.dump(stats_payload, f, indent=2, sort_keys=True)
                f.write('\n')
        except Exception as exc:
            print(f"   Async writer stats 写入失败: {exc}")
        self.csv_file.close()
        if self.lidar_raw_file:
            self.lidar_raw_file.close()
        if self.serial_reader:
            self.serial_reader.stop()
        if self.lidar_reader:
            self.lidar_reader.stop()
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"\n{'='*60}")
        print(f"DataCollector 关闭")
        print(f"   总帧数: {self.frame_count}")
        print(f"   运行时间: {elapsed:.1f}s")
        print(f"   平均 FPS: {avg_fps:.1f}")
        if self.serial_reader:
            sd = self.serial_reader.get_data()
            print(f"   RP2040 帧: {sd.get('frame_count',0)}, 解析错误: {sd.get('parse_errors',0)}")
        if self.lidar_reader:
            ld = self.lidar_reader.get_data()
            print(f"   LiDAR 帧: {ld.get('frame_count',0)}, 最近障碍: {ld.get('nearest_min',-1.0):.2f}m")
        print(f"   日志已保存: {self.log_path}")
        if self.lidar_raw_path:
            print(f"   LiDAR Raw: {self.lidar_raw_path}")
        if self.async_writer_stats_path:
            print(f"   Async writer stats: {self.async_writer_stats_path}")
        if writer_stats:
            print(f"   Async queue max depth: {writer_stats.get('max_queue_depth', 0)}/{writer_stats.get('max_queue_size', 0)}")
            print(f"   Async records written: csv={writer_stats.get('records_written', 0)}, raw={writer_stats.get('raw_records_written', 0)}")
            print(f"   日志丢弃样本: {writer_stats.get('dropped_records', 0)}")
            print(f"   Process RSS MB: start={self.process_rss_mb_start:.3f}, end={process_rss_mb_end:.3f}, max={self.process_rss_mb_max:.3f}")
        print(f"{'='*60}\n")


# ============================================================
# Monkey-patch Vehicle.start 注入采集器
# ============================================================
class ShadowModeGuard:
    """Force user/manual mode during shadow deployment."""

    def __init__(self):
        self._warned = False

    def run(self, mode):
        if mode and mode != 'user' and not self._warned:
            print(f"ShadowModeGuard: forcing user/mode from {mode!r} to 'user'")
            self._warned = True
        return 'user'


class ForceRecording:
    """Force DonkeyCar recording on for data-collection deployment runs."""

    def run(self, recording=None):
        return True


class DeploymentDurationStopper:
    """Stop a deployment run after a wall-clock duration."""

    def __init__(self, duration_sec, label='deployment'):
        self.duration_sec = float(duration_sec)
        self.label = str(label)
        self.start_time = time.time()
        self.vehicle = None
        self._announced = False

    def run(self):
        if self.duration_sec <= 0:
            return
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration_sec and self.vehicle is not None:
            if not self._announced:
                print(f"DeploymentDurationStopper[{self.label}]: reached {elapsed:.1f}s, stopping vehicle loop")
            self._announced = True
            self.vehicle.on = False


class DeploymentPreflightError(RuntimeError):
    """Raised before Vehicle.start when deployment prerequisites are invalid."""


def _expanded_abs_path(path):
    if not path:
        return ''
    return os.path.abspath(os.path.expanduser(str(path)))


def _require_existing_file(path, label):
    resolved = _expanded_abs_path(path)
    if not resolved:
        raise DeploymentPreflightError(f"{label} path is required")
    if not os.path.isfile(resolved):
        raise DeploymentPreflightError(f"{label} not found: {resolved}")
    return resolved


def _load_json_file(path, label):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        raise DeploymentPreflightError(f"{label} is not readable JSON: {path}: {exc}")


def _expected_trt_binding_shapes(shape):
    return {
        'image': (
            1,
            int(shape.get('image_channels', 6)),
            int(shape.get('obs_size', 128)),
            int(shape.get('obs_size', 128)),
        ),
        'state': (1, int(shape.get('state_dim', 7))),
        'lidar': (1, int(shape.get('lidar_dim', 144))),
        'lidar_meta': (1, int(shape.get('lidar_meta_dim', 2))),
        'h': (
            int(shape.get('lstm_layers', 2)),
            1,
            int(shape.get('lstm_hidden_size', 256)),
        ),
        'c': (
            int(shape.get('lstm_layers', 2)),
            1,
            int(shape.get('lstm_hidden_size', 256)),
        ),
        'action': (1, 3),
        'next_h': (
            int(shape.get('lstm_layers', 2)),
            1,
            int(shape.get('lstm_hidden_size', 256)),
        ),
        'next_c': (
            int(shape.get('lstm_layers', 2)),
            1,
            int(shape.get('lstm_hidden_size', 256)),
        ),
    }


def _validate_trt_metadata(metadata, metadata_path, expected_obs_size=None,
                           expected_state_dim=None):
    if not isinstance(metadata, dict):
        raise DeploymentPreflightError(f"TensorRT metadata must be a JSON object: {metadata_path}")

    required_inputs = {'image', 'state', 'lidar', 'lidar_meta', 'h', 'c'}
    required_outputs = {'action', 'next_h', 'next_c'}
    inputs = set(metadata.get('inputs') or [])
    outputs = set(metadata.get('outputs') or [])
    missing_inputs = sorted(required_inputs - inputs)
    missing_outputs = sorted(required_outputs - outputs)
    if missing_inputs:
        raise DeploymentPreflightError(
            f"TensorRT metadata missing inputs {missing_inputs}: {metadata_path}"
        )
    if missing_outputs:
        raise DeploymentPreflightError(
            f"TensorRT metadata missing outputs {missing_outputs}: {metadata_path}"
        )

    shape = metadata.get('shape') or {}
    required_shape_keys = [
        'image_channels', 'obs_size', 'state_dim', 'lidar_dim',
        'lidar_meta_dim', 'lstm_layers', 'lstm_hidden_size',
    ]
    missing_shape = [key for key in required_shape_keys if key not in shape]
    if missing_shape:
        raise DeploymentPreflightError(
            f"TensorRT metadata missing shape keys {missing_shape}: {metadata_path}"
        )

    expected_fixed = {
        'image_channels': 6,
        'lidar_dim': 144,
        'lidar_meta_dim': 2,
    }
    for key, expected in expected_fixed.items():
        actual = int(shape.get(key))
        if actual != expected:
            raise DeploymentPreflightError(
                f"TensorRT metadata shape mismatch for {key}: expected {expected}, got {actual}"
            )
    if int(shape.get('lstm_layers')) <= 0 or int(shape.get('lstm_hidden_size')) <= 0:
        raise DeploymentPreflightError(
            f"TensorRT metadata has invalid LSTM shape: {shape}"
        )
    if expected_obs_size is not None and int(shape.get('obs_size')) != int(expected_obs_size):
        raise DeploymentPreflightError(
            f"TensorRT metadata obs_size mismatch: CLI={expected_obs_size}, metadata={shape.get('obs_size')}"
        )
    if expected_state_dim is not None and int(shape.get('state_dim')) != int(expected_state_dim):
        raise DeploymentPreflightError(
            f"TensorRT metadata state_dim mismatch: CLI={expected_state_dim}, metadata={shape.get('state_dim')}"
        )
    return shape


def _preflight_trt_engine_bindings(engine_path, metadata):
    try:
        import tensorrt as trt
    except Exception as exc:
        raise DeploymentPreflightError(f"TensorRT import failed during preflight: {exc}")

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        payload = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(payload)
    if engine is None:
        raise DeploymentPreflightError(f"TensorRT engine deserialize failed: {engine_path}")

    expected = _expected_trt_binding_shapes(metadata.get('shape') or {})
    actual = {}
    for index in range(int(engine.num_bindings)):
        name = engine.get_binding_name(index)
        try:
            dims = tuple(int(x) for x in engine.get_binding_shape(index))
        except Exception:
            dims = ()
        try:
            is_input = bool(engine.binding_is_input(index))
        except Exception:
            is_input = None
        actual[name] = {
            'shape': list(dims),
            'is_input': is_input,
        }

    missing = sorted(set(expected.keys()) - set(actual.keys()))
    if missing:
        raise DeploymentPreflightError(
            f"TensorRT engine missing bindings {missing}: {engine_path}"
        )

    for name, expected_shape in expected.items():
        dims = tuple(actual[name].get('shape') or ())
        if dims and all(int(x) >= 0 for x in dims) and dims != tuple(expected_shape):
            raise DeploymentPreflightError(
                f"TensorRT binding shape mismatch for {name}: expected {expected_shape}, got {dims}"
            )
    return actual


def _write_preflight_report(log_dir, report):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, 'preflight_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    return path


def run_deployment_preflight(args, shadow_model, deployment_enabled):
    report = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'ok': False,
        'control_mode': args.control_mode,
        'model_path': _expanded_abs_path(shadow_model) if shadow_model else '',
        'engine_path': _expanded_abs_path(args.shadow_engine) if args.shadow_engine else '',
        'metadata_path': _expanded_abs_path(args.shadow_engine_metadata) if args.shadow_engine_metadata else '',
        'checks': [],
    }
    if not deployment_enabled:
        report['ok'] = True
        report['checks'].append('deployment_disabled')
        return report

    model_path = _require_existing_file(shadow_model, 'V17 model')
    report['model_path'] = model_path
    report['checks'].append('model_exists')

    if args.shadow_engine:
        engine_path = _require_existing_file(args.shadow_engine, 'TensorRT engine')
        metadata_path = args.shadow_engine_metadata
        if not metadata_path:
            metadata_path = os.path.join(os.path.dirname(engine_path), 'v17_actor_export.json')
        metadata_path = _require_existing_file(metadata_path, 'TensorRT metadata')
        metadata = _load_json_file(metadata_path, 'TensorRT metadata')
        shape = _validate_trt_metadata(
            metadata,
            metadata_path,
            expected_obs_size=args.shadow_obs_size,
            expected_state_dim=args.shadow_state_dim,
        )
        bindings = _preflight_trt_engine_bindings(engine_path, metadata)
        report.update({
            'engine_path': engine_path,
            'metadata_path': metadata_path,
            'metadata_shape': shape,
            'binding_summary': bindings,
        })
        report['checks'].extend([
            'engine_exists',
            'metadata_exists',
            'metadata_shape_valid',
            'engine_bindings_match_metadata',
        ])

    report['ok'] = True
    return report


def _wait_for_reader_data(reader, max_age_ms=None, timeout_sec=2.0):
    deadline = time.time() + max(0.0, float(timeout_sec))
    last_data = {}
    while time.time() <= deadline:
        if not reader or not getattr(reader, 'is_connected', False):
            time.sleep(0.05)
            continue
        try:
            last_data = reader.get_data()
        except Exception:
            last_data = {}
        if int(last_data.get('frame_count', 0) or 0) > 0:
            last_update = float(last_data.get('last_update', 0.0) or 0.0)
            if last_update <= 0:
                return True, last_data
            age_ms = max(0.0, (time.time() - last_update) * 1000.0)
            if max_age_ms is None or age_ms <= float(max_age_ms):
                return True, last_data
        time.sleep(0.05)
    return False, last_data


def _format_lidar_startup_data(data):
    return (
        f"frames={data.get('frame_count', 0)}, "
        f"age_ms={data.get('scan_age_ms', -1.0)}, "
        f"last_update={data.get('last_update', 0.0)}, "
        f"valid_points={data.get('valid_points', 0)}, "
        f"points_total={data.get('points_total', 0)}, "
        f"parse_errors={data.get('parse_errors', 0)}"
    )


class DeploymentSafetyGate:
    """Monitor deployment safety thresholds and stop active runs on violation."""

    def __init__(self, control_mode='shadow', serial_reader=None, lidar_reader=None,
                 require_lidar=False, require_rp2040=False,
                 max_lidar_age_ms=None, max_inference_ms=None,
                 max_rp2040_age_ms=None):
        self.control_mode = str(control_mode or 'shadow')
        self.serial_reader = serial_reader
        self.lidar_reader = lidar_reader
        self.require_lidar = bool(require_lidar)
        self.require_rp2040 = bool(require_rp2040)
        self.max_lidar_age_ms = (
            None if max_lidar_age_ms is None else float(max_lidar_age_ms)
        )
        self.max_inference_ms = (
            None if max_inference_ms is None else float(max_inference_ms)
        )
        self.max_rp2040_age_ms = (
            None if max_rp2040_age_ms is None else float(max_rp2040_age_ms)
        )
        self.vehicle = None
        self.blocked = False
        self.block_reason = ''
        self.inference_timeout_count = 0
        self.lidar_missing_count = 0
        self.lidar_stale_count = 0
        self.rp2040_missing_count = 0
        self.last_lidar_age_ms = -1.0
        self.last_rp2040_age_ms = -1.0
        self._printed_reason = None

    def _record_reason(self, reason):
        if not reason:
            return
        self.block_reason = reason
        reason_key = str(reason).split(':', 1)[0]
        if self._printed_reason != reason_key:
            print(f"DeploymentSafetyGate: {reason}")
            self._printed_reason = reason_key
        if self.control_mode == 'active':
            self.blocked = True
            if self.vehicle is not None:
                self.vehicle.on = False

    def _check_lidar(self):
        if not self.require_lidar and self.max_lidar_age_ms is None:
            return
        if not self.lidar_reader or not self.lidar_reader.is_connected:
            self.lidar_missing_count += 1
            if self.require_lidar:
                self._record_reason('lidar_missing')
            return
        data = self.lidar_reader.get_data()
        frame_count = int(data.get('frame_count', 0) or 0)
        age = float(data.get('scan_age_ms', -1.0) or -1.0)
        self.last_lidar_age_ms = age
        if frame_count <= 0:
            self.lidar_missing_count += 1
            if self.require_lidar:
                self._record_reason('lidar_no_frames')
            return
        if self.max_lidar_age_ms is not None and age > self.max_lidar_age_ms:
            self.lidar_stale_count += 1
            self._record_reason(f'lidar_stale:{age:.1f}ms>{self.max_lidar_age_ms:.1f}ms')

    def _check_rp2040(self):
        if not self.require_rp2040 and self.max_rp2040_age_ms is None:
            return
        if not self.serial_reader or not self.serial_reader.is_connected:
            self.rp2040_missing_count += 1
            if self.require_rp2040:
                self._record_reason('rp2040_missing')
            return
        data = self.serial_reader.get_data()
        frame_count = int(data.get('frame_count', 0) or 0)
        last_update = float(data.get('last_update', 0.0) or 0.0)
        age = max(0.0, (time.time() - last_update) * 1000.0) if last_update > 0 else -1.0
        self.last_rp2040_age_ms = age
        if frame_count <= 0:
            self.rp2040_missing_count += 1
            if self.require_rp2040:
                self._record_reason('rp2040_no_frames')
            return
        if self.max_rp2040_age_ms is not None and age > self.max_rp2040_age_ms:
            self.rp2040_missing_count += 1
            self._record_reason(f'rp2040_stale:{age:.1f}ms>{self.max_rp2040_age_ms:.1f}ms')

    def run(self, pilot_angle, pilot_throttle, pilot_inference_latency_ms):
        latency = -1.0
        if pilot_inference_latency_ms is not None:
            try:
                latency = float(pilot_inference_latency_ms)
            except Exception:
                latency = -1.0
        if self.max_inference_ms is not None and latency > self.max_inference_ms:
            self.inference_timeout_count += 1
            self._record_reason(f'inference_timeout:{latency:.1f}ms>{self.max_inference_ms:.1f}ms')

        self._check_lidar()
        self._check_rp2040()

        safe_angle = 0.0 if self.blocked else (pilot_angle or 0.0)
        safe_throttle = 0.0 if self.blocked else (pilot_throttle or 0.0)
        return (
            safe_angle,
            safe_throttle,
            bool(self.blocked),
            self.block_reason,
            int(self.inference_timeout_count),
            int(self.lidar_missing_count),
            int(self.lidar_stale_count),
            int(self.rp2040_missing_count),
            float(self.last_lidar_age_ms),
            float(self.last_rp2040_age_ms),
        )


def _move_last_part_before(vehicle, predicate, fallback_index=None):
    """Move the most recently added part before the first part matching predicate."""
    if not vehicle.parts:
        return
    entry = vehicle.parts.pop()
    insert_at = fallback_index if fallback_index is not None else len(vehicle.parts)
    for idx, part_entry in enumerate(vehicle.parts):
        if predicate(part_entry):
            insert_at = idx
            break
    vehicle.parts.insert(insert_at, entry)


def _move_last_part_before_output(vehicle, output_name, fallback_index=None):
    """Move the most recently added part before the first producer of output_name."""
    _move_last_part_before(
        vehicle,
        lambda part_entry: output_name in part_entry.get('outputs', []),
        fallback_index=fallback_index,
    )


def _move_last_part_before_run_condition(vehicle, run_condition, fallback_index=None):
    """Move the most recently added part before the first part using run_condition."""
    _move_last_part_before(
        vehicle,
        lambda part_entry: part_entry.get('run_condition') == run_condition,
        fallback_index=fallback_index,
    )


def _write_run_notes_template(log_dir, log_path, metadata, deployment_config):
    """Create a small editable notes file for report labels and outcomes."""
    if not deployment_config.get('write_run_notes', True):
        return None

    notes_path = os.path.join(log_dir, 'run_notes.json')
    if os.path.exists(notes_path):
        return notes_path

    notes = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'csv_log': log_path,
        'control_mode': metadata.get('control_mode', 'normal'),
        'track_condition': metadata.get('track_condition', ''),
        'run_label': metadata.get('run_label', ''),
        'model_name': metadata.get('model_name', ''),
        'model_path': metadata.get('model_path', ''),
        'planned_duration_sec': deployment_config.get('duration_sec'),
        'obstacle_layout': deployment_config.get('obstacle_layout', ''),
        'run_outcome': deployment_config.get('run_outcome', 'unknown'),
        'collision_or_contact': deployment_config.get('collision_or_contact'),
        'stuck_detected': deployment_config.get('stuck_detected'),
        'obstacle_recovery_success': deployment_config.get('obstacle_recovery_success', 'unknown'),
        'safety': deployment_config.get('safety', {}),
        'notes': deployment_config.get('notes', ''),
    }
    with open(notes_path, 'w') as f:
        json.dump(notes, f, indent=2, sort_keys=True)
        f.write('\n')
    return notes_path


def install_monitor(log_dir, log_interval, serial_port='/dev/ttyACM0',
                    lidar_topic='/scan', enable_lidar=True,
                    lidar_auto_start_driver=True,
                    lidar_launch='jetracer lidar.launch',
                    lidar_ready_timeout=12.0,
                    shadow_config=None):
    """
    Monkey-patch dk.vehicle.Vehicle.start，在 Vehicle 真正启动前
    自动注入 DataCollector Part 和 RP2040 串口读取器。
    manage.py 完全不需要修改。
    """
    import donkeycar as dk
    from donkeycar.vehicle import Vehicle

    shadow_config = dict(shadow_config or {})
    control_mode = shadow_config.get(
        'control_mode',
        'shadow' if shadow_config.get('enabled') else 'normal',
    )
    safety_config = dict(shadow_config.get('safety') or {})

    # 启动 RP2040 串口读取器
    serial_reader = RP2040SerialReader(port=serial_port)
    serial_ok = serial_reader.start()
    lidar_reader = None
    lidar_ok = False
    if enable_lidar:
        lidar_reader = RosLidarReader(
            topic=lidar_topic,
            auto_start_driver=lidar_auto_start_driver,
            driver_launch=lidar_launch,
            driver_ready_timeout=lidar_ready_timeout,
            driver_log_dir=log_dir,
        )
        lidar_ok = lidar_reader.start()

    if shadow_config.get('enabled'):
        require_lidar = bool(safety_config.get('require_lidar', False))
        require_rp2040 = bool(safety_config.get('require_rp2040', False))
        max_lidar_age_ms = safety_config.get('max_lidar_age_ms')
        max_rp2040_age_ms = safety_config.get('max_rp2040_age_ms')
        if require_lidar:
            if not enable_lidar:
                raise DeploymentPreflightError('require_lidar set but LiDAR is disabled')
            lidar_ready, lidar_data = _wait_for_reader_data(
                lidar_reader, max_age_ms=max_lidar_age_ms, timeout_sec=8.0
            )
            if not lidar_ready:
                raise DeploymentPreflightError(
                    'require_lidar startup check failed: connected='
                    f"{bool(lidar_ok)}, {_format_lidar_startup_data(lidar_data)}"
                )
        if require_rp2040:
            rp_ready, rp_data = _wait_for_reader_data(
                serial_reader, max_age_ms=max_rp2040_age_ms, timeout_sec=3.0
            )
            if not rp_ready:
                raise DeploymentPreflightError(
                    'require_rp2040 startup check failed: '
                    f"connected={bool(serial_ok)}, frames={rp_data.get('frame_count', 0)}"
                )

    _original_start = Vehicle.start

    def _patched_start(self, rate_hz=10, max_loop_count=None, verbose=False):
        # 生成日志文件名
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'run_{ts}.csv')

        shadow_metadata = {}
        if shadow_config.get('enabled'):
            model_path = shadow_config.get('model_path')
            if not model_path:
                raise ValueError(f"{control_mode} mode requires --model or --shadow-model")

            if control_mode == 'shadow':
                self.add(ShadowModeGuard(),
                         inputs=['user/mode'],
                         outputs=['user/mode'],
                         threaded=False)
                _move_last_part_before_output(self, 'run_pilot', fallback_index=0)

            from v17_pilot import V17Pilot
            deployment_pilot = V17Pilot(
                model_path=model_path,
                obs_size=shadow_config.get('obs_size', 128),
                state_dim=shadow_config.get('state_dim'),
                domain=shadow_config.get('domain', 'ws'),
                max_throttle=shadow_config.get('max_throttle', 0.8),
                delta_max=shadow_config.get('delta_max', 0.35),
                enable_lpf=shadow_config.get('enable_lpf', True),
                beta=shadow_config.get('beta', 0.6),
                use_cuda=shadow_config.get('use_cuda', True),
                serial_reader=serial_reader if serial_ok else None,
                lidar_reader=lidar_reader if lidar_ok else None,
                warmup_frames=shadow_config.get('warmup_frames', 5),
                trt_engine_path=shadow_config.get('trt_engine_path'),
                trt_metadata_path=shadow_config.get('trt_metadata_path'),
            )
            shadow_metadata = deployment_pilot.metadata
            shadow_metadata.update({
                'control_mode': control_mode,
                'track_condition': shadow_config.get('track_condition', ''),
                'run_label': shadow_config.get('run_label', ''),
            })
            self.add(deployment_pilot,
                     inputs=['cam/image_array', 'user/angle', 'user/throttle',
                             'angle', 'throttle'],
                     outputs=['pilot/angle', 'pilot/throttle',
                              'pilot/inference_latency_ms',
                              'pilot/raw_angle', 'pilot/raw_throttle',
                              'pilot/preprocess_latency_ms'],
                     threaded=False)
            safety_gate = DeploymentSafetyGate(
                control_mode=control_mode,
                serial_reader=serial_reader if serial_ok else None,
                lidar_reader=lidar_reader if lidar_ok else None,
                require_lidar=safety_config.get('require_lidar', False),
                require_rp2040=safety_config.get('require_rp2040', False),
                max_lidar_age_ms=safety_config.get('max_lidar_age_ms'),
                max_inference_ms=safety_config.get('max_inference_ms'),
                max_rp2040_age_ms=safety_config.get('max_rp2040_age_ms'),
            )
            safety_gate.vehicle = self
            self.add(safety_gate,
                     inputs=['pilot/angle', 'pilot/throttle',
                             'pilot/inference_latency_ms'],
                     outputs=['pilot/angle', 'pilot/throttle',
                              'safety/blocked',
                              'safety/block_reason',
                              'safety/inference_timeout_count',
                              'safety/lidar_missing_count',
                              'safety/lidar_stale_count',
                              'safety/rp2040_missing_count',
                              'safety/last_lidar_age_ms',
                              'safety/last_rp2040_age_ms'],
                     threaded=False)
            if control_mode == 'active':
                _move_last_part_before(
                    self,
                    lambda part_entry: (
                        'angle' in part_entry.get('outputs', []) or
                        (
                            'pilot/throttle' in part_entry.get('inputs', []) and
                            'pilot/throttle' in part_entry.get('outputs', [])
                        )
                    ),
                    fallback_index=len(self.parts),
                )
                _move_last_part_before(
                    self,
                    lambda part_entry: (
                        'angle' in part_entry.get('outputs', []) or
                        (
                            'pilot/throttle' in part_entry.get('inputs', []) and
                            'pilot/throttle' in part_entry.get('outputs', [])
                        )
                    ),
                    fallback_index=len(self.parts),
                )

            if shadow_config.get('force_recording'):
                self.add(ForceRecording(),
                         inputs=['recording'],
                         outputs=['recording'],
                         threaded=False)
                _move_last_part_before_run_condition(
                    self, 'recording', fallback_index=len(self.parts)
                )

            duration = shadow_config.get('duration_sec')
            if duration:
                stopper = DeploymentDurationStopper(duration, label=control_mode)
                stopper.vehicle = self
                self.add(stopper, inputs=[], outputs=[], threaded=False)
            notes_path = _write_run_notes_template(
                log_dir, log_path, shadow_metadata, shadow_config
            )
            if control_mode == 'active':
                print("V17 active pilot injected; local mode uses V17 pilot outputs")
                print("   Safety: switch back to user mode for manual override")
            else:
                print("V17 shadow pilot injected; actuator path remains user/manual")
            if notes_path:
                print(f"   Run notes: {notes_path}")

        # 创建并注入采集器（带串口读取器）
        collector = DataCollector(log_path=log_path, log_interval=log_interval,
                                  serial_reader=serial_reader if serial_ok else None,
                                  lidar_reader=lidar_reader if lidar_ok else None,
                                  metadata=shadow_metadata)
        collector.vehicle = self
        self.add(collector,
                 inputs=DataCollector.INPUT_KEYS,
                 outputs=[],
                 threaded=False)

        print(f"DataCollector 已注入 Vehicle 循环 (rate={rate_hz}Hz)")

        return _original_start(self, rate_hz=rate_hz,
                               max_loop_count=max_loop_count,
                               verbose=verbose)

    Vehicle.start = _patched_start


# ============================================================
# CLI 入口
# ============================================================
def main():
    """
    命令行入口 —— 参数与 manage.py 完全兼容，额外支持监控参数。
    """
    # 强制关闭 stdout 缓冲（SSH 管道下 print 会被缓冲导致终端无输出）
    import io
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), 'wb', 0), write_through=True)

    import argparse

    parser = argparse.ArgumentParser(
        description='DonkeyCar 运行时数据采集（独立于 manage.py）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python runtime_monitor.py drive --model ~/mycar/models/v8_140000_steps_policy.pth --type v8
  python runtime_monitor.py drive --model xxx --type v8 --js --log-interval 0.2
  python runtime_monitor.py drive --model ~/mycar/models/v17_latest.zip --type v17 --control-mode shadow --js
  python runtime_monitor.py drive --model ~/mycar/models/v17_latest.zip --type v17 --control-mode active --js
  python runtime_monitor.py drive  # 手动模式，仅采集用户操作 + 系统数据
        """)

    parser.add_argument('command', choices=['drive'], help='运行命令')
    parser.add_argument('--model', type=str, default=None, help='模型文件路径')
    parser.add_argument('--type', type=str, default=None,
                        choices=['linear', 'categorical', 'v8', 'v17'],
                        help='模型类型')
    parser.add_argument('--js', action='store_true', help='使用物理手柄')
    parser.add_argument('--myconfig', type=str, default='myconfig.py',
                        help='自定义配置文件')
    parser.add_argument('--meta', type=str, nargs='*', default=[],
                        help='元数据 key:value')

    # 监控专用参数
    parser.add_argument('--log-interval', type=float, default=0.5,
                        help='日志写入间隔(秒), 默认 0.5')
    parser.add_argument('--log-dir', type=str,
                        default=None,
                        help='日志输出目录')
    parser.add_argument('--serial-port', type=str, default='/dev/ttyACM0',
                        help='RP2040 串口设备 (默认 /dev/ttyACM0)')
    parser.add_argument('--lidar-topic', type=str, default='/scan',
                        help='LiDAR LaserScan topic (默认 /scan)')
    parser.add_argument('--disable-lidar', action='store_true',
                        help='禁用 LiDAR 采集')
    parser.add_argument('--no-start-lidar-driver', action='store_true',
                        help='不自动启动 ROS LiDAR driver，仅订阅已有 topic')
    parser.add_argument('--lidar-launch', type=str, default='jetracer lidar.launch',
                        help='自动启动的 roslaunch 目标，例如 "jetracer lidar.launch"')
    parser.add_argument('--lidar-ready-timeout', type=float, default=12.0,
                        help='等待 LiDAR /scan topic ready 的秒数')
    parser.add_argument('--require-lidar', dest='require_lidar',
                        action='store_true', default=None,
                        help='部署 run 启动前要求 LiDAR ready；active 默认开启')
    parser.add_argument('--no-require-lidar', dest='require_lidar',
                        action='store_false',
                        help='关闭 LiDAR ready gate')
    parser.add_argument('--require-rp2040', dest='require_rp2040',
                        action='store_true', default=None,
                        help='部署 run 启动前要求 RP2040 ready；active 默认开启')
    parser.add_argument('--no-require-rp2040', dest='require_rp2040',
                        action='store_false',
                        help='关闭 RP2040 ready gate')
    parser.add_argument('--max-lidar-age-ms', type=float, default=None,
                        help='LiDAR scan age 上限；active 默认 350ms')
    parser.add_argument('--max-rp2040-age-ms', type=float, default=None,
                        help='RP2040 数据 freshness 上限；active 默认 1000ms')
    parser.add_argument('--max-inference-ms', type=float, default=None,
                        help='V17 推理延时上限；active 默认 350ms，超限触发 safety gate')
    parser.add_argument('--control-mode', type=str, default='normal',
                        choices=['normal', 'shadow', 'active'],
                        help='normal 使用 manage.py 默认控制; shadow 只记录 pilot 输出; active 由 V17 接管 local 模式')
    parser.add_argument('--shadow-model', type=str, default=None,
                        help='shadow pilot 模型路径；默认复用 --model')
    parser.add_argument('--shadow-engine', type=str, default=None,
                        help='V17 shadow/active pilot 使用的 TensorRT engine 路径')
    parser.add_argument('--shadow-engine-metadata', type=str, default=None,
                        help='V17 TensorRT engine 元数据 JSON 路径')
    parser.add_argument('--shadow-duration', type=float, default=None,
                        help='shadow run 秒数；例如 30 smoke 或 180 正式采集')
    parser.add_argument('--active-duration', type=float, default=180.0,
                        help='active run 秒数；例如 180 实地采集')
    parser.add_argument('--shadow-obs-size', type=int, default=128,
                        help='V17 observation image size, default 128')
    parser.add_argument('--shadow-state-dim', type=int, default=None,
                        help='V17 state dim override; 默认从模型推断或 7')
    parser.add_argument('--shadow-domain', type=str, default='ws',
                        choices=['ws', 'gt', 'rrl', 'generic'],
                        help='CanonicalSemanticWrapper domain, default ws')
    parser.add_argument('--shadow-max-throttle', type=float, default=0.8,
                        help='ActionAdapter throttle clamp, default 0.8')
    parser.add_argument('--shadow-delta-max', type=float, default=0.35,
                        help='ActionSafety steering delta_max, default 0.35')
    parser.add_argument('--shadow-beta', type=float, default=0.6,
                        help='ActionSafety LPF beta, default 0.6')
    parser.add_argument('--shadow-disable-lpf', action='store_true',
                        help='禁用 ActionSafety LPF')
    parser.add_argument('--shadow-cpu', action='store_true',
                        help='强制 V17 shadow pilot 使用 CPU')
    parser.add_argument('--shadow-warmup-frames', type=int, default=5,
                        help='V17 pilot warmup frames before logging')
    parser.add_argument('--track-condition', type=str, default='',
                        help='报告场景标签，例如 obstacle_recovery')
    parser.add_argument('--run-label', type=str, default='',
                        help='报告 run 标签，例如 v17_active_obstacle_run1')
    parser.add_argument('--force-recording', dest='force_recording',
                        action='store_true', default=None,
                        help='强制 recording=True，active 默认开启')
    parser.add_argument('--no-force-recording', dest='force_recording',
                        action='store_false',
                        help='关闭强制 recording')
    parser.add_argument('--obstacle-layout', type=str, default='',
                        help='run_notes.json 中的障碍布局描述')
    parser.add_argument('--run-outcome', type=str, default='unknown',
                        choices=['unknown', 'completed', 'manual_override',
                                 'collision', 'stuck', 'timeout'],
                        help='run_notes.json 中的人工结果标签，可事后修改')
    parser.add_argument('--collision-or-contact', action='store_true',
                        default=None,
                        help='标记本 run 发生碰撞或接触')
    parser.add_argument('--stuck-detected', action='store_true',
                        default=None,
                        help='标记本 run 发生卡死')
    parser.add_argument('--obstacle-recovery-success', type=str,
                        default='unknown',
                        choices=['unknown', 'true', 'false'],
                        help='障碍恢复是否成功，可事后修改')
    parser.add_argument('--notes', type=str, default='',
                        help='写入 run_notes.json 的自由文本备注')

    args = parser.parse_args()
    default_log_dir = os.path.expanduser('~/mycar/monitor_logs')
    log_dir = args.log_dir
    if log_dir is None:
        if args.control_mode == 'active':
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            condition = args.track_condition or 'obstacle_recovery'
            safe_condition = ''.join(
                ch if ch.isalnum() or ch in ('-', '_') else '_'
                for ch in condition
            ).strip('_') or 'field'
            log_dir = os.path.join(
                default_log_dir, f'v17_active_{safe_condition}_{stamp}'
            )
        else:
            log_dir = default_log_dir
    args.log_dir = os.path.expanduser(log_dir)

    if args.force_recording is None:
        args.force_recording = args.control_mode == 'active'
    if args.require_lidar is None:
        args.require_lidar = args.control_mode == 'active'
    if args.require_rp2040 is None:
        args.require_rp2040 = args.control_mode == 'active'
    if args.control_mode == 'active':
        if args.max_lidar_age_ms is None:
            args.max_lidar_age_ms = 350.0
        if args.max_rp2040_age_ms is None:
            args.max_rp2040_age_ms = 1000.0
        if args.max_inference_ms is None:
            args.max_inference_ms = 350.0

    print("\n" + "=" * 60)
    print("DonkeyCar Runtime Monitor + RP2040 Sensor Bridge")
    print("=" * 60)
    print(f"   模型: {args.model or '无（手动模式）'}")
    print(f"   类型: {args.type or '自动检测'}")
    print(f"   手柄: {args.js}")
    print(f"   日志间隔: {args.log_interval}s")
    print(f"   日志目录: {args.log_dir}")
    print(f"   RP2040 串口: {args.serial_port}")
    print(f"   LiDAR: {'关闭' if args.disable_lidar else f'{args.lidar_topic} (raw /scan)'}")
    print(f"   控制模式: {args.control_mode}")
    print(f"   Safety gate: require_lidar={args.require_lidar}, require_rp2040={args.require_rp2040}, "
          f"max_lidar_age_ms={args.max_lidar_age_ms}, max_rp2040_age_ms={args.max_rp2040_age_ms}, "
          f"max_inference_ms={args.max_inference_ms}")
    print(f"   场景标签: {args.track_condition or '未设置'}")
    print(f"   强制录制: {args.force_recording}")
    print("=" * 60 + "\n")

    import donkeycar as dk
    cfg = dk.load_config(myconfig=args.myconfig)

    # 禁用 DonkeyCar 自带的 IMU/编码器 Part（传感器数据由 RP2040 串口提供）
    cfg.HAVE_IMU = False
    cfg.HAVE_ODOM = False
    cfg.RP2040_SERIAL_PORT = args.serial_port
    if args.control_mode == 'active' and args.force_recording:
        cfg.RECORD_DURING_AI = True
        cfg.AUTO_CREATE_NEW_TUB = True

    shadow_model = args.shadow_model or args.model
    deployment_enabled = args.control_mode in ('shadow', 'active')
    shadow_enabled = args.control_mode == 'shadow'
    active_enabled = args.control_mode == 'active'
    if deployment_enabled and not shadow_model:
        raise ValueError(f"--control-mode {args.control_mode} requires --model or --shadow-model")
    try:
        preflight_report = run_deployment_preflight(args, shadow_model, deployment_enabled)
        preflight_path = _write_preflight_report(args.log_dir, preflight_report)
        if deployment_enabled:
            print(f"Deployment preflight passed: {preflight_path}")
    except DeploymentPreflightError as exc:
        failed_report = {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'ok': False,
            'control_mode': args.control_mode,
            'error': str(exc),
            'model_path': _expanded_abs_path(shadow_model) if shadow_model else '',
            'engine_path': _expanded_abs_path(args.shadow_engine) if args.shadow_engine else '',
            'metadata_path': _expanded_abs_path(args.shadow_engine_metadata) if args.shadow_engine_metadata else '',
        }
        try:
            preflight_path = _write_preflight_report(args.log_dir, failed_report)
            print(f"Deployment preflight failed: {preflight_path}", file=sys.stderr)
        except Exception:
            pass
        print(f"Deployment preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
    duration_sec = args.active_duration if active_enabled else args.shadow_duration
    shadow_config = {
        'enabled': deployment_enabled,
        'control_mode': args.control_mode,
        'model_path': shadow_model,
        'duration_sec': duration_sec,
        'obs_size': args.shadow_obs_size,
        'state_dim': args.shadow_state_dim,
        'domain': args.shadow_domain,
        'max_throttle': args.shadow_max_throttle,
        'delta_max': args.shadow_delta_max,
        'enable_lpf': not args.shadow_disable_lpf,
        'beta': args.shadow_beta,
        'use_cuda': not args.shadow_cpu,
        'warmup_frames': args.shadow_warmup_frames,
        'trt_engine_path': args.shadow_engine,
        'trt_metadata_path': args.shadow_engine_metadata,
        'track_condition': args.track_condition,
        'run_label': args.run_label,
        'force_recording': args.force_recording,
        'obstacle_layout': args.obstacle_layout,
        'run_outcome': args.run_outcome,
        'collision_or_contact': args.collision_or_contact,
        'stuck_detected': args.stuck_detected,
        'obstacle_recovery_success': args.obstacle_recovery_success,
        'safety': {
            'require_lidar': args.require_lidar,
            'require_rp2040': args.require_rp2040,
            'max_lidar_age_ms': args.max_lidar_age_ms,
            'max_rp2040_age_ms': args.max_rp2040_age_ms,
            'max_inference_ms': args.max_inference_ms,
        },
        'notes': args.notes,
    }

    try:
        install_monitor(log_dir=args.log_dir, log_interval=args.log_interval,
                        serial_port=args.serial_port,
                        lidar_topic=args.lidar_topic,
                        enable_lidar=not args.disable_lidar,
                        lidar_auto_start_driver=not args.no_start_lidar_driver,
                        lidar_launch=args.lidar_launch,
                        lidar_ready_timeout=args.lidar_ready_timeout,
                        shadow_config=shadow_config)
    except DeploymentPreflightError as exc:
        failed_report = {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'ok': False,
            'stage': 'startup_safety_gate',
            'control_mode': args.control_mode,
            'error': str(exc),
            'serial_port': args.serial_port,
            'lidar_topic': args.lidar_topic,
            'safety': shadow_config.get('safety', {}),
        }
        try:
            preflight_path = _write_preflight_report(args.log_dir, failed_report)
            print(f"Startup safety gate failed: {preflight_path}", file=sys.stderr)
        except Exception:
            pass
        print(f"Startup safety gate failed: {exc}", file=sys.stderr)
        raise SystemExit(2)

    from manage import drive
    drive_model_path = None if deployment_enabled and args.type == 'v17' else args.model
    drive_model_type = None if deployment_enabled and args.type == 'v17' else args.type
    drive(cfg,
          model_path=drive_model_path,
          use_joystick=args.js,
          model_type=drive_model_type,
          camera_type='single',
          meta=args.meta or [])


if __name__ == '__main__':
    main()
