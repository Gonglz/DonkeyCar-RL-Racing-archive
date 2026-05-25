#!/usr/bin/env python
"""Publish a normal-then-faulty LaserScan sequence for runtime gate tests.

Python 2 compatible for ROS Melodic.
"""

from __future__ import print_function

import argparse
import math
import time

import rospy
from sensor_msgs.msg import LaserScan


def build_scan(seq, stamp, rate_hz):
    msg = LaserScan()
    msg.header.seq = int(seq)
    msg.header.stamp = stamp
    msg.header.frame_id = "fault_sequence_laser"
    msg.angle_min = -math.pi
    msg.angle_max = math.pi
    msg.angle_increment = (2.0 * math.pi) / 360.0
    msg.time_increment = 0.0
    msg.scan_time = 1.0 / max(rate_hz, 1e-6)
    msg.range_min = 0.18
    msg.range_max = 20.0
    msg.ranges = [1.0] * 360
    msg.intensities = []
    return msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/fault_scan")
    parser.add_argument("--mode", choices=["freeze", "drop"], default="freeze")
    parser.add_argument("--duration-sec", type=float, default=45.0)
    parser.add_argument("--normal-sec", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    args = parser.parse_args()

    rospy.init_node("publish_lidar_fault_sequence", anonymous=True)
    pub = rospy.Publisher(args.topic, LaserScan, queue_size=1, latch=False)
    rate = rospy.Rate(args.rate_hz)
    started = time.time()
    deadline = started + args.duration_sec
    normal_deadline = started + args.normal_sec
    frozen_stamp = None
    seq = 0
    normal_frames = 0
    fault_frames = 0

    while not rospy.is_shutdown() and time.time() < deadline:
        now = time.time()
        if now < normal_deadline:
            msg = build_scan(seq, rospy.Time.now(), args.rate_hz)
            pub.publish(msg)
            normal_frames += 1
        else:
            if args.mode == "drop":
                rate.sleep()
                continue
            if frozen_stamp is None:
                frozen_stamp = rospy.Time.now()
            msg = build_scan(seq, frozen_stamp, args.rate_hz)
            pub.publish(msg)
            fault_frames += 1
        seq += 1
        rate.sleep()

    print(
        "published topic=%s mode=%s normal_frames=%d fault_frames=%d"
        % (args.topic, args.mode, normal_frames, fault_frames)
    )


if __name__ == "__main__":
    main()
