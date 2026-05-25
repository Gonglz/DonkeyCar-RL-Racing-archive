#!/usr/bin/env python
"""Publish a stale LaserScan for runtime safety-gate tests.

This script is Python 2 compatible for ROS Melodic.
"""

from __future__ import print_function

import math
import sys
import time

import rospy
from sensor_msgs.msg import LaserScan


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "/stale_scan"
    duration_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    rate_hz = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    rospy.init_node("publish_stale_lidar_scan", anonymous=True)
    pub = rospy.Publisher(topic, LaserScan, queue_size=1, latch=True)
    rate = rospy.Rate(rate_hz)
    deadline = time.time() + duration_sec
    frame = 0

    while not rospy.is_shutdown() and time.time() < deadline:
        msg = LaserScan()
        msg.header.seq = frame
        msg.header.stamp = rospy.Time(1, 0)
        msg.header.frame_id = "stale_laser"
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = (2.0 * math.pi) / 360.0
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / max(rate_hz, 1e-6)
        msg.range_min = 0.18
        msg.range_max = 20.0
        msg.ranges = [1.0] * 360
        msg.intensities = []
        pub.publish(msg)
        frame += 1
        rate.sleep()

    print("published stale scans to %s frames=%d" % (topic, frame))


if __name__ == "__main__":
    main()
