#!/usr/bin/env python3
"""Projection-only check: no Torch, Ultralytics, Open3D or TF required."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2
from c32_common import load_calibration, xyz_from_cloud, project_xyz, check_input_headers


class Projection:
    def __init__(self):
        rospy.init_node('c32_projection')
        self.matrix, self.intrinsic, self.distortion, self.width, self.height = load_calibration()
        self.camera_frame = rospy.get_param('~camera_frame', 'oak_rgb_camera_optical_frame')
        self.lidar_frame = rospy.get_param('~lidar_frame', 'velodyne')
        self.bridge = CvBridge()
        self.point_stride = max(1, int(rospy.get_param('~projection_point_stride', 2)))
        self.max_range = float(rospy.get_param('~max_range', 20.0))
        self.color_range = max(0.1, float(rospy.get_param('~color_range', 8.0)))
        self.last_pair_wall = None
        self.started_wall = time.monotonic()
        self.overlay_pub = rospy.Publisher('~overlay_image', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('~depth_image', Image, queue_size=1)
        self.image_sub = Subscriber(rospy.get_param('~camera_topic'), Image,
                                    queue_size=1, buff_size=16 * 1024 * 1024)
        self.cloud_sub = Subscriber(rospy.get_param('~lidar_topic'), PointCloud2,
                                    queue_size=1, buff_size=16 * 1024 * 1024)
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.cloud_sub],
                    queue_size=10, slop=float(rospy.get_param('~sync_slop', 0.05)))
        self.sync.registerCallback(self.callback)
        rospy.loginfo('Projection ready; waiting for matched image/cloud headers.')

    def callback(self, image_msg, cloud_msg):
        try:
            self.last_pair_wall = time.monotonic()
            check_input_headers(image_msg, cloud_msg, self.width, self.height,
                                self.camera_frame, self.lidar_frame)
            image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8').copy()
            xyz = xyz_from_cloud(cloud_msg, max_range=self.max_range)[::self.point_stride]
            uv, depths, _ = project_xyz(xyz, self.matrix, self.intrinsic, self.distortion,
                                         self.width, self.height)
            depth_image = np.full((self.height, self.width), np.inf, dtype=np.float32)
            if len(uv):
                pixels = np.floor(uv).astype(np.int32)
                np.minimum.at(depth_image, (pixels[:, 1], pixels[:, 0]), depths)
                colors = cv2.applyColorMap(np.uint8(np.clip(depths / self.color_range, 0, 1) * 255),
                                          cv2.COLORMAP_JET).reshape(-1, 3)
                for index in np.argsort(depths)[::-1]:
                    cv2.circle(image, tuple(pixels[index]), 1,
                               tuple(int(c) for c in colors[index]), -1)
            depth_image[~np.isfinite(depth_image)] = 0
            dt_ms = (image_msg.header.stamp - cloud_msg.header.stamp).to_sec() * 1000
            cv2.putText(image, 'visible=%d  image-lidar=%+.1f ms' % (len(uv), dt_ms),
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
            overlay = self.bridge.cv2_to_imgmsg(image, 'bgr8')
            overlay.header = image_msg.header
            self.overlay_pub.publish(overlay)
            depth = self.bridge.cv2_to_imgmsg(depth_image, '32FC1')
            depth.header = image_msg.header
            self.depth_pub.publish(depth)
            rospy.loginfo_throttle(5, 'Projection: %d visible points, header difference %.1f ms' % (len(uv), dt_ms))
        except Exception as error:
            rospy.logerr_throttle(3, 'Projection failed: %s' % error)

    def run(self):
        # Wall clock keeps the diagnostic useful even when /clock is paused.
        while not rospy.is_shutdown():
            last = self.last_pair_wall if self.last_pair_wall is not None else self.started_wall
            if time.monotonic() - last > 10:
                rospy.logwarn('No synchronized pair for 10s. Check playback, input topics, and header stamps.')
            time.sleep(5)


if __name__ == '__main__':
    Projection().run()

