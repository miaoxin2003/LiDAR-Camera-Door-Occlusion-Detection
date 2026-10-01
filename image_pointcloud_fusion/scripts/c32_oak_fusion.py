#!/usr/bin/env python3
"""CPU deployment adapter around the project's original post-fusion algorithm.

The inherited code still performs radial Open3D DBSCAN, projects 3D cluster
boxes, and assigns YOLO classes by 2D IoU. This adapter adds recorded calibration,
pairing before inference, a bounded worker queue, CPU controls and diagnostics.
"""
import os
import sys
import threading
import time
from queue import Queue, Empty, Full

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import open3d as o3d
import rospy
import torch
from ultralytics import YOLO
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
from c32_common import (load_calibration, xyz_from_cloud, project_xyz,
                        check_input_headers, camera_ray_bounds,
                        localize_detection_points, fit_door_top_line,
                        DoorLineSmoother, estimate_door_occlusion)
from imgpc_fusion_detection import MultiModalFusion


class CPUFusion(MultiModalFusion):
    def __init__(self):
        # The original initializer starts independent unsynchronized workers and
        # installs hardcoded RealSense calibration. Initialize the adapter here.
        rospy.init_node('c32_fusion')
        (self.lidar_to_camera, self.camera_matrix, self.dist_coeffs,
         self.image_width, self.image_height) = load_calibration()
        self.camera_frame = rospy.get_param('~camera_frame', 'oak_rgb_camera_optical_frame')
        self.lidar_frame = rospy.get_param('~lidar_frame', 'velodyne')
        self.sensor_model = rospy.get_param('~sensor_model', 'HDL-32E')
        if self.sensor_model not in ('VLP-16', 'HDL-32E', 'HDL-64E'):
            raise ValueError('sensor_model selects a clustering schedule; use HDL-32E for the supplied C32 starting configuration')
        self.leaf = max(1, int(rospy.get_param('~leaf', 1)))
        self.z_axis_min = float(rospy.get_param('~z_axis_min', -0.5))
        self.z_axis_max = float(rospy.get_param('~z_axis_max', 1.8))
        self.cluster_size_min = max(1, int(rospy.get_param('~cluster_size_min', 3)))
        self.cluster_size_max = int(rospy.get_param('~cluster_size_max', 5000))
        self.fusion_iou_threshold = float(rospy.get_param('~fusion_iou_threshold', 0.3))
        self.region_max = 10
        self.regions = np.zeros(100, dtype=int)
        self.init_regions()
        self.max_range = float(rospy.get_param('~max_range', 15.0))
        self.voxel_size = float(rospy.get_param('~voxel_size', 0.05))
        self.crop_to_camera = bool(rospy.get_param('~crop_to_camera', True))
        self.frustum_margin = int(rospy.get_param('~frustum_margin_pixels', 32))
        self.projection_ray_bounds = camera_ray_bounds(self.camera_matrix,
            self.dist_coeffs, self.image_width, self.image_height, self.frustum_margin)
        self.max_hz = float(rospy.get_param('~max_processing_hz', 2.0))
        self.marker_lifetime = float(rospy.get_param('~marker_lifetime', 1.5))
        self.roi_localization_enabled = bool(rospy.get_param('~roi_localization_enabled', True))
        self.roi_min_points = int(rospy.get_param('~roi_min_points', 8))
        self.roi_depth_band = float(rospy.get_param('~roi_depth_band', 0.4))
        self.roi_ambiguity_ratio = float(rospy.get_param('~roi_ambiguity_ratio', 0.7))
        self.door_geometry_enabled = bool(rospy.get_param('~door_geometry_enabled', True))
        self.door_class_name = rospy.get_param('~door_class_name', 'menkuang')
        self.door_line_tolerance = float(rospy.get_param('~door_line_tolerance_m', 0.01))
        self.door_line_min_points = int(rospy.get_param('~door_line_min_points', 12))
        self.door_line_min_span_ratio = float(rospy.get_param('~door_line_min_span_ratio', 0.75))
        self.door_max_depth_jump = float(rospy.get_param('~door_max_depth_jump_m', 0.12))
        self.door_smoothing_alpha = float(rospy.get_param('~door_smoothing_alpha', 0.35))
        self.door_ground_z = float(rospy.get_param('~door_ground_z', -1.17))
        self.occlusion_cell_size = float(rospy.get_param('~door_occlusion_cell_size_m', 0.15))
        self.occlusion_range_margin = float(rospy.get_param('~door_occlusion_range_margin_m', 0.15))
        self.occlusion_edge_margin = float(rospy.get_param('~door_occlusion_edge_margin_m', 0.10))
        self.occlusion_min_coverage = float(rospy.get_param('~door_occlusion_min_coverage', 0.05))
        if self.z_axis_min >= self.z_axis_max or not 0 <= self.fusion_iou_threshold <= 1:
            raise ValueError('Invalid height range or IoU threshold')
        if (self.roi_min_points < 1 or self.roi_depth_band <= 0 or
                not 0 < self.roi_ambiguity_ratio <= 1):
            raise ValueError('Invalid YOLO ROI localization parameters')
        if (self.door_line_tolerance <= 0 or self.door_line_min_points < 2 or
                not 0 < self.door_line_min_span_ratio <= 1 or
                self.door_max_depth_jump <= 0 or
                not 0 < self.door_smoothing_alpha <= 1 or
                not np.isfinite(self.door_ground_z)):
            raise ValueError('Invalid door geometry parameters')
        if (self.occlusion_cell_size <= 0 or self.occlusion_range_margin <= 0 or
                self.occlusion_edge_margin < 0 or
                not 0 <= self.occlusion_min_coverage <= 1):
            raise ValueError('Invalid door occlusion parameters')
        self.door_smoother = DoorLineSmoother(alpha=self.door_smoothing_alpha)
        self.bridge = CvBridge()
        cv2.setNumThreads(1)
        torch.set_num_threads(max(1, int(rospy.get_param('~torch_threads', 2))))
        model_path = os.path.expanduser(rospy.get_param('~yolo_model_path'))
        if not os.path.isfile(model_path):
            raise FileNotFoundError('Model missing (no automatic download): ' + model_path)
        if rospy.get_param('~device', 'cpu') != 'cpu':
            raise ValueError('This deployment profile is for CPU; set device=cpu')
        self.predict_options = dict(device='cpu',
            imgsz=int(rospy.get_param('~imgsz', 416)),
            conf=float(rospy.get_param('~yolo_confidence', 0.35)),
            max_det=int(rospy.get_param('~max_detections', 50)),
            verbose=False, half=False)
        classes = rospy.get_param('~classes', [])
        if classes:
            self.predict_options['classes'] = classes
        rospy.loginfo('Loading local YOLO model on CPU: %s', model_path)
        self.model = YOLO(model_path)
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8), **self.predict_options)

        self.cloud_filtered_pub = rospy.Publisher('~cloud_filtered', PointCloud2, queue_size=1)
        self.cluster_marker_pub = rospy.Publisher('~cluster_markers', MarkerArray, queue_size=1)
        self.debug_marker_pub = rospy.Publisher('~all_cluster_markers', MarkerArray, queue_size=1)
        self.pose_array_pub = rospy.Publisher('~cluster_poses', PoseArray, queue_size=1)
        self.detection_image_pub = rospy.Publisher('~detection_image', Image, queue_size=1)
        self.yolo_image_pub = rospy.Publisher('~yolo_image', Image, queue_size=1)
        self.bbox3d_pub = rospy.Publisher('~bounding_boxes3d', BoundingBoxArray, queue_size=1)
        self.roi_marker_pub = rospy.Publisher('~yolo_point_markers', MarkerArray, queue_size=1)
        self.roi_bbox3d_pub = rospy.Publisher('~yolo_point_boxes3d', BoundingBoxArray, queue_size=1)
        self.door_line_pub = rospy.Publisher('~door_top_line_markers', MarkerArray, queue_size=1)
        self.door_plane_pub = rospy.Publisher('~door_plane_markers', MarkerArray, queue_size=1)
        self.occlusion_marker_pub = rospy.Publisher('~door_occlusion_markers', MarkerArray, queue_size=1)
        self.occlusion_ratio_pub = rospy.Publisher('~door_occlusion_ratio', Float32, queue_size=1)
        self.occlusion_coverage_pub = rospy.Publisher('~door_occlusion_coverage', Float32, queue_size=1)

        self.pairs = Queue(maxsize=1)
        self.last_submit = -float('inf')
        self.last_pair_wall = None
        self.started_wall = time.monotonic()
        self.worker = threading.Thread(target=self.work, daemon=True)
        self.worker.start()
        self.image_sub = Subscriber(rospy.get_param('~camera_topic'), Image,
                                    queue_size=1, buff_size=16 * 1024 * 1024)
        self.lidar_sub = Subscriber(rospy.get_param('~lidar_topic'), PointCloud2,
                                    queue_size=1, buff_size=16 * 1024 * 1024)
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub],
                    queue_size=10, slop=float(rospy.get_param('~sync_slop', 0.05)))
        self.sync.registerCallback(self.enqueue)
        rospy.loginfo('CPU fusion ready. Pairs are matched BEFORE inference; target <= %.2f processed pairs/s.', self.max_hz)

    def enqueue(self, image_msg, cloud_msg):
        now = time.monotonic()
        self.last_pair_wall = now
        if self.max_hz > 0 and now - self.last_submit < 1.0 / self.max_hz:
            return
        self.last_submit = now
        try:
            check_input_headers(image_msg, cloud_msg, self.image_width, self.image_height,
                                self.camera_frame, self.lidar_frame)
        except ValueError as error:
            rospy.logerr_throttle(3, str(error))
            return
        try:
            self.pairs.put_nowait((image_msg, cloud_msg))
        except Full:
            try:
                self.pairs.get_nowait()
            except Empty:
                pass
            try:
                self.pairs.put_nowait((image_msg, cloud_msg))
            except Full:
                pass

    def work(self):
        while not rospy.is_shutdown():
            try:
                pair = self.pairs.get(timeout=0.2)
            except Empty:
                continue
            try:
                self.process_pair(*pair)
            except Exception as error:
                rospy.logerr_throttle(3, 'CPU fusion failed: %s' % error)

    def process_pair(self, image_msg, cloud_msg):
        start = time.monotonic()
        image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8').copy()
        raw_points = xyz_from_cloud(cloud_msg, max_range=self.max_range)
        points = raw_points
        count_before = len(points)
        projected_pixels = np.empty((0, 2))
        projected_depths = np.empty(0)
        projected_points = np.empty((0, 3))
        if self.crop_to_camera or self.roi_localization_enabled or self.door_geometry_enabled:
            projected_pixels, projected_depths, indices = project_xyz(
                points, self.lidar_to_camera, self.camera_matrix,
                self.dist_coeffs, self.image_width, self.image_height, self.frustum_margin)
            projected_points = points[indices]
            if self.crop_to_camera:
                points = projected_points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if self.voxel_size > 0 and len(points):
            pcd = pcd.voxel_down_sample(self.voxel_size)
        if len(pcd.points):
            try:
                filtered, clusters, centroids, boxes = self.process_point_cloud(pcd)
            except Exception as error:
                rospy.logerr_throttle(3, 'Point-cloud clustering failed; trying YOLO ROI localization: %s' % error)
                filtered, clusters, centroids, boxes = pcd, [], [], []
        else:
            filtered, clusters, centroids, boxes = pcd, [], [], []
        if self.cloud_filtered_pub.get_num_connections():
            self.cloud_filtered_pub.publish(self.numpy_to_ros(cloud_msg.header,
                                                               np.asarray(filtered.points)))
        results = self.model.predict(image, **self.predict_options)
        processed = image.copy()
        detections = []
        for result in results:
            for box in result.boxes:
                xyxy = tuple(int(v) for v in box.xyxy[0].cpu().tolist())
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                name = result.names[class_id]
                detections.append(dict(bbox=xyxy, confidence=confidence,
                                       class_id=class_id, class_name=name))
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed, '%s %.2f' % (name, confidence), (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
        msg = self.bridge.cv2_to_imgmsg(processed, 'bgr8')
        msg.header = image_msg.header
        self.yolo_image_pub.publish(msg)
        lidar_data = dict(header=cloud_msg.header, clusters=clusters, centroids=centroids, boxes=boxes)
        image_data = dict(header=image_msg.header, detections=detections, processed_image=processed)
        self.publish_debug_boxes(cloud_msg.header, boxes)
        # The original project's cluster-box / YOLO IoU association is reused.
        matched_detection_ids = self.fuse_lidar_and_image(lidar_data, image_data)
        roi_count = self.publish_roi_detections(cloud_msg.header, detections,
            matched_detection_ids, projected_points, projected_pixels, projected_depths)
        door_count = self.publish_door_geometry(cloud_msg.header, detections,
            projected_points, projected_pixels, projected_depths, raw_points)
        elapsed = time.monotonic() - start
        dt_ms = (image_msg.header.stamp - cloud_msg.header.stamp).to_sec() * 1000
        rospy.loginfo_throttle(3, 'Fusion: %.3fs/pair, dt=%+.1fms, points=%d -> %d, clusters=%d, YOLO=%d, ROI3D=%d, DOOR=%d' %
            (elapsed, dt_ms, count_before, len(filtered.points), len(boxes), len(detections), roi_count, door_count))

    def publish_roi_detections(self, header, detections, matched_detection_ids,
                               points, pixels, depths):
        markers = self.clear_array(header)
        boxes = BoundingBoxArray()
        boxes.header = header
        if self.roi_localization_enabled:
            for detection in detections:
                if id(detection) in matched_detection_ids:
                    continue
                estimate = localize_detection_points(points, pixels, depths,
                    detection['bbox'], self.roi_min_points, self.roi_depth_band,
                    self.roi_ambiguity_ratio)
                if estimate is None:
                    continue
                index = len(boxes.boxes)
                bounds = dict(min=estimate['minimum'], max=estimate['maximum'])
                marker = self.cube(header, bounds, index, 'yolo_point_support',
                                   (0.1, 0.7, 1.0))
                marker.scale.x, marker.scale.y, marker.scale.z = np.maximum(
                    estimate['maximum'] - estimate['minimum'], 0.05).tolist()
                marker.color.a = 0.4
                markers.markers.append(marker)
                box = BoundingBox()
                box.header = header
                center = (estimate['minimum'] + estimate['maximum']) / 2
                box.pose.position.x, box.pose.position.y, box.pose.position.z = center.tolist()
                box.pose.orientation.w = 1.0
                box.dimensions.x, box.dimensions.y, box.dimensions.z = (
                    estimate['maximum'] - estimate['minimum']).tolist()
                box.value = detection['class_id']
                box.label = detection['class_id']
                boxes.boxes.append(box)
        self.roi_marker_pub.publish(markers)
        self.roi_bbox3d_pub.publish(boxes)
        return len(boxes.boxes)

    def publish_door_geometry(self, header, detections, points, pixels, depths, raw_points):
        lines = self.clear_array(header)
        planes = self.clear_array(header)
        count = 0
        door_top = None
        if self.door_geometry_enabled:
            candidates = []
            for detection in detections:
                if detection['class_name'] != self.door_class_name:
                    continue
                line = fit_door_top_line(points, pixels, detection['bbox'],
                    self.door_line_tolerance, self.door_line_min_points,
                    self.door_line_min_span_ratio, depths, self.door_max_depth_jump)
                if line is None:
                    continue
                candidates.append(line)
            if candidates:
                # Show the best supported measured door instead of switching
                # marker IDs when YOLO returns detections in a different order.
                line = max(candidates, key=lambda item: (item['point_count'], item['pixel_span']))
                line = self.door_smoother.update(line, header.stamp.to_sec())
                top_left, top_right = line['start'], line['end']
                if min(top_left[2], top_right[2]) > self.door_ground_z + 0.5:
                    bottom_left = top_left.copy()
                    bottom_right = top_right.copy()
                    bottom_left[2] = bottom_right[2] = self.door_ground_z

                    marker = Marker()
                    marker.header = header
                    marker.ns = 'measured_door_top_line'
                    marker.id = count
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.06
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.9, 0.0, 1.0
                    marker.points = [Point(*top_left.tolist()), Point(*top_right.tolist())]
                    marker.lifetime = rospy.Duration(self.marker_lifetime)
                    lines.markers.append(marker)

                    surface = Marker()
                    surface.header = header
                    surface.ns = 'door_position_plane'
                    surface.id = count
                    surface.type = Marker.TRIANGLE_LIST
                    surface.action = Marker.ADD
                    surface.pose.orientation.w = 1.0
                    surface.scale.x = surface.scale.y = surface.scale.z = 1.0
                    surface.color.r, surface.color.g, surface.color.b, surface.color.a = 0.35, 0.85, 1.0, 0.35
                    surface.points = [Point(*p.tolist()) for p in
                    (top_left, bottom_left, bottom_right,
                     top_left, bottom_right, top_right)]
                    surface.lifetime = rospy.Duration(self.marker_lifetime)
                    planes.markers.append(surface)
                    count = 1
                    door_top = (top_left, top_right)
        if not count:
            self.door_smoother.reset()
        self.door_line_pub.publish(lines)
        self.door_plane_pub.publish(planes)
        self.publish_door_occlusion(header, door_top, raw_points)
        return count

    def publish_door_occlusion(self, header, door_top, raw_points):
        markers = self.clear_array(header)
        if door_top is None:
            self.occlusion_ratio_pub.publish(Float32(float('nan')))
            self.occlusion_coverage_pub.publish(Float32(float('nan')))
            self.occlusion_marker_pub.publish(markers)
            return
        left, right = door_top
        estimate = estimate_door_occlusion(raw_points, left, right, self.door_ground_z,
            self.occlusion_cell_size, self.occlusion_range_margin,
            self.occlusion_edge_margin)
        if estimate is None:
            self.occlusion_ratio_pub.publish(Float32(float('nan')))
            self.occlusion_coverage_pub.publish(Float32(float('nan')))
            self.occlusion_marker_pub.publish(markers)
            return
        coverage = estimate['coverage']
        ratio = estimate['occlusion_ratio'] if coverage >= self.occlusion_min_coverage else float('nan')
        self.occlusion_ratio_pub.publish(Float32(ratio))
        self.occlusion_coverage_pub.publish(Float32(coverage))

        cells = estimate['cells']
        nz, nx = cells.shape
        edge = right - left
        width = np.linalg.norm(edge[:2])
        height = (left[2] + right[2]) / 2 - self.door_ground_z
        usable_width = width - 2 * self.occlusion_edge_margin
        usable_height = height - 2 * self.occlusion_edge_margin
        normal = np.array([-edge[1], edge[0], 0.0])
        normal /= np.linalg.norm(normal)
        offset = -np.sign(np.dot(normal, left)) * normal * 0.025

        blocked = Marker()
        blocked.header = header
        blocked.ns = 'confirmed_door_occlusion'
        blocked.id = 0
        blocked.type = Marker.TRIANGLE_LIST
        blocked.action = Marker.ADD
        blocked.pose.orientation.w = 1.0
        blocked.scale.x = blocked.scale.y = blocked.scale.z = 1.0
        blocked.color.r, blocked.color.g, blocked.color.b, blocked.color.a = 1.0, 0.12, 0.08, 0.75
        blocked.lifetime = rospy.Duration(self.marker_lifetime)

        def corner(u, drop):
            point = left + u * edge + offset
            point[2] -= drop
            return Point(*point.tolist())

        for iz, ix in np.argwhere(cells == 2):
            u0 = (self.occlusion_edge_margin + ix * usable_width / nx) / width
            u1 = (self.occlusion_edge_margin + (ix + 1) * usable_width / nx) / width
            d0 = self.occlusion_edge_margin + iz * usable_height / nz
            d1 = self.occlusion_edge_margin + (iz + 1) * usable_height / nz
            blocked.points.extend((corner(u0, d0), corner(u0, d1), corner(u1, d1),
                                   corner(u0, d0), corner(u1, d1), corner(u1, d0)))
        if blocked.points:
            markers.markers.append(blocked)

        label = Marker()
        label.header = header
        label.ns = 'door_occlusion_status'
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.orientation.w = 1.0
        label.pose.position = Point(*((left + right) / 2 + np.array([0, 0, 0.25])).tolist())
        label.scale.z = 0.18
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = ('Occlusion %.0f%% | observed %.0f%%' % (ratio * 100, coverage * 100)
                      if np.isfinite(ratio) else
                      'Occlusion unknown | observed %.0f%%' % (coverage * 100))
        label.lifetime = rospy.Duration(self.marker_lifetime)
        markers.markers.append(label)
        self.occlusion_marker_pub.publish(markers)

    @staticmethod
    def clear_array(header):
        marker = Marker()
        marker.header = header
        marker.action = Marker.DELETEALL
        return MarkerArray(markers=[marker])

    def clear_markers(self, header):
        self.cluster_marker_pub.publish(self.clear_array(header))
        self.pose_array_pub.publish(PoseArray(header=header))

    def cube(self, header, box, index, namespace, rgb):
        marker = Marker()
        marker.header = header
        marker.ns = namespace
        marker.id = index
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        center = (box['min'] + box['max']) / 2
        size = np.maximum(box['max'] - box['min'], 0.01)
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = center.tolist()
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = size.tolist()
        marker.color.r, marker.color.g, marker.color.b = rgb
        marker.color.a = 0.25
        marker.lifetime = rospy.Duration(self.marker_lifetime)
        return marker

    def publish_debug_boxes(self, header, boxes):
        if self.debug_marker_pub.get_num_connections():
            array = self.clear_array(header)
            for index, box in enumerate(boxes):
                array.markers.append(self.cube(header, box, index, 'all_clusters', (0.0, 1.0, 1.0)))
            self.debug_marker_pub.publish(array)

    def publish_cluster_markers(self, header, clusters, boxes, classified_clusters):
        array = self.clear_array(header)
        poses = PoseArray(header=header)
        for index, classified in enumerate(classified_clusters or []):
            box = boxes[classified['cluster_idx']]
            cube = self.cube(header, box, index, 'classified_clusters', (1.0, 0.3, 0.0))
            array.markers.append(cube)
            label = Marker()
            label.header = header
            label.ns = 'classified_labels'
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = cube.pose.position.x
            label.pose.position.y = cube.pose.position.y
            label.pose.position.z = float(box['max'][2]) + 0.2
            label.pose.orientation.w = 1.0
            label.scale.z = 0.22
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = '%s %.2f' % (classified['class_name'], classified['confidence'])
            label.lifetime = rospy.Duration(self.marker_lifetime)
            array.markers.append(label)
            pose = Pose()
            centroid = np.mean(np.asarray(clusters[classified['cluster_idx']].points), axis=0)
            pose.position.x, pose.position.y, pose.position.z = centroid.tolist()
            pose.orientation.w = 1.0
            poses.poses.append(pose)
        self.cluster_marker_pub.publish(array)
        self.pose_array_pub.publish(poses)

    def run(self):
        while not rospy.is_shutdown():
            last = self.last_pair_wall if self.last_pair_wall is not None else self.started_wall
            if time.monotonic() - last > 10:
                rospy.logwarn('No matched input for 10s. Check sensor topics, playback and header stamps.')
            time.sleep(5)


if __name__ == '__main__':
    CPUFusion().run()
