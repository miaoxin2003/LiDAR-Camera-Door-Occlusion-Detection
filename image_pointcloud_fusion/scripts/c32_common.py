#!/usr/bin/env python3
"""Shared calibration/PointCloud2 helpers for the C32 + OAK deployment.

Geometry functions are ROS-independent so recorded data can exercise them on
Windows. ROS is imported only while loading node parameters.
"""
import cv2
import numpy as np


def xyz_from_cloud(msg, min_range=0.1, max_range=100.0):
    if msg.width == 0 or msg.height == 0:
        return np.empty((0, 3), dtype=np.float64)
    formats = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4',
               6: 'u4', 7: 'f4', 8: 'f8'}
    fields = {field.name: field for field in msg.fields}
    columns = []
    for name in ('x', 'y', 'z'):
        if name not in fields:
            raise ValueError('PointCloud2 missing field: ' + name)
        field = fields[name]
        if field.count != 1:
            raise ValueError('Expected scalar point field: ' + name)
        dtype = np.dtype(('>' if msg.is_bigendian else '<') + formats[field.datatype])
        data = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                          offset=field.offset, strides=(msg.row_step, msg.point_step))
        columns.append(data.reshape(-1))
    xyz = np.column_stack(columns).astype(np.float64)
    distance = np.linalg.norm(xyz, axis=1)
    valid = (np.isfinite(xyz).all(axis=1) & (distance >= min_range) &
             (distance <= max_range))
    return xyz[valid]


def camera_ray_bounds(camera_matrix, distortion, width, height, margin=0):
    """Conservative normalized-ray bounds of this calibrated image domain.

    Polynomial distortion is not a valid global map of the forward hemisphere.
    With the recorded negative k2, rays far outside the real image can fold back
    into it. Bound physical rays before evaluating that polynomial.
    """
    xs = np.linspace(-margin, width - 1 + margin, 17)
    ys = np.linspace(-margin, height - 1 + margin, 17)
    border = np.vstack((
        np.column_stack((xs, np.full_like(xs, -margin))),
        np.column_stack((xs, np.full_like(xs, height - 1 + margin))),
        np.column_stack((np.full_like(ys, -margin), ys)),
        np.column_stack((np.full_like(ys, width - 1 + margin), ys))))
    rays = cv2.undistortPoints(border.reshape(-1, 1, 2), camera_matrix, distortion).reshape(-1, 2)
    if not np.isfinite(rays).all():
        raise ValueError('Image boundary cannot be undistorted with this calibration')
    low, high = rays.min(axis=0), rays.max(axis=0)
    padding = np.maximum(high - low, 1e-6) * 0.05
    return low - padding, high + padding


def project_xyz(points, matrix, camera_matrix, distortion, width, height, margin=0):
    """Return visible pixel coordinates, camera Z, and original point indices."""
    if not len(points):
        return np.empty((0, 2)), np.empty(0), np.empty(0, dtype=int)
    cam = points @ matrix[:3, :3].T + matrix[:3, 3]
    indices = np.flatnonzero(np.isfinite(cam).all(axis=1) & (cam[:, 2] > 0.1))
    if not len(indices):
        return np.empty((0, 2)), np.empty(0), indices
    front = cam[indices]
    ray_low, ray_high = camera_ray_bounds(camera_matrix, distortion, width, height, margin)
    rays = front[:, :2] / front[:, 2:3]
    physical = ((rays >= ray_low) & (rays <= ray_high)).all(axis=1)
    front, indices = front[physical], indices[physical]
    if not len(indices):
        return np.empty((0, 2)), np.empty(0), indices
    uv, _ = cv2.projectPoints(front, np.zeros(3), np.zeros(3), camera_matrix, distortion)
    uv = uv.reshape(-1, 2)
    valid = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= -margin) &
             (uv[:, 0] < width + margin) & (uv[:, 1] >= -margin) &
             (uv[:, 1] < height + margin))
    return uv[valid], front[valid, 2], indices[valid]


def localize_detection_points(points, pixels, depths, bbox, min_points=8,
                              depth_band=0.4, ambiguity_ratio=0.7):
    """Estimate a measured 3D patch inside a 2D detection, without clustering.

    All three input arrays must describe the same visible LiDAR points. A broad
    box containing two similarly supported depth layers is deliberately left
    unlocalized rather than assigning a background or occluder to the class.
    """
    points = np.asarray(points)
    pixels = np.asarray(pixels)
    depths = np.asarray(depths)
    if len(points) != len(pixels) or len(points) != len(depths):
        raise ValueError('Point, pixel and depth arrays must be aligned')
    if depth_band <= 0 or min_points < 1 or not 0 < ambiguity_ratio <= 1:
        raise ValueError('Invalid ROI localization parameters')
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    inside = ((pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) &
              (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2))
    if np.count_nonzero(inside) < min_points:
        return None
    roi_points, roi_depths = points[inside], depths[inside]
    order = np.argsort(roi_depths)
    ordered_depths = roi_depths[order]

    def best_window(sorted_depths):
        left = 0
        best_left, best_right = 0, 0
        for right, value in enumerate(sorted_depths):
            while value - sorted_depths[left] > depth_band:
                left += 1
            if right + 1 - left > best_right - best_left:
                best_left, best_right = left, right + 1
        return best_left, best_right

    left, right = best_window(ordered_depths)
    if right - left < min_points:
        return None
    selected_depth = np.median(ordered_depths[left:right])
    other = ordered_depths[np.abs(ordered_depths - selected_depth) > depth_band]
    if len(other):
        other_left, other_right = best_window(other)
        if other_right - other_left >= ambiguity_ratio * (right - left):
            return None

    selected = roi_points[order[left:right]]
    low, high = np.percentile(selected, [5, 95], axis=0)
    return dict(center=np.median(selected, axis=0), minimum=low, maximum=high,
                point_count=len(selected), roi_point_count=len(roi_points),
                camera_depth=float(selected_depth))


def fit_door_top_line(points, pixels, bbox, tolerance=0.01,
                      min_points=12, min_span_ratio=0.75, depths=None,
                      max_depth_jump=0.12):
    """Fit the measured door top, excluding a separate depth surface.

    A door top may slant in camera depth from one end to the other. ``tolerance``
    limits each inlier's distance to the 3D line, rather than requiring the
    absolute camera depths of both ends to be identical. Adjacent inliers with
    a large depth jump belong to separate surfaces. Endpoints remain within
    the measured inliers.
    """
    points = np.asarray(points, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    if len(points) != len(pixels):
        raise ValueError('Point and pixel arrays must be aligned')
    if depths is not None:
        depths = np.asarray(depths, dtype=np.float64)
        if len(depths) != len(points):
            raise ValueError('Point and depth arrays must be aligned')
    if (tolerance <= 0 or min_points < 2 or not 0 < min_span_ratio <= 1 or
            max_depth_jump <= 0):
        raise ValueError('Invalid door line fitting parameters')
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    inside = ((pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) &
              (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2))
    roi_points, roi_pixels = points[inside], pixels[inside]
    roi_depths = depths[inside] if depths is not None else None
    count = len(roi_points)
    if count < min_points:
        return None

    rng = np.random.default_rng(0)
    pairs = rng.integers(0, count, size=(max(1500, min(4000, count * 30)), 2))
    best = None
    min_pair_span = (x2 - x1) * min_span_ratio * 0.5
    for first, second in pairs:
        if first == second or abs(roi_pixels[first, 0] - roi_pixels[second, 0]) < min_pair_span:
            continue
        direction = roi_points[second] - roi_points[first]
        length = np.linalg.norm(direction)
        if length < 0.1 or abs(direction[2]) / length > 0.1:
            continue
        direction /= length
        residuals = np.linalg.norm(np.cross(roi_points - roi_points[first], direction), axis=1)
        inliers = residuals <= tolerance
        support = int(inliers.sum())
        if support < min_points:
            continue
        span = float(np.ptp(roi_pixels[inliers, 0]))
        if span < (x2 - x1) * min_span_ratio:
            continue
        score = (support, span)
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None:
        return None

    inliers = best[1]
    for _ in range(2):
        selected = roi_points[inliers]
        center = selected.mean(axis=0)
        _, _, axes = np.linalg.svd(selected - center, full_matrices=False)
        direction = axes[0]
        if abs(direction[2]) > 0.1:
            return None
        residuals = np.linalg.norm(np.cross(roi_points - center, direction), axis=1)
        refined = residuals <= tolerance
        if int(refined.sum()) < min_points:
            return None
        inliers = refined
    if roi_depths is not None:
        # A close foreground object can be almost collinear in the image. Keep
        # the longest supported continuous run in horizontal image order.
        ordered = np.flatnonzero(inliers)[np.argsort(roi_pixels[inliers, 0])]
        breaks = np.flatnonzero(np.abs(np.diff(roi_depths[ordered])) > max_depth_jump) + 1
        runs = np.split(ordered, breaks)
        ordered = max(runs, key=lambda run: (len(run),
            float(np.ptp(roi_pixels[run, 0])) if len(run) else 0))
        if len(ordered) < min_points:
            return None
        inliers = np.zeros(count, dtype=bool)
        inliers[ordered] = True
    selected = roi_points[inliers]
    span = float(np.ptp(roi_pixels[inliers, 0]))
    if span < (x2 - x1) * min_span_ratio:
        return None
    center = selected.mean(axis=0)
    _, _, axes = np.linalg.svd(selected - center, full_matrices=False)
    direction = axes[0]
    distances = (selected - center) @ direction
    first = center + direction * distances.min()
    second = center + direction * distances.max()
    if first[1] < second[1]:
        first, second = second, first
    if np.linalg.norm(second - first) < 0.5:
        return None
    return dict(start=first, end=second, point_count=int(inliers.sum()),
                pixel_span=span, length=float(np.linalg.norm(second - first)))


class DoorLineSmoother:
    """Smooth measured endpoints while a continuously detected door is tracked."""

    def __init__(self, alpha=0.35, max_gap=2.5, max_shift=0.5):
        if not 0 < alpha <= 1 or max_gap <= 0 or max_shift <= 0:
            raise ValueError('Invalid door line smoothing parameters')
        self.alpha = alpha
        self.max_gap = max_gap
        self.max_shift = max_shift
        self.previous = None
        self.stamp = None

    def reset(self):
        self.previous = None
        self.stamp = None

    def update(self, line, stamp):
        start, end = line['start'], line['end']
        if self.previous is not None and self.stamp is not None:
            gap = stamp - self.stamp
            shift = np.linalg.norm((start + end - self.previous[0] - self.previous[1]) / 2)
            if 0 < gap <= self.max_gap and shift <= self.max_shift:
                start = self.alpha * start + (1 - self.alpha) * self.previous[0]
                end = self.alpha * end + (1 - self.alpha) * self.previous[1]
        self.previous = (start.copy(), end.copy())
        self.stamp = stamp
        result = line.copy()
        result['start'], result['end'] = start, end
        return result


def estimate_door_occlusion(points, top_left, top_right, ground_z,
                            cell_size=0.15, range_margin=0.15,
                            edge_margin=0.10):
    """Classify virtual door cells using returns along rays from the LiDAR origin.

    Each return defines a ray from (0, 0, 0) in the LiDAR frame. A return
    before the virtual door plane blocks its cell; a return at or beyond the
    plane shows that the ray reached the door position. Cells with no return
    remain unknown. A blocked return wins if a cell contains both kinds.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    left = np.asarray(top_left, dtype=np.float64).reshape(3)
    right = np.asarray(top_right, dtype=np.float64).reshape(3)
    if cell_size <= 0 or range_margin <= 0 or edge_margin < 0:
        raise ValueError('Invalid door occlusion parameters')
    edge = right - left
    width = float(np.linalg.norm(edge[:2]))
    height = float((left[2] + right[2]) / 2 - ground_z)
    usable_width = width - 2 * edge_margin
    usable_height = height - 2 * edge_margin
    if usable_width <= 0 or usable_height <= 0:
        return None
    nx = max(1, int(np.ceil(usable_width / cell_size)))
    nz = max(1, int(np.ceil(usable_height / cell_size)))
    cells = np.zeros((nz, nx), dtype=np.uint8)  # 0 unknown, 1 clear, 2 blocked
    normal = np.array([-edge[1], edge[0], 0.0], dtype=np.float64)
    plane_offset = float(np.dot(normal, left))
    if abs(plane_offset) < 1e-6:
        return None
    denominator = points @ normal
    valid = np.isfinite(points).all(axis=1) & (np.abs(denominator) > 1e-9)
    if not np.any(valid):
        return dict(cells=cells, blocked=0, clear=0, unknown=nx * nz,
                    coverage=0.0, occlusion_ratio=float('nan'))
    rays = points[valid]
    factors = plane_offset / denominator[valid]
    forward = np.isfinite(factors) & (factors > 0)
    rays, factors = rays[forward], factors[forward]
    intersections = rays * factors[:, None]
    u = (intersections[:, :2] - left[:2]) @ edge[:2] / (width * width)
    # The top can slope in Z. Define the vertical coordinate below its
    # interpolated height so the grid follows the actual fitted top line.
    column_height = left[2] + u * edge[2] - ground_z
    drop = left[2] + u * edge[2] - intersections[:, 2]
    within = ((u * width >= edge_margin) &
              (u * width < width - edge_margin) &
              (drop >= edge_margin) &
              (drop < column_height - edge_margin) &
              (column_height > 2 * edge_margin))
    if np.any(within):
        u, drop, rays, intersections = (array[within] for array in
                                        (u, drop, rays, intersections))
        x = np.clip(((u * width - edge_margin) /
                     usable_width * nx).astype(int), 0, nx - 1)
        z = np.clip(((drop - edge_margin) /
                     usable_height * nz).astype(int), 0, nz - 1)
        # Positive separation means the return is closer than the plane.
        separation = np.linalg.norm(intersections, axis=1) - np.linalg.norm(rays, axis=1)
        labels = np.where(separation > range_margin, 2, 1).astype(np.uint8)
        np.maximum.at(cells, (z, x), labels)
    blocked = int(np.count_nonzero(cells == 2))
    clear = int(np.count_nonzero(cells == 1))
    observed = blocked + clear
    return dict(cells=cells, blocked=blocked, clear=clear,
                unknown=nx * nz - observed, coverage=observed / (nx * nz),
                occlusion_ratio=blocked / observed if observed else float('nan'))


def calibrated_geometry(matrix, camera_matrix, distortion, invert=False,
                        rectified=False, projection=None, rectification=None):
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    intrinsic = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if not all(np.isfinite(value).all() for value in (matrix, intrinsic, distortion)):
        raise ValueError('Non-finite calibration data')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('Invalid homogeneous transform last row')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
        raise ValueError('Extrinsic rotation must be orthonormal with determinant +1')
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError('Invalid focal length')
    if len(distortion) not in (4, 5, 8, 12, 14):
        raise ValueError('Unsupported OpenCV distortion coefficient count')
    matrix = np.linalg.inv(matrix) if invert else matrix.copy()
    if rectified:
        projection = np.asarray(projection, dtype=np.float64).reshape(3, 4)
        rectification = np.asarray(rectification, dtype=np.float64).reshape(3, 3)
        if not np.allclose(projection[:, 3], 0):
            raise ValueError('This RGB adapter requires P[:,3] == 0')
        intrinsic = projection[:, :3].copy()
        distortion = np.zeros(5)
        matrix[:3, :] = rectification @ matrix[:3, :]
    return matrix, intrinsic, distortion


def load_calibration():
    import rospy
    mode = rospy.get_param('~distortion_model', 'plumb_bob')
    if mode not in ('plumb_bob', 'rational_polynomial'):
        raise ValueError('Unsupported distortion_model: ' + mode)
    matrix, intrinsic, distortion = calibrated_geometry(
        rospy.get_param('~extrinsic_matrix'), rospy.get_param('~camera_matrix'),
        rospy.get_param('~dist_coeffs'),
        invert=rospy.get_param('~invert_extrinsic', True),
        rectified=rospy.get_param('~image_is_rectified', False),
        projection=rospy.get_param('~projection_matrix'),
        rectification=rospy.get_param('~rectification_matrix'))
    width = int(rospy.get_param('~image_width'))
    height = int(rospy.get_param('~image_height'))
    rospy.logwarn('Calibration is a candidate: verify a static target before accepting fusion. '
                  'invert_extrinsic=%s, image_is_rectified=%s',
                  rospy.get_param('~invert_extrinsic', True),
                  rospy.get_param('~image_is_rectified', False))
    rospy.loginfo('Using lidar-to-optical transform:\n%s', matrix)
    return matrix, intrinsic, distortion, width, height


def check_input_headers(image_msg, cloud_msg, width, height, camera_frame, lidar_frame):
    if image_msg.width != width or image_msg.height != height:
        raise ValueError('Image is %dx%d; calibration is %dx%d. Match the camera mode or recalibrate.' %
                         (image_msg.width, image_msg.height, width, height))
    if camera_frame and image_msg.header.frame_id != camera_frame:
        raise ValueError('Image frame_id %r differs from calibration frame %r' %
                         (image_msg.header.frame_id, camera_frame))
    if lidar_frame and cloud_msg.header.frame_id != lidar_frame:
        raise ValueError('PointCloud frame_id %r differs from calibration frame %r' %
                         (cloud_msg.header.frame_id, lidar_frame))
