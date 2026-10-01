#!/usr/bin/env python3
"""Inspect a ROS1 bag on Windows/Linux without installing ROS.

Dependencies: rosbags, numpy, opencv-python-headless (only for PNG export).
Does not alter the input bag. Records all topics and all header timestamps.
"""
import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


def plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name))
                for f in dataclasses.fields(value) if f.name != '__msgtype__'}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [plain(x) for x in value]
    return value


def stamp_ns(header):
    return int(header.stamp.sec) * 1000000000 + int(header.stamp.nanosec)


def point_array(msg):
    """Respect PointField offsets, row padding, byte order and NaN points."""
    kinds = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4',
             6: 'u4', 7: 'f4', 8: 'f8'}
    endian = '>' if msg.is_bigendian else '<'
    fields = {f.name: f for f in msg.fields}
    columns = []
    for name in ('x', 'y', 'z'):
        f = fields[name]
        dtype = np.dtype(endian + kinds[f.datatype])
        column = np.ndarray((msg.height, msg.width), dtype=dtype,
                            buffer=msg.data, offset=f.offset,
                            strides=(msg.row_step, msg.point_step))
        columns.append(column.reshape(-1))
    points = np.column_stack(columns).astype(np.float64)
    return points[np.isfinite(points).all(axis=1)]


def image_array(msg):
    import cv2
    channels = {'bgr8': 3, 'rgb8': 3, 'mono8': 1,
                'bgra8': 4, 'rgba8': 4}.get(msg.encoding)
    if channels is None:
        raise ValueError('Unsupported preview image encoding: ' + msg.encoding)
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding == 'rgb8':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif msg.encoding == 'rgba8':
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif msg.encoding == 'bgra8':
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif msg.encoding == 'mono8':
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def read_extrinsic(path):
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    for i, line in enumerate(lines):
        if line.startswith('T_lidar_camera ('):
            matrix = np.array([[float(v) for v in row.split()]
                               for row in lines[i + 1:i + 5]])
            if matrix.shape != (4, 4):
                raise ValueError('Expected 4 x 4 T_lidar_camera')
            return matrix
    raise ValueError('T_lidar_camera matrix not found in ' + str(path))


def projection_overlay(image, points, matrix, camera_matrix, distortion):
    import cv2
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'image_pointcloud_fusion/scripts'))
    from c32_common import project_xyz
    h, w = image.shape[:2]
    uv, depths, _ = project_xyz(points, matrix, camera_matrix, distortion, w, h)
    uv = np.floor(uv).astype(int)
    result = image.copy()
    if not len(uv):
        return result, 0
    colors = cv2.applyColorMap(np.uint8(np.clip(depths / 20, 0, 1) * 255),
                              cv2.COLORMAP_JET).reshape(-1, 3)
    # Far points first, so the nearer return remains visible at shared pixels.
    for i in np.argsort(depths)[::-1]:
        cv2.circle(result, tuple(uv[i]), 1, tuple(int(c) for c in colors[i]), -1)
    return result, int(len(uv))


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {key: float(value) for key, value in zip(
        ('min', 'p50', 'p95', 'max'), np.percentile(values, [0, 50, 95, 100]))}


def inspect(args):
    args.output.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    headers, records, first_messages, topic_data = {}, {}, {}, {}
    camera_variants, image_variants, cloud_variants = {}, {}, {}
    sampled_images = []
    with Reader(args.bag) as reader:
        report = {'bag': str(args.bag.resolve()), 'size_bytes': args.bag.stat().st_size,
                  'duration_seconds': (reader.end_time - reader.start_time) / 1e9,
                  'start_utc': dt.datetime.fromtimestamp(reader.start_time / 1e9,
                                                         dt.timezone.utc).isoformat(),
                  'start_china': dt.datetime.fromtimestamp(reader.start_time / 1e9,
                      dt.timezone(dt.timedelta(hours=8))).isoformat(),
                  'message_count': reader.message_count, 'topics': topic_data}
        sample_targets = np.linspace(reader.start_time, reader.end_time - 1,
                                     args.samples + 2, dtype=np.int64)[1:-1]
        for connection in reader.connections:
            if connection.msgtype not in typestore.types:
                typestore.register(get_types_from_msg(connection.msgdef.data,
                                                       connection.msgtype))
            info = topic_data.setdefault(connection.topic, {
                'type': connection.msgtype.replace('/msg/', '/'), 'count': 0,
                'connections': [], 'frame_ids': {}})
            info['count'] += connection.msgcount
            info['connections'].append({'callerid': connection.ext.callerid,
                                        'latching': connection.ext.latching})
            headers.setdefault(connection.topic, [])
            records.setdefault(connection.topic, [])
        image_topic = args.image_topic
        lidar_topic = args.lidar_topic
        for connection, record_time, raw in reader.messages():
            topic = connection.topic
            msg = typestore.deserialize_ros1(raw, connection.msgtype)
            records[topic].append(record_time)
            if hasattr(msg, 'header'):
                stamp = stamp_ns(msg.header)
                headers[topic].append(stamp)
                frames = topic_data[topic]['frame_ids']
                frames[msg.header.frame_id] = frames.get(msg.header.frame_id, 0) + 1
            else:
                stamp = record_time
            if topic not in first_messages:
                first_messages[topic] = msg
            if connection.msgtype.endswith('/CameraInfo'):
                values = {f.name: plain(getattr(msg, f.name)) for f in dataclasses.fields(msg)
                          if f.name not in ('header', '__msgtype__')}
                variants = camera_variants.setdefault(topic, {})
                key = json.dumps(values, sort_keys=True)
                variants.setdefault(key, {'configuration': values, 'count': 0})['count'] += 1
            if connection.msgtype.endswith('/Image'):
                meta = {key: plain(getattr(msg, key)) for key in
                        ('height', 'width', 'encoding', 'is_bigendian', 'step')}
                image_variants.setdefault(topic, {})[json.dumps(meta, sort_keys=True)] = meta
            if connection.msgtype.endswith('/PointCloud2'):
                meta = {key: plain(getattr(msg, key)) for key in
                        ('height', 'width', 'fields', 'point_step', 'row_step',
                         'is_dense', 'is_bigendian')}
                cloud_variants.setdefault(topic, {})[json.dumps(meta, sort_keys=True)] = meta
            if topic == image_topic and len(sampled_images) < len(sample_targets):
                if record_time >= sample_targets[len(sampled_images)]:
                    sampled_images.append({'stamp_ns': stamp, 'record_ns': record_time,
                                           'image': image_array(msg)})
        for topic, info in topic_data.items():
            r = np.asarray(records[topic], dtype=np.int64)
            h = np.asarray(headers[topic], dtype=np.int64)
            info['record_rate_hz'] = float((len(r) - 1) * 1e9 / (r[-1] - r[0])) if len(r) > 1 else 0
            if len(h):
                gaps = np.diff(h)
                info['header_rate_hz'] = float((len(h) - 1) * 1e9 / (h[-1] - h[0])) if len(h) > 1 and h[-1] > h[0] else 0
                info['header_interval_ms'] = stats(gaps / 1e6)
                info['header_backwards'] = int(np.count_nonzero(gaps < 0))
                info['header_zero'] = int(np.count_nonzero(h == 0))
                info['record_minus_header_ms'] = stats((r - h) / 1e6) if len(r) == len(h) else None
                info['first_header_ns'] = int(h[0])
                info['last_header_ns'] = int(h[-1])
            if topic in camera_variants:
                info['camera_info_variants'] = list(camera_variants[topic].values())
            if topic in image_variants:
                info['image_variants'] = list(image_variants[topic].values())
            if topic in cloud_variants:
                info['pointcloud_layout_variants'] = list(cloud_variants[topic].values())
                points = point_array(first_messages[topic])
                info['first_cloud_finite_points'] = len(points)
                info['first_cloud_xyz_min'] = points.min(axis=0).tolist() if len(points) else []
                info['first_cloud_xyz_max'] = points.max(axis=0).tolist() if len(points) else []
                info['first_cloud_xyz_percentiles'] = np.percentile(points, [1, 10, 50, 90, 99], axis=0).tolist() if len(points) else []
        if image_topic in headers and lidar_topic in headers:
            image_times = np.sort(np.array(headers[image_topic], dtype=np.int64))
            lidar_times = np.array(headers[lidar_topic], dtype=np.int64)
            nearest = np.array([image_times[np.argmin(np.abs(image_times - t))] - t
                                for t in lidar_times])
            report['time_alignment'] = {
                'definition': 'For each lidar header, nearest image header minus lidar header; no offset applied',
                'signed_image_minus_lidar_ms': stats(nearest / 1e6),
                'absolute_nearest_ms': stats(np.abs(nearest) / 1e6),
                'lidar_pairs_within_50ms': int(np.count_nonzero(np.abs(nearest) < 50e6)),
                'lidar_pairs_within_100ms': int(np.count_nonzero(np.abs(nearest) < 100e6)),
                'lidar_count': int(len(lidar_times))}

    report['contains_tf'] = '/tf' in topic_data or '/tf_static' in topic_data
    report['previews'] = []
    if sampled_images and args.calibration and lidar_topic in headers:
        import cv2
        # Only retain the few nearest clouds, not the whole 1.27 GB recording.
        lidar_times = np.array(headers[lidar_topic], dtype=np.int64)
        wanted = {int(lidar_times[np.argmin(np.abs(lidar_times - sample['stamp_ns']))])
                  for sample in sampled_images}
        clouds = {}
        with Reader(args.bag) as reader:
            connections = [c for c in reader.connections if c.topic == lidar_topic]
            for connection, _, raw in reader.messages(connections=connections):
                msg = typestore.deserialize_ros1(raw, connection.msgtype)
                stamp = stamp_ns(msg.header)
                if stamp in wanted:
                    clouds[stamp] = point_array(msg)
        camera = first_messages[args.camera_info_topic]
        matrix = read_extrinsic(args.calibration)
        report['extrinsic_file_first_matrix'] = matrix.tolist()
        report['extrinsic_rotation_determinant'] = float(np.linalg.det(matrix[:3, :3]))
        report['extrinsic_rotation_orthogonality_max_error'] = float(np.max(np.abs(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3))))
        for i, sample in enumerate(sampled_images):
            nearest = min(clouds, key=lambda s: abs(s - sample['stamp_ns']))
            original = sample['image']
            cv2.imwrite(str(args.output / ('sample_%02d_image.png' % i)), original)
            np.savez_compressed(args.output / ('sample_%02d_points.npz' % i), xyz=clouds[nearest])
            tiles, counts = [], {}
            for label, extrinsic, intrinsics, distortion in (
                ('file T_lidar_camera | K + D', matrix, camera.K.reshape(3, 3), camera.D),
                ('inverse T_lidar_camera | K + D', np.linalg.inv(matrix), camera.K.reshape(3, 3), camera.D),
                ('file T_lidar_camera | P + zero D', matrix, camera.P.reshape(3, 4)[:, :3], np.zeros(5)),
                ('inverse T_lidar_camera | P + zero D', np.linalg.inv(matrix), camera.P.reshape(3, 4)[:, :3], np.zeros(5))):
                overlay, count = projection_overlay(original, clouds[nearest], extrinsic, intrinsics, distortion)
                cv2.rectangle(overlay, (0, 0), (original.shape[1], 42), (0, 0, 0), -1)
                cv2.putText(overlay, label + ' | n=' + str(count), (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
                tiles.append(overlay)
                counts[label] = count
            comparison = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
            filename = 'sample_%02d_projection_comparison.jpg' % i
            cv2.imwrite(str(args.output / filename), comparison)
            report['previews'].append({'image': 'sample_%02d_image.png' % i,
                'comparison': filename, 'image_stamp_ns': sample['stamp_ns'],
                'lidar_stamp_ns': nearest, 'image_minus_lidar_ms': (sample['stamp_ns'] - nearest) / 1e6,
                'projected_point_counts': counts})

    (args.output / 'bag_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'header_timestamps.json').write_text(json.dumps(headers, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'topics'}, ensure_ascii=False, indent=2))
    for topic, info in topic_data.items():
        print('%s | %s | %d msgs | %.3f Hz | frames=%s' % (
            topic, info['type'], info['count'], info['record_rate_hz'], info['frame_ids']))
    print('Full report:', args.output / 'bag_report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--output', type=Path, default=Path('inspection'))
    parser.add_argument('--image-topic', default='/stereo_inertial_publisher/color/image')
    parser.add_argument('--lidar-topic', default='/velodyne_points')
    parser.add_argument('--camera-info-topic', default='/camera_info')
    parser.add_argument('--samples', type=int, default=3)
    inspect(parser.parse_args())
