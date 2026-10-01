#!/usr/bin/env python3
"""Run on the Linux ROS1 robot, with the same Python used by the chosen node."""
import argparse
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('projection', 'fusion'), default='projection')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--output', type=Path, default=Path('robot_checks'))
    args = parser.parse_args()
    report = {'platform': platform.platform(), 'machine': platform.machine(),
              'python': sys.version, 'executable': sys.executable,
              'ROS_DISTRO': os.environ.get('ROS_DISTRO'),
              'ROS_MASTER_URI': os.environ.get('ROS_MASTER_URI'),
              'mode': args.mode, 'modules': {}, 'errors': []}
    print(json.dumps({k: v for k, v in report.items() if k not in ('modules', 'errors')}, indent=2))
    modules = ['numpy', 'cv2', 'rospy', 'cv_bridge', 'message_filters', 'sensor_msgs.msg']
    if args.mode == 'fusion':
        modules += ['torch', 'torchvision', 'ultralytics', 'open3d',
                    'jsk_recognition_msgs.msg', 'visualization_msgs.msg', 'geometry_msgs.msg']
    for name in modules:
        try:
            module = importlib.import_module(name)
            version = str(getattr(module, '__version__', 'ROS/system'))
            report['modules'][name] = {'version': version, 'path': getattr(module, '__file__', '')}
            print('OK import:', name, version)
        except Exception as error:
            report['errors'].append('%s: %s' % (name, error))
            print('FAIL import:', name, error)
    if not report['errors']:
        import cv2
        import numpy as np
        from cv_bridge import CvBridge
        try:
            bridge = CvBridge()
            test_image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
            ros_image = bridge.cv2_to_imgmsg(test_image, 'bgr8')
            if not np.array_equal(test_image, bridge.imgmsg_to_cv2(ros_image, 'bgr8')):
                raise RuntimeError('cv_bridge changed the test image')
            report['cv_bridge_roundtrip'] = 'PASS'
            print('PASS cv_bridge roundtrip (tests the NumPy / ROS binary boundary)')
        except Exception as error:
            report['errors'].append('cv_bridge roundtrip: ' + str(error))
        if args.mode == 'fusion':
            try:
                import open3d as o3d
                import torch
                from ultralytics import YOLO
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(np.array([
                    [0, 0, 0], [.01, 0, 0], [.02, 0, 0],
                    [1, 0, 0], [1.01, 0, 0], [1.02, 0, 0]]))
                labels = np.asarray(cloud.cluster_dbscan(eps=.05, min_points=2))
                if len(set(labels.tolist())) != 2 or np.any(labels < 0):
                    raise RuntimeError('Open3D clustering smoke check failed')
                print('PASS Open3D DBSCAN')
                torch.set_num_threads(2)
                report['torch_cuda_available'] = torch.cuda.is_available()
                model_path = args.model or Path(__file__).resolve().parents[1] / 'best.pt'
                if not model_path.is_file():
                    raise FileNotFoundError(str(model_path))
                image = cv2.imread(str(args.image)) if args.image else np.zeros((480, 640, 3), np.uint8)
                if image is None:
                    raise ValueError('Cannot read --image')
                model = YOLO(str(model_path))
                options = dict(device='cpu', imgsz=640, conf=.35, half=False, verbose=False)
                model.predict(image, **options)
                start = time.monotonic()
                results = model.predict(image, **options)
                report['cpu_inference_seconds_after_warmup'] = time.monotonic() - start
                report['detections'] = []
                for result in results:
                    for box in result.boxes:
                        xyxy = [int(v) for v in box.xyxy[0].tolist()]
                        name = result.names[int(box.cls[0].item())]
                        conf = float(box.conf[0].item())
                        report['detections'].append({'class': name, 'confidence': conf, 'bbox': xyxy})
                        cv2.rectangle(image, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
                        cv2.putText(image, '%s %.2f' % (name, conf), (xyxy[0], max(20, xyxy[1] - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                args.output.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.output / 'cpu_yolo_check.jpg'), image)
                print('PASS local YOLO weight, CPU seconds/frame after warmup:',
                      round(report['cpu_inference_seconds_after_warmup'], 3))
            except Exception as error:
                report['errors'].append('fusion smoke check: ' + str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / ('check_' + args.mode + '.json')
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    for error in report['errors']:
        print('FAIL:', error)
    print('Report:', target)
    print('RESULT:', 'FAIL' if report['errors'] else 'PASS')
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
