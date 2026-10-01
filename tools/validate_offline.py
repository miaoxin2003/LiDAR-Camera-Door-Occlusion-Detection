#!/usr/bin/env python3
"""Meaningful geometry/binary-layout checks runnable on Windows without ROS."""
import json
import struct
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'image_pointcloud_fusion/scripts'))
from c32_common import xyz_from_cloud, project_xyz, calibrated_geometry


class GeometryTests(unittest.TestCase):
    def test_organized_cloud_with_padding_and_byte_order(self):
        for big_endian in (False, True):
            data = bytearray(160)
            coords = [(1., 2., 3.), (0., 0., 0.), (float('nan'), 1., 1.), (4., 5., 6.)]
            for index, point in enumerate(coords):
                offset = (index // 2) * 80 + (index % 2) * 32
                struct.pack_into(('>' if big_endian else '<') + 'fff', data, offset, *point)
                # intensity is at offset 16, not 12, and never used as XYZ.
                struct.pack_into(('>' if big_endian else '<') + 'f', data, offset + 16, 999.)
            fields = [SimpleNamespace(name=n, offset=o, datatype=7, count=1)
                      for n, o in [('x', 0), ('y', 4), ('z', 8), ('intensity', 16)]]
            cloud = SimpleNamespace(data=data, height=2, width=2, point_step=32,
                                    row_step=80, fields=fields, is_bigendian=big_endian)
            np.testing.assert_array_equal(xyz_from_cloud(cloud), [[1, 2, 3], [4, 5, 6]])

    def test_empty_cloud(self):
        cloud = SimpleNamespace(width=0, height=1)
        self.assertEqual(xyz_from_cloud(cloud).shape, (0, 3))

    def test_known_pinhole_and_points_behind_camera(self):
        points = np.array([[0, 0, 2], [.2, .1, 2], [0, 0, -1], [100, 0, 1]], dtype=float)
        intrinsic = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]], dtype=float)
        uv, depth, indices = project_xyz(points, np.eye(4), intrinsic, np.zeros(5), 100, 80)
        np.testing.assert_allclose(uv, [[50, 40], [60, 45]])
        np.testing.assert_allclose(depth, [2, 2])
        np.testing.assert_array_equal(indices, [0, 1])

    def test_distortion_is_applied(self):
        intrinsic = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]], dtype=float)
        uv, _, _ = project_xyz(np.array([[.5, .5, 1.]]), np.eye(4), intrinsic,
                              np.array([.1, 0, 0, 0, 0]), 200, 200)
        np.testing.assert_allclose(uv, [[102.5, 92.5]])

    def test_recorded_config_and_inverse_direction(self):
        config = YAML(typ='safe').load(ROOT / 'image_pointcloud_fusion/config/c32_oak.yaml')
        matrix, k, d = calibrated_geometry(config['extrinsic_matrix'], config['camera_matrix'],
                                           config['dist_coeffs'], invert=True)
        lines = (ROOT / 'calibration_2026_9_13_3_27_59.txt').read_text().splitlines()
        index = lines.index('T_camera_lidar:')
        inverse_from_file = np.array([[float(v) for v in line.split()] for line in lines[index + 1:index + 5]])
        np.testing.assert_allclose(matrix, inverse_from_file, atol=1e-6)
        report = json.loads((ROOT / 'inspection/bag_report.json').read_text(encoding='utf-8'))
        info = report['topics']['/camera_info']['camera_info_variants'][0]['configuration']
        np.testing.assert_array_equal(k.ravel(), info['K'])
        np.testing.assert_array_equal(d, info['D'])
        self.assertEqual(config['image_width'], info['width'])
        self.assertEqual(config['image_height'], info['height'])

    def test_rectified_uses_p_and_zero_distortion(self):
        config = YAML(typ='safe').load(ROOT / 'image_pointcloud_fusion/config/c32_oak.yaml')
        _, k, d = calibrated_geometry(config['extrinsic_matrix'], config['camera_matrix'],
            config['dist_coeffs'], invert=True, rectified=True,
            projection=config['projection_matrix'], rectification=config['rectification_matrix'])
        np.testing.assert_array_equal(k, np.array(config['projection_matrix']).reshape(3, 4)[:, :3])
        np.testing.assert_array_equal(d, np.zeros(5))

    def test_invalid_transform_is_rejected(self):
        with self.assertRaises(ValueError):
            calibrated_geometry(np.zeros((4, 4)), np.eye(3), np.zeros(5))

    def test_extreme_rays_cannot_fold_back_through_distortion(self):
        config = YAML(typ='safe').load(ROOT / 'image_pointcloud_fusion/config/c32_oak.yaml')
        k = np.array(config['camera_matrix']).reshape(3, 3)
        d = np.array(config['dist_coeffs'])
        # The negative k2 maps this ~54 degree ray back inside the pixel array
        # if the distortion polynomial is applied outside its calibrated domain.
        points = np.array([[0., 0., 2.], [1.4, 0., 1.]])
        _, _, indices = project_xyz(points, np.eye(4), k, d, 1280, 720)
        np.testing.assert_array_equal(indices, [0])

    def test_recorded_cloud_projects_without_invalid_pixels(self):
        config = YAML(typ='safe').load(ROOT / 'image_pointcloud_fusion/config/c32_oak.yaml')
        points = np.load(ROOT / 'inspection/sample_00_points.npz')['xyz']
        matrix, k, d = calibrated_geometry(config['extrinsic_matrix'], config['camera_matrix'],
                                           config['dist_coeffs'], invert=True)
        uv, depth, indices = project_xyz(points, matrix, k, d, 1280, 720)
        self.assertGreater(len(indices), 1000)
        self.assertTrue(np.isfinite(uv).all())
        self.assertTrue(np.all(depth > .1))
        self.assertTrue(np.all((uv[:, 0] >= 0) & (uv[:, 0] < 1280)))
        self.assertTrue(np.all((uv[:, 1] >= 0) & (uv[:, 1] < 720)))

    def test_launch_xml_and_rviz_yaml_parse(self):
        package = ROOT / 'image_pointcloud_fusion'
        ET.parse(package / 'package.xml')
        for path in (package / 'launch').glob('*.launch'):
            self.assertEqual(ET.parse(path).getroot().tag, 'launch')
        rviz = YAML(typ='safe').load(package / 'rviz/c32_oak.rviz')
        self.assertEqual(rviz['Visualization Manager']['Global Options']['Fixed Frame'], 'velodyne')


if __name__ == '__main__':
    unittest.main(verbosity=2)
