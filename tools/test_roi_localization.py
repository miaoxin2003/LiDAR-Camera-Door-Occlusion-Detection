#!/usr/bin/env python3
"""Focused checks for YOLO ROI localization without ROS or OpenCV runtime."""
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.modules.setdefault('cv2', types.ModuleType('cv2'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'image_pointcloud_fusion' / 'scripts'))
from c32_common import localize_detection_points


class RoiLocalizationTests(unittest.TestCase):
    def test_surface_patch_localizes_without_any_cluster(self):
        points = np.array([[3.0, x, 2.2] for x in np.linspace(-0.2, 0.2, 12)])
        pixels = np.array([[x, 100.0] for x in np.linspace(40, 60, 12)])
        depths = np.full(12, 3.0)
        result = localize_detection_points(points, pixels, depths, (35, 95, 65, 105))
        self.assertIsNotNone(result)
        self.assertEqual(result['point_count'], 12)
        self.assertAlmostEqual(result['center'][2], 2.2)

    def test_competing_foreground_and_background_remains_unlocalized(self):
        points = np.array([[depth, x, 1.0] for depth in (2.0, 5.0)
                           for x in np.linspace(-0.1, 0.1, 10)])
        pixels = np.tile(np.array([[50.0, 50.0]]), (20, 1))
        depths = points[:, 0]
        self.assertIsNone(localize_detection_points(points, pixels, depths,
                                                   (40, 40, 60, 60)))

    def test_empty_or_sparse_roi_has_no_invented_depth(self):
        points = np.array([[2.0, 0.0, 2.5]] * 7)
        pixels = np.array([[50.0, 50.0]] * 7)
        depths = np.full(7, 2.0)
        self.assertIsNone(localize_detection_points(points, pixels, depths,
                                                   (40, 40, 60, 60)))
        self.assertIsNone(localize_detection_points(points, pixels, depths,
                                                   (70, 70, 90, 90), min_points=1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
