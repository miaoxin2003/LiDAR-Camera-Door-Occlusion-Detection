#!/usr/bin/env python3
"""Checks for measured door-top fitting, including a slanted top and outliers."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'image_pointcloud_fusion' / 'scripts'))
from c32_common import (fit_door_top_line, DoorLineSmoother,
                        estimate_door_occlusion)


class DoorLineTests(unittest.TestCase):
    def test_slanted_measured_line_with_outliers(self):
        rng = np.random.default_rng(3)
        lateral = np.linspace(-0.65, 0.65, 70)
        line = np.column_stack((4.4 + 0.04 * lateral, lateral,
                                0.85 + 0.008 * lateral))
        line += rng.normal(0, 0.0004, line.shape)
        outliers = np.column_stack((rng.uniform(3.5, 5, 30),
                                    rng.uniform(-0.65, 0.65, 30),
                                    rng.uniform(0.5, 1.2, 30)))
        points = np.vstack((line, outliers))
        pixels = np.vstack((np.column_stack((620 - lateral * 245,
                                             np.full(len(line), 135.0))),
                            np.column_stack((rng.uniform(455, 785, 30),
                                             rng.uniform(127, 146, 30)))))
        result = fit_door_top_line(points, pixels, (450, 125, 790, 150))
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result['point_count'], 60)
        self.assertGreater(result['length'], 1.2)
        self.assertLess(result['length'], 1.4)
        self.assertAlmostEqual(result['start'][2], 0.85, delta=0.02)
        self.assertAlmostEqual(result['end'][2], 0.85, delta=0.02)

    def test_sparse_points_do_not_make_a_door(self):
        points = np.array([[4.4, y, 0.85] for y in np.linspace(-0.2, 0.2, 8)])
        pixels = np.array([[x, 135] for x in np.linspace(500, 700, 8)])
        self.assertIsNone(fit_door_top_line(points, pixels, (450, 125, 790, 150)))

    def test_depth_jump_excludes_short_foreground_segment(self):
        lateral = np.linspace(-0.65, 0.65, 80)
        points = np.column_stack((4.4 + 0.02 * lateral, lateral,
                                  np.full(len(lateral), 0.85)))
        pixels = np.column_stack((620 - lateral * 245,
                                  np.full(len(lateral), 135.0)))
        depths = 4.4 + 0.02 * lateral
        # A short displaced run creates a sharp depth discontinuity.
        points[:10, 0] += 0.15
        depths[:10] += 0.15
        result = fit_door_top_line(points, pixels, (450, 125, 790, 150),
                                   depths=depths)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result['point_count'], 65)
        self.assertLess(result['length'], 1.3)

    def test_smoother_resets_on_gap_or_time_rewind(self):
        smoother = DoorLineSmoother(alpha=0.25)
        first = dict(start=np.array([4.0, 0.6, 0.8]),
                     end=np.array([4.0, -0.6, 0.8]))
        second = dict(start=np.array([4.1, 0.6, 0.8]),
                      end=np.array([4.1, -0.6, 0.8]))
        smoother.update(first, 10.0)
        smoothed = smoother.update(second, 11.0)
        self.assertAlmostEqual(smoothed['start'][0], 4.025)
        self.assertAlmostEqual(smoother.update(second, 15.0)['start'][0], 4.1)
        self.assertAlmostEqual(smoother.update(first, 1.0)['start'][0], 4.0)

    def test_door_occlusion_keeps_unknown_separate(self):
        left = np.array([4.0, 1.0, 1.0])
        right = np.array([4.0, -1.0, 1.0])
        blocked = np.array([2.0, 0.5, 0.0])
        clear = np.array([5.0, -0.5, 0.0])
        result = estimate_door_occlusion([blocked, clear], left, right, -1.0,
                                         cell_size=0.5, edge_margin=0)
        self.assertEqual(result['blocked'], 1)
        self.assertEqual(result['clear'], 1)
        self.assertEqual(result['unknown'], 14)
        self.assertAlmostEqual(result['occlusion_ratio'], 0.5)
        self.assertAlmostEqual(result['coverage'], 0.125)

    def test_door_occlusion_same_cell_blocked_wins(self):
        left = np.array([4.0, 1.0, 1.0])
        right = np.array([4.0, -1.0, 1.0])
        blocked = np.array([2.0, 0.5, 0.0])
        clear_same_ray = blocked * 2.5
        result = estimate_door_occlusion([blocked, clear_same_ray],
                                         left, right, -1.0,
                                         cell_size=0.5, edge_margin=0)
        self.assertEqual(result['blocked'], 1)
        self.assertEqual(result['clear'], 0)

    def test_door_occlusion_empty_returns_unknown(self):
        left = np.array([4.0, 1.0, 1.0])
        right = np.array([4.0, -1.0, 1.0])
        result = estimate_door_occlusion(np.empty((0, 3)),
                                         left, right, -1.0)
        self.assertEqual(result['blocked'], 0)
        self.assertEqual(result['clear'], 0)
        self.assertTrue(np.isnan(result['occlusion_ratio']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
