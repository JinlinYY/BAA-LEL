import numpy as np
import unittest

from bua_lel.utils.metrics_seg import binary_surface_distance_metrics
from baselines.metrics import segmentation_case_metrics


class SurfaceDistanceTests(unittest.TestCase):
    def test_surface_distances_are_zero_for_identical_masks(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True

        hd95, assd = binary_surface_distance_metrics(mask, mask)

        self.assertAlmostEqual(hd95, 0.0)
        self.assertAlmostEqual(assd, 0.0)


    def test_surface_distances_for_one_pixel_translation(self):
        target = np.zeros((8, 8), dtype=bool)
        pred = np.zeros((8, 8), dtype=bool)
        target[3, 3] = True
        pred[3, 4] = True

        hd95, assd = binary_surface_distance_metrics(pred, target)

        self.assertAlmostEqual(hd95, 1.0)
        self.assertAlmostEqual(assd, 1.0)


    def test_one_empty_mask_uses_image_diagonal_penalty(self):
        pred = np.zeros((3, 4), dtype=bool)
        target = np.zeros((3, 4), dtype=bool)
        target[1, 1] = True

        hd95, assd = binary_surface_distance_metrics(pred, target)

        self.assertAlmostEqual(hd95, 5.0)
        self.assertAlmostEqual(assd, 5.0)

    def test_assd_matches_case_level_benchmark_definition(self):
        pred = np.zeros((16, 16), dtype=bool)
        target = np.zeros((16, 16), dtype=bool)
        pred[2:14, 2:14] = True
        target[5:10, 6:11] = True

        hd95, assd = binary_surface_distance_metrics(pred, target)
        expected = segmentation_case_metrics(pred, target)

        self.assertAlmostEqual(hd95, expected["HD95"])
        self.assertAlmostEqual(assd, expected["ASSD"])
