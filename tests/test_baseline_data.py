import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from baselines.data import BaselineDataset, CaseRecord


class BaselineDataTests(unittest.TestCase):
    def test_ultrasound_pipeline_matches_bua_grayscale_three_channel_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "case.png"
            bgr = np.zeros((8, 8, 3), dtype=np.uint8)
            bgr[..., 0], bgr[..., 1], bgr[..., 2] = 10, 100, 240
            cv2.imwrite(str(image_path), bgr)
            dataset = BaselineDataset([CaseRecord("p1", image_path, (), 0)], None, image_size=8, grayscale=True)
            image = dataset[0]["image"].numpy()
            self.assertTrue(np.array_equal(image[0], image[1]))
            self.assertTrue(np.array_equal(image[1], image[2]))


if __name__ == "__main__":
    unittest.main()
