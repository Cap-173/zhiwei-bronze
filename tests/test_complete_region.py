from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from complete_region import complete
from completion import LocalCompletion
from damage import sphere_mask, write_cloud
from main import read_cloud


class CompletionEntryTests(unittest.TestCase):
    def test_region_inference_preserves_visible_coordinates_and_colors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            points = np.random.default_rng(4).uniform(-1, 1, (150, 3))
            colors = np.arange(450).reshape(150, 3).astype(np.uint8)
            write_cloud(root/'source.ply', points, colors)
            model = LocalCompletion(16, 'tanh')
            torch.save(dict(state_dict=model.state_dict(), output_points=16, output_mode='tanh', input_points=32, seed=42), root/'model.pt')
            result = complete(root/'source.ply', root/'model.pt', [0, 0, 0], .7, root/'out', True)
            actual, rgb, _, invalid = read_cloud(root/'out/completed.ply')
            keep = ~sphere_mask(points, [0, 0, 0], .7)
            np.testing.assert_array_equal(actual[:keep.sum()], points[keep])
            np.testing.assert_array_equal(rgb[:keep.sum()], colors[keep])
            self.assertEqual(len(actual), int(keep.sum())+16)
            self.assertEqual(invalid, 0)
            self.assertTrue(np.all(np.linalg.norm(actual[keep.sum():], axis=1) <= .7+1e-6))
            self.assertEqual(result['scope'], 'inference only; no ground truth used')
            with self.assertRaises(FileExistsError):
                complete(root/'source.ply', root/'model.pt', [0, 0, 0], .7, root/'out', True)
