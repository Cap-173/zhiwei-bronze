import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from damage import save_sample, sphere_mask
from main import read_cloud


class DamageTests(unittest.TestCase):
    def test_boundary_and_validation(self):
        points = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
        np.testing.assert_array_equal(sphere_mask(points, [0, 0, 0], 1), [True, True, False])
        for radius in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                sphere_mask(points, [0, 0, 0], radius)
        with self.assertRaises(ValueError):
            sphere_mask(points, [float('nan'), 0, 0], 1)

    def test_roundtrip_and_preservation(self):
        points = np.array([[0.1234567890123, 0, 0], [1., 0, 0], [2., 0, 0]])
        colors = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
        mask = sphere_mask(points, [0, 0, 0], 0.5)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / 'sample'
            save_sample(directory, points, colors, mask, [0, 0, 0], 0.5, 'source.ply')
            target, target_rgb, _, _ = read_cloud(directory / 'target.ply')
            retained, retained_rgb, _, _ = read_cloud(directory / 'input.ply')
            removed, _, _, _ = read_cloud(directory / 'removed.ply')
            np.testing.assert_array_equal(target, points)
            np.testing.assert_array_equal(target_rgb, colors)
            np.testing.assert_array_equal(retained, points[~mask])
            np.testing.assert_array_equal(retained_rgb, colors[~mask])
            np.testing.assert_array_equal(removed, points[mask])
            np.testing.assert_array_equal(np.load(directory / 'mask.npy'), mask)
            metadata = json.loads((directory / 'region.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['remaining_points'] + metadata['removed_points'], len(points))
            with self.assertRaises(FileExistsError):
                save_sample(directory, points, colors, mask, [0, 0, 0], 0.5, 'source.ply')

    def test_empty_or_full_selection_leaves_no_sample(self):
        points = np.array([[0., 0, 0], [1., 0, 0]])
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / 'sample'
            for mask in (np.zeros(2, dtype=bool), np.ones(2, dtype=bool)):
                with self.assertRaises(ValueError):
                    save_sample(directory, points, None, mask, [0, 0, 0], 1, 'source.ply')
                self.assertFalse(directory.exists())
