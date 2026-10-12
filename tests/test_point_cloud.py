import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from main import main, make_demo, read_cloud, render_cloud


class PointCloudTests(unittest.TestCase):
    def test_ascii_and_binary_filter_coordinates_and_align_colors(self):
        vertices = np.array([(1, 2, 3, 10, 20, 30), (np.nan, 0, 0, 40, 50, 60),
                             (4, 5, 6, 70, 80, 90)],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        with tempfile.TemporaryDirectory() as folder:
            for ascii_format in (True, False):
                with self.subTest(ascii=ascii_format):
                    path = Path(folder) / 'colored.ply'
                    PlyData([PlyElement.describe(vertices, 'vertex')], text=ascii_format).write(str(path))
                    points, colors, total, invalid = read_cloud(path)
                    self.assertEqual((total, invalid), (3, 1))
                    np.testing.assert_array_equal(points, [[1, 2, 3], [4, 5, 6]])
                    np.testing.assert_array_equal(colors, [[10, 20, 30], [70, 80, 90]])

    def test_demo_and_preview_sampling_leave_source_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '示例.ply'
            make_demo(path)
            before = path.read_bytes()
            points, colors, total, invalid = read_cloud(path)
            self.assertEqual((total, invalid), (12000, 0))
            output = Path(folder) / 'preview.html'
            self.assertEqual(render_cloud(points, colors, output, '示例', 100), 100)
            self.assertIn('Plotly.newPlot', output.read_text(encoding='utf-8'))
            self.assertEqual(path.read_bytes(), before)

    def test_missing_coordinates_and_empty_cloud(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'invalid.ply'
            vertices = np.zeros(1, dtype=[('x', 'f4')])
            PlyData([PlyElement.describe(vertices, 'vertex')], text=True).write(str(path))
            with self.assertRaisesRegex(ValueError, '坐标'):
                read_cloud(path)
            vertices = np.zeros(0, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            PlyData([PlyElement.describe(vertices, 'vertex')], text=True).write(str(path))
            self.assertEqual(main([str(path)]), 1)


if __name__ == '__main__':
    unittest.main()
