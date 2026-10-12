import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from import_modelnet import import_archive, sample_off
from prepare_dataset import build_dataset


def tetrahedron(shift=0):
    return f'OFF4 4 0\n{shift} 0 0\n{shift+1} 0 0\n{shift} 1 0\n{shift} 0 1\n3 0 2 1\n3 0 1 3\n3 0 3 2\n3 1 2 3\n'.encode()


class PublicDataTests(unittest.TestCase):
    def test_surface_sampling_reproducible(self):
        a, info = sample_off(tetrahedron(), 200, 42)
        b, _ = sample_off(tetrahedron(), 200, 42)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(info['faces'], 4)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue((a >= -1e-10).all())
        self.assertTrue((a.sum(axis=1) <= 1+1e-10).all())

    def test_off_import_preserves_official_test_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'ModelNet40.zip'
            with zipfile.ZipFile(archive, 'w') as package:
                for i in range(3):
                    package.writestr(f'ModelNet40/cup/train/cup_{i:04d}.off', tetrahedron(i))
                package.writestr('ModelNet40/cup/test/cup_0004.off', tetrahedron(4))
            output = root / 'imported'
            report = import_archive(archive, output, 3, 1, 200, categories=['cup'])
            self.assertEqual(len(report['models']), 4)
            splits = json.loads((output / 'splits.json').read_text())
            self.assertEqual(list(splits.values()).count('val'), 1)
            for row in report['models']:
                if row['official_split'] == 'test':
                    self.assertEqual(splits[row['model_id']], 'test')
            mapping = json.loads((output / 'models.json').read_text())
            manifest = build_dataset([(key, output / value) for key, value in mapping.items()],
                                     root / 'dataset', samples=1, point_limit=200, split_assignments=splits)
            self.assertEqual(manifest['splits'], splits)
            self.assertTrue(manifest['evaluation_ready'])

    def test_invalid_explicit_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, '覆盖全部'):
                build_dataset([('a', root / 'a.ply')], root / 'out', split_assignments={'other': 'train'})
