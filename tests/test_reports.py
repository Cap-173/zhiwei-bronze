import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from damage import write_cloud
from inspect_dataset import inspect_dataset, render_report
from prepare_dataset import build_dataset
from report_run import render_history


class ReportTests(unittest.TestCase):
    def test_valid_data_and_corrupt_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.ply'
            write_cloud(source, np.random.default_rng(4).normal(size=(300, 3)))
            dataset = root / 'dataset'
            manifest = build_dataset([('a', source)], dataset, samples=2, point_limit=200)
            before = (dataset / 'manifest.json').read_bytes()
            report = inspect_dataset(dataset)
            self.assertTrue(report['passed'])
            self.assertEqual(report['checked_samples'], 2)
            self.assertEqual(report['splits']['test']['samples'], 0)
            render_report(report, root / 'report.html')
            self.assertEqual((dataset / 'manifest.json').read_bytes(), before)
            sample = dataset / manifest['samples'][0]['path']
            mask = np.load(sample / 'mask.npy')
            np.save(sample / 'mask.npy', ~mask)
            report = inspect_dataset(dataset)
            self.assertFalse(report['passed'])
            self.assertEqual(len(report['errors']), 1)
            self.assertIn('选区不一致', report['errors'][0])

    def test_curve_reports_old_and_new_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for validation in (False, True):
                history = dict(evaluation_scope='held-out models' if validation else 'training samples only',
                               history=[dict(epoch=1, train_loss=0.2, train_cd_squared=0.1,
                                             val_cd_squared=0.15 if validation else None),
                                        dict(epoch=2, train_loss=0.1, train_cd_squared=0.05,
                                             val_cd_squared=0.12 if validation else None,
                                             train_fscore=0.5)])
                path = root / 'history.json'
                path.write_text(json.dumps(history))
                result = render_history(path, root / f'{validation}.html')
                self.assertEqual(result, history)
                self.assertIn('Plotly.newPlot', (root / f'{validation}.html').read_text(encoding='utf-8'))
            path.write_text(json.dumps(dict(history=[])))
            with self.assertRaises(ValueError):
                render_history(path, root / 'empty.html')
