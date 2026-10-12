from pathlib import Path
import tempfile
import unittest

import torch

from compare_runs import compare


class AblationTests(unittest.TestCase):
    def test_mismatched_training_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = dict(seed=42, input_points=512, output_points=128, target_points=256,
                          batch_size=8, lr=0.001, amp=False)
            runs = {}
            for label, epoch in [('a', 20), ('b', 10)]:
                path = root/label
                path.mkdir()
                torch.save(dict(epoch=epoch, dataset_fingerprint='same', config=config), path/'last.pt')
                runs[label] = path
            with self.assertRaisesRegex(ValueError, '训练预算'):
                compare(root/'dataset', runs, root/'output')
            self.assertFalse((root/'output').exists())

    def test_labels_cannot_escape_output(self):
        with self.assertRaisesRegex(ValueError, '标签'):
            compare(Path('unused'), {'../escape': Path('a'), 'b': Path('b')}, Path('output'))
