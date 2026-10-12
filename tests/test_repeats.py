import unittest

from repeat_experiments import summarize


class RepeatTests(unittest.TestCase):
    def test_paired_statistics_include_negative_result(self):
        rows = [dict(seed=42, output_mode='soft', val_cd_squared=0.01, val_fscore=0.2),
                dict(seed=42, output_mode='tanh', val_cd_squared=0.008, val_fscore=0.3),
                dict(seed=43, output_mode='soft', val_cd_squared=0.02, val_fscore=0.1),
                dict(seed=43, output_mode='tanh', val_cd_squared=0.022, val_fscore=0.09)]
        result = summarize(rows)
        self.assertEqual(result['improved_seeds'], 1)
        self.assertAlmostEqual(result['pairs'][0]['relative_reduction_percent'], 20.)
        self.assertAlmostEqual(result['pairs'][1]['relative_reduction_percent'], -10.)
        self.assertAlmostEqual(result['aggregate']['soft']['mean_cd_squared'], 0.015)
        self.assertGreater(result['aggregate']['soft']['sample_std_cd_squared'], 0.)

    def test_unpaired_or_single_seed_is_rejected(self):
        row = dict(seed=42, output_mode='soft', val_cd_squared=0.01, val_fscore=0.2)
        with self.assertRaises(ValueError):
            summarize([row])
        with self.assertRaisesRegex(ValueError, '一对'):
            summarize([row, {**row, 'seed': 43}])
