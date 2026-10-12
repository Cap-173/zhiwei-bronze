import unittest

import torch

from analyze_validation import diagnose


class DiagnosticTests(unittest.TestCase):
    def test_perfect_patch_covers_boundary(self):
        target = torch.tensor([[0., 0, 0], [0.9, 0, 0], [0, 0.9, 0]])
        metrics = diagnose(target, target, [0, 0, 0], 1.)
        self.assertEqual(metrics['local_cd_squared'], 0.)
        self.assertEqual(metrics['boundary_coverage_at_0_1_radius'], 1.)
        self.assertEqual(metrics['boundary_mean_distance_over_radius'], 0.)

    def test_central_collapse_and_no_boundary(self):
        target = torch.tensor([[0.9, 0., 0], [0, 0.9, 0]])
        metrics = diagnose(torch.zeros((4, 3)), target, [0, 0, 0], 1.)
        self.assertEqual(metrics['boundary_coverage_at_0_1_radius'], 0.)
        self.assertEqual(metrics['predicted_outer_fraction'], 0.)
        small = diagnose(torch.zeros((4, 3)), target*0.1, [0, 0, 0], 1.)
        self.assertIsNone(small['boundary_coverage_at_0_1_radius'])
