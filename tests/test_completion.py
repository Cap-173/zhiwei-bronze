from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from completion import LocalCompletion, chamfer_squared, load_checkpoint, boundary_coverage_loss
from predict import predict_cloud


class CompletionTests(unittest.TestCase):
    def test_boundary_loss_uses_only_outer_targets(self):
        prediction = torch.tensor([[[0., 0, 0]]], requires_grad=True)
        target = torch.tensor([[[0., 0, 0], [0.9, 0, 0]]])
        region = torch.tensor([[0., 0, 0, 1.]])
        loss = boundary_coverage_loss(prediction, target, region)
        self.assertAlmostEqual(float(loss.detach()), 0.81, places=6)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(boundary_coverage_loss(prediction.detach(), target*0.1, region)), 0.)

    def test_tanh_checkpoint_and_ball_constraint(self):
        model = LocalCompletion(16, 'tanh')
        points = torch.randn(2, 40, 3)
        region = torch.tensor([[0., 0, 0, 1.], [1., 2., 3., 0.5]])
        prediction = model(points, region)
        self.assertTrue((torch.linalg.vector_norm(prediction-region[:, None, :3], dim=-1) <= region[:, 3:4]+1e-6).all())
        prediction.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'tanh.pt'
            torch.save(dict(state_dict=model.state_dict(), output_points=16, output_mode='tanh'), path)
            restored, _ = load_checkpoint(path)
            torch.testing.assert_close(restored(points, region), prediction)

    def test_chamfer_known_distance_and_gradient(self):
        a = torch.tensor([[[0., 0, 0]]], requires_grad=True)
        b = torch.tensor([[[1., 0, 0]]])
        loss = chamfer_squared(a, b)
        self.assertAlmostEqual(float(loss.detach()), 2.)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertEqual(float(chamfer_squared(b, b)), 0.)

    def test_permutation_invariance_and_region_bound(self):
        torch.manual_seed(3)
        model = LocalCompletion(16)
        points = torch.randn(1, 40, 3)
        region = torch.tensor([[2., 3, 4, 0.5]])
        a = model(points, region)
        b = model(points[:, torch.randperm(40)], region)
        torch.testing.assert_close(a, b)
        self.assertEqual(tuple(a.shape), (1, 16, 3))
        self.assertTrue((torch.linalg.vector_norm(a-region[:, None, :3], dim=-1) <= 0.5).all())

    def test_checkpoint_and_prediction_preserve_input(self):
        torch.manual_seed(3)
        model = LocalCompletion(16)
        points = np.random.default_rng(9).normal(size=(100, 3))
        before = points.copy()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            checkpoint = dict(state_dict=model.state_dict(), output_points=16, input_points=32, seed=42)
            torch.save(checkpoint, path)
            loaded, config = load_checkpoint(path)
            prediction, _, _ = predict_cloud(loaded, config, points, [0, 0, 0], 0.5)
            self.assertTrue(np.isfinite(prediction).all())
            self.assertTrue((np.linalg.norm(prediction, axis=1) <= 0.500001).all())
            np.testing.assert_array_equal(points, before)
