import unittest
import torch
from density_loss import density_chamfer


class DensityTests(unittest.TestCase):
    def test_identity_and_permutation(self):
        points = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        self.assertLess(float(density_chamfer(points, points[:, [2, 0, 1]])), 1e-7)

    def test_duplicate_density_penalty(self):
        target = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]])
        duplicate = target[:, [0, 0, 2, 3]]
        value = density_chamfer(duplicate, target, alpha=50, exponent=1)
        # pred->gt: .5,.5,1,1; gt->pred: .5,.5,1,1, d²=1 for the missing point.
        expected = (0.25 + (0.5 + 1 - .5*torch.exp(torch.tensor(-50.)))/4) / 2
        torch.testing.assert_close(value[0], expected)
        self.assertGreater(float(value), 0.)

    def test_gradient_and_symmetry(self):
        a = torch.tensor([[[0.1, 0., 0.], [0.9, 0., 0.]]], requires_grad=True)
        b = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
        value = density_chamfer(a, b)
        torch.testing.assert_close(value, density_chamfer(b, a))
        value.sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(float(a.grad.abs().sum()), 0.)

    def test_unequal_points_rejected(self):
        with self.assertRaisesRegex(ValueError, '等点数'):
            density_chamfer(torch.zeros(1, 2, 3), torch.zeros(1, 3, 3))
