import unittest
import torch
from repulsion_loss import repulsion_per_sample


class RepulsionTests(unittest.TestCase):
    def test_formula_and_scale(self):
        r = torch.tensor([[0., 0., 0., 1.]])
        p = torch.tensor([[[0., 0., 0.], [.02, 0., 0.]]], requires_grad=True)
        value = repulsion_per_sample(p, r, neighbors=1)
        expected = .07 - .02 * torch.exp(torch.tensor(-.02**2/.03**2))
        torch.testing.assert_close(value[0], expected)
        torch.testing.assert_close(value, repulsion_per_sample(p*3, r*3, neighbors=1))
        value.sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertLess(p.grad[0, 1, 0].item(), 0.)

    def test_duplicate_gradient_finite(self):
        p = torch.zeros(1, 6, 3, requires_grad=True)
        repulsion_per_sample(p, torch.tensor([[0., 0., 0., 1.]])).sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == '__main__':
    unittest.main()
