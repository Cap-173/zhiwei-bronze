import itertools
import unittest
import torch
from assignment_loss import assignment_per_sample


class AssignmentTests(unittest.TestCase):
    def test_exact_cost_permutation_and_radius_units(self):
        p=torch.tensor([[[0.,0,0],[2.,1,0],[4.,0,0]]])
        t=torch.tensor([[[4.,1,0],[0.,1,0],[2.,0,0]]])
        region=torch.tensor([[0.,0,0,2.]])
        value=assignment_per_sample(p,t,region)
        expected=min(torch.linalg.vector_norm(p[0]-t[0,list(order)],dim=-1).mean().item()/2 for order in itertools.permutations(range(3)))
        self.assertAlmostEqual(value.item(),expected,places=6)
        torch.testing.assert_close(value,assignment_per_sample(p.flip(1),t.flip(1),region))
        torch.testing.assert_close(value,assignment_per_sample(p*3,t*3,region*3))

    def test_known_gradient_and_identical_points(self):
        p=torch.tensor([[[0.,0,0],[2.,0,0]]],requires_grad=True)
        t=torch.tensor([[[1.,0,0],[3.,0,0]]])
        region=torch.tensor([[0.,0,0,1.]])
        loss=assignment_per_sample(p,t,region).sum();loss.backward()
        torch.testing.assert_close(p.grad,torch.tensor([[[-.5,0,0],[-.5,0,0]]]))
        same=torch.zeros(1,4,3,requires_grad=True)
        zero=assignment_per_sample(same,same.detach().clone(),region).sum();zero.backward()
        self.assertEqual(zero.item(),0.)
        self.assertTrue(torch.isfinite(same.grad).all())

    def test_invalid_counts_and_values(self):
        with self.assertRaises(ValueError):
            assignment_per_sample(torch.zeros(1,4,3),torch.zeros(1,3,3),torch.tensor([[0.,0,0,1.]]))
        with self.assertRaises(ValueError):
            assignment_per_sample(torch.zeros(1,4,3),torch.zeros(1,4,3),torch.zeros(1,4))


if __name__=='__main__':unittest.main()
