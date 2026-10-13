import unittest
import tempfile
from pathlib import Path
import torch
from completion import GeometryCompletion, load_checkpoint
from neighborhood_encoder import local_groups


class NeighborhoodTests(unittest.TestCase):
    def test_groups_and_permutation(self):
        points=torch.tensor([[[3.,0.,0.],[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]])
        centers,offsets=local_groups(points,2,2)
        torch.testing.assert_close(centers,points[:,[1,0]])
        torch.testing.assert_close(offsets,torch.tensor([[[[0.,0.,0.],[1.,0.,0.]],[[0.,0.,0.],[-1.,0.,0.]]]]))
        permuted=local_groups(points.flip(1),2,2)
        for a,b in zip((centers,offsets),permuted):
            torch.testing.assert_close(a,b)

    def test_model_capacity_gradient_and_checkpoint(self):
        torch.manual_seed(123)
        local=GeometryCompletion(16,'tanh',256,'local',4,8)
        torch.manual_seed(123)
        control=GeometryCompletion(16,'tanh',256,'pointwise_xyz',4,8)
        self.assertEqual(sum(p.numel() for p in local.parameters()),sum(p.numel() for p in control.parameters()))
        for key,value in local.state_dict().items():
            torch.testing.assert_close(value,control.state_dict()[key])
        points=torch.randn(2,32,3)
        region=torch.tensor([[0.,0.,0.,.5],[2.,3.,4.,2.]])
        prediction=local(points,region)
        torch.testing.assert_close(prediction,local(points.flip(1),region))
        self.assertTrue((torch.linalg.vector_norm(prediction-region[:,None,:3],dim=-1)<=region[:,3:4]+1e-6).all())
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in local.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'local.pt'
            torch.save(dict(state_dict=local.state_dict(),output_points=16,output_mode='tanh',
                            config=dict(architecture='local',encoder_anchors=4,encoder_neighbors=8)),path)
            restored,_=load_checkpoint(path)
            torch.testing.assert_close(prediction,restored(points,region))

    def test_duplicate_and_invalid_groups(self):
        centers,offsets=local_groups(torch.zeros(2,8,3),4,4)
        self.assertTrue(torch.isfinite(offsets).all())
        self.assertEqual(centers.shape,(2,4,3))
        with self.assertRaises(ValueError):
            local_groups(torch.zeros(1,4,3),5,2)


if __name__=='__main__':
    unittest.main()
