from pathlib import Path
import tempfile
import unittest
import torch
from completion import LocalCompletion, HierarchicalCompletion, farthest_subset, load_checkpoint


class HierarchicalTests(unittest.TestCase):
    def test_wide_capacity_control_checkpoint(self):
        model=LocalCompletion(16, 'tanh', 512)
        points=torch.randn(1,32,3)
        region=torch.tensor([[0.,0.,0.,1.]])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'wide.pt'
            torch.save(dict(state_dict=model.state_dict(),output_points=16,output_mode='tanh',
                            config=dict(decoder_width=512)),path)
            restored,_=load_checkpoint(path)
            torch.testing.assert_close(model(points,region),restored(points,region))

    def test_geometry_gradient_and_checkpoint(self):
        model=HierarchicalCompletion(16, 'tanh', 4)
        points=torch.randn(2, 32, 3)
        region=torch.tensor([[0.,0.,0.,.5],[2.,3.,4.,2.]])
        coarse,fine=model.forward_stages(points,region)
        self.assertEqual(coarse.shape,(2,4,3))
        self.assertEqual(fine.shape,(2,16,3))
        torch.testing.assert_close(fine,model(points.flip(1),region))
        self.assertTrue((torch.linalg.vector_norm(fine-region[:,None,:3],dim=-1)<=region[:,3:4]+1e-6).all())
        (fine.square().mean()+coarse.square().mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt'
            torch.save(dict(state_dict=model.state_dict(),output_points=16,output_mode='tanh',
                            config=dict(architecture='hierarchical',coarse_points=4)),path)
            loaded,_=load_checkpoint(path)
            torch.testing.assert_close(loaded(points,region),fine)

    def test_fps_unique_and_endpoints(self):
        points=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]]])
        torch.testing.assert_close(farthest_subset(points,2),points[:,[0,3]])
        all_points=farthest_subset(points,4)[0]
        self.assertEqual(len(torch.unique(all_points,dim=0)),4)
        duplicates=torch.zeros(1,5,3)
        self.assertTrue(torch.isfinite(farthest_subset(duplicates,4)).all())

    def test_invalid_parent_count(self):
        with self.assertRaises(ValueError):
            HierarchicalCompletion(16,'tanh',5)
