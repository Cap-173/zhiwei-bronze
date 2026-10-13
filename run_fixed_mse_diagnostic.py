"""Ordered-coordinate sanity check only: not a completion training objective."""
import json
import numpy as np
import torch
from completion import make_model, context_points, sample_points, chamfer_per_sample
from training_data import checked_sample
from main import ROOT


def main():
    torch.set_num_threads(2)
    output=ROOT/'outputs/diagnostics/basic_fit'
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    sample=checked_sample(ROOT/'outputs/datasets/bronze_20261013_seed42'/protocol['single_hole']['path']/'training.npz')
    points=torch.tensor(context_points(sample['input'],sample['center'],512,np.random.default_rng(43)),dtype=torch.float32)[None]
    target=torch.tensor(sample_points(sample['local_target'],128,np.random.default_rng(42)),dtype=torch.float32)[None]
    region=torch.tensor(np.r_[sample['center'],sample['radius']],dtype=torch.float32)[None]
    for architecture in protocol['architectures']:
        torch.manual_seed(42)
        model=make_model(128,'tanh',architecture,32,1140,32,16)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        history=[]
        for step in range(1001):
            prediction=model(points,region)
            loss=((prediction-target).square().sum(-1)/region[:,3,None].square()).mean()
            if step%100==0:
                cd=(chamfer_per_sample(prediction,target)/region[:,3].square()).mean()
                history.append(dict(step=step,ordered_mse_over_R2=float(loss.detach()),cd_over_R2=float(cd.detach())))
                print(f'{architecture} ordered sanity step{step} MSE/R2={float(loss.detach()):.9f} CD/R2={float(cd.detach()):.9f}',flush=True)
            if step==1000: break
            if not torch.isfinite(loss):raise ValueError('非有限诊断损失。')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        np.savez_compressed(output/f'single_hole_mse_{architecture}.npz',prediction=prediction.detach().numpy()[0],target=target[0].numpy())
        (output/f'single_hole_mse_{architecture}.json').write_text(json.dumps(dict(history=history,
            reason='Added after CD plateau; matched model initialization, context, target, Adam lr and1000 steps; fixed arbitrary target order. Diagnostic only; not usable for unordered completion targets.'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
