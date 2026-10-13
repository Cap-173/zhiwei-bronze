"""Frozen fixed-hole fit and single-object16-hole training for one architecture."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from completion import make_model, context_points, sample_points, chamfer_per_sample
from training_data import checked_sample
from train import train
from main import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture',required=True,choices=['local','pointwise_xyz'])
    args=parser.parse_args()
    output=ROOT/'outputs/diagnostics/basic_fit'
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    torch.set_num_threads(2);torch.manual_seed(42)
    source=Path(protocol['source_dataset'])
    sample=checked_sample(source/protocol['single_hole']['path']/'training.npz')
    context=context_points(sample['input'],sample['center'],512,np.random.default_rng(43))
    target=sample_points(sample['local_target'],128,np.random.default_rng(42))
    points=torch.tensor(context,dtype=torch.float32)[None]
    region=torch.tensor(np.r_[sample['center'],sample['radius']],dtype=torch.float32)[None]
    targets=torch.tensor(target,dtype=torch.float32)[None]
    model=make_model(128,'tanh',args.architecture,32,1140,32,16)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    records=[]
    for step in range(1001):
        prediction=model(points,region)
        loss=(chamfer_per_sample(prediction,targets)/region[:,3].square()).mean()
        if step%100==0:
            records.append(dict(step=step,loss=float(loss.detach())))
            print(f'{args.architecture} fixed-hole step{step}/1000 loss={float(loss.detach()):.8f}',flush=True)
        if step==1000: break
        if not torch.isfinite(loss): raise ValueError('拟合损失非有限。')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    torch.save(dict(state_dict=model.state_dict(),output_points=128,input_points=512,seed=42,output_mode='tanh',
        config=dict(architecture=args.architecture,decoder_width=1140,encoder_anchors=32,encoder_neighbors=16),
        diagnostic='fixed single hole and target; 1000 optimizer steps; not a generalization model'),output/f'single_hole_{args.architecture}.pt')
    np.savez_compressed(output/f'single_hole_{args.architecture}.npz',input=context,target=target,prediction=prediction.detach().numpy()[0])
    (output/f'single_hole_{args.architecture}.json').write_text(json.dumps(dict(history=records,initial=records[0]['loss'],final=records[-1]['loss']),indent=2),encoding='utf-8')
    train(Path(protocol['single_object_dataset']),ROOT/'outputs/runs'/f'bronze_single_object_{args.architecture}_seed42',
          epochs=200,input_points=512,output_points=128,target_points=128,batch_size=4,lr=.001,
          device='cpu',output_mode='tanh',decoder_width=1140,architecture=args.architecture,
          encoder_anchors=32,encoder_neighbors=16,seed=42)


if __name__=='__main__':
    main()
