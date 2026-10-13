"""Frozen exact-matching ablation on the same fixed hole and16-hole dataset."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import scipy
from assignment_loss import assignment_per_sample
from completion import make_model, context_points, sample_points, chamfer_per_sample
from training_data import checked_sample, read_manifest, dataset_fingerprint
from train import train
from main import ROOT


def main():
    torch.set_num_threads(2)
    basic=ROOT/'outputs/diagnostics/basic_fit'
    previous=json.loads((basic/'protocol.json').read_text(encoding='utf-8'))
    output=ROOT/'outputs/diagnostics/assignment_fit'
    output.mkdir(parents=True,exist_ok=False)
    dataset=Path(previous['single_object_dataset']);source=Path(previous['source_dataset'])
    manifest=read_manifest(dataset)
    protocol=dict(previous,architecture='pointwise_xyz',loss_mode='emd_exact',
        dataset_fingerprint=dataset_fingerprint(dataset,manifest),
        single_hole_steps=1000,single_object_epochs=200,batch_size=4,seed=42,
        reference_paper='https://arxiv.org/pdf/1808.00671 section4.3',
        reference_code='https://github.com/wentaoyuan/pcn/blob/master/tf_util.py',
        adaptation='Exact SciPy Jonker-Volgenant instead of PCN approximate CUDA matching; mean Euclidean/R;128 local missing points instead of1024 coarse full-object points; no fine decoder or fine CD term',
        scipy_version=scipy.__version__,torch_version=str(torch.__version__),
        baseline_checkpoint_sha256=hashlib.sha256((ROOT/'outputs/runs/bronze_single_object_pointwise_xyz_seed42/last.pt').read_bytes()).hexdigest(),
        baseline_fixed_result_sha256=hashlib.sha256((basic/'single_hole_pointwise_xyz.npz').read_bytes()).hexdigest(),
        selection='fixed1000 steps and finalepoch200; no unseen-hole early stopping or tuning')
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    sample=checked_sample(source/previous['single_hole']['path']/'training.npz')
    context=context_points(sample['input'],sample['center'],512,np.random.default_rng(43))
    target=sample_points(sample['local_target'],128,np.random.default_rng(42))
    with np.load(basic/'single_hole_pointwise_xyz.npz') as data:
        np.testing.assert_array_equal(context,data['input']);np.testing.assert_array_equal(target,data['target'])
    torch.manual_seed(42)
    model=make_model(128,'tanh','pointwise_xyz',32,1140,32,16)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    points=torch.tensor(context,dtype=torch.float32)[None]
    targets=torch.tensor(target,dtype=torch.float32)[None]
    region=torch.tensor(np.r_[sample['center'],sample['radius']],dtype=torch.float32)[None]
    history=[]
    for step in range(1001):
        prediction=model(points,region)
        loss=assignment_per_sample(prediction,targets,region).mean()
        if step%100==0:
            cd=(chamfer_per_sample(prediction,targets)/region[:,3].square()).mean()
            history.append(dict(step=step,loss_euclidean_over_R=float(loss.detach()),cd_over_R2=float(cd.detach())))
            print(f'Exact matching step{step}/1000 distance/R={float(loss.detach()):.7f} CD2/R2={float(cd.detach()):.8f}',flush=True)
        if step==1000:break
        if not torch.isfinite(loss):raise ValueError('匹配损失非有限。')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    np.savez_compressed(output/'single_hole_emd_exact.npz',input=context,target=target,prediction=prediction.detach().numpy()[0])
    (output/'single_hole_emd_exact.json').write_text(json.dumps(dict(history=history),indent=2),encoding='utf-8')
    torch.save(dict(state_dict=model.state_dict(),output_points=128,input_points=512,seed=42,output_mode='tanh',
        config=dict(architecture='pointwise_xyz',decoder_width=1140,encoder_anchors=32,encoder_neighbors=16),
        diagnostic='exact-matching fixed-hole1000 steps; not an independent generalization model'),output/'single_hole_emd_exact.pt')
    train(dataset,ROOT/'outputs/runs/bronze_single_object_emd_exact_seed42',epochs=200,input_points=512,
          output_points=128,target_points=128,batch_size=4,lr=.001,device='cpu',output_mode='tanh',
          architecture='pointwise_xyz',decoder_width=1140,encoder_anchors=32,encoder_neighbors=16,seed=42,
          loss_mode='emd_exact')


if __name__=='__main__':main()
