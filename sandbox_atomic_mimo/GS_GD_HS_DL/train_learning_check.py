"""Standalone direct-DL diagnostic. No GS/EM/HS estimate enters inference.

Run after export_learning_check.m. Requires numpy, scipy, torch.
python train_learning_check.py --data learning_check_data --pilots 10
This tiny noiseless dataset checks implementation; it is not a benchmark.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn

def read_examples(path):
    d = loadmat(path)
    # MATLAB arrays use (cell,user/slot,sample); explicit sample-first order.
    def flatten(name):
        return np.moveaxis(d[name],-1,0).reshape(d[name].shape[-1],-1)
    y,s,b,g = (flatten(n) for n in ['Ymag','pilots','references','channels'])
    # Center magnitudes by the known reference magnitude. This is a feature,
    # not a replacement for the exact nonlinear observation model.
    reference_rms = np.maximum(np.sqrt(np.mean(abs(b)**2,axis=1,keepdims=True)),1e-12)
    x = np.concatenate([y-abs(b),s.real,s.imag,
                        (b/reference_rms).real,(b/reference_rms).imag,
                        np.log(reference_rms)],axis=1)
    target = np.concatenate([g.real,g.imag],axis=1)
    return x.astype('float32'), target.astype('float32')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path(__file__).with_name('learning_check_data'))
    parser.add_argument('--pilots',type=int,choices=[10,30],default=10)
    parser.add_argument('--steps',type=int,default=3000)
    args=parser.parse_args()
    torch.manual_seed(20261018)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pairs=[read_examples(args.data/f'{name}_P{args.pilots}.mat') for name in ['train','validation']]
    mean=pairs[0][0].mean(0); std=np.maximum(pairs[0][0].std(0),1e-5)
    scale=max(float(np.sqrt(np.mean(pairs[0][1]**2))),1e-8)
    tensors=[(torch.tensor((x-mean)/std,device=device),torch.tensor(y/scale,device=device)) for x,y in pairs]
    model=nn.Sequential(nn.Linear(len(mean),256),nn.ReLU(),nn.Linear(256,256),
                        nn.ReLU(),nn.Linear(256,48)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    x,y=tensors[0]; history=[]
    for step in range(args.steps):
        model.train(); optimizer.zero_grad()
        loss=(model(x)-y).square().mean();loss.backward();optimizer.step()
        if step%100==0 or step==args.steps-1:
            model.eval()
            with torch.no_grad():
                metrics=[]
                for xx,yy in tensors:
                    pred=model(xx)
                    metrics.append(float((pred-yy).square().sum()/yy.square().sum()))
            history.append(dict(step=step,train_nmse=metrics[0],validation_nmse=metrics[1]))
            print(step, 'train/validation NMSE dB:',[round(10*np.log10(max(v,1e-15)),3) for v in metrics],flush=True)
    # Do not consume the test split while developing the model.
    report=dict(pilots=args.pilots,device=str(device),history=history,
                training_fit_pass=history[-1]['train_nmse']<1e-3,
                note='Training-fit threshold is a local diagnostic, not proof of generalization. Test set remains unused.')
    destination=args.data.parent/f'learning_check_P{args.pilots}'
    destination.mkdir(exist_ok=True)
    (destination/'report.json').write_text(json.dumps(report,indent=2))
    torch.save(dict(model=model.state_dict(),mean=mean,std=std,target_scale=scale),destination/'checkpoint.pt')

if __name__=='__main__':main()
