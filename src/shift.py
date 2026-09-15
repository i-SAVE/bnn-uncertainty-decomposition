"""Graded distribution shift: rotate the FashionMNIST test set by 0..180 degrees
and track all four uncertainty measures.

This reuses the rotation idea already present in the notebook (task 1.2) but
applies it to the whole test set rather than one image, which turns an
illustration into a controlled near-to-far shift experiment: small angles are a
mild, same-domain shift; large angles are a strong one. It answers the
reviewer's request for "more similar domains" without introducing a new dataset.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST, MNIST
from torchvision.transforms import v2

from core import (TRANSFORM, ROOT, tensors, train_bnn, probs_samples,
                  decompose, auroc, fpr_at_tpr, NUM_SAMPLES)

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'results'))

ANGLES = [0, 15, 30, 45, 60, 90, 120, 150, 180]


def rotate_batch(X, angle):
    if angle == 0:
        return X
    return v2.functional.rotate(X, angle, fill=[-1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--out', default=RESULTS)
    ap.add_argument('--subset', type=int, default=4000)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 2)

    ftr = FashionMNIST(ROOT, train=True, transform=TRANSFORM, download=True)
    fte = FashionMNIST(ROOT, train=False, transform=TRANSFORM, download=True)
    mte = MNIST(ROOT, train=False, transform=TRANSFORM, download=True)
    Xtr, ytr = tensors(ftr)
    Xid, yid = tensors(fte)
    Xm, _ = tensors(mte)
    Xid, yid = Xid[:a.subset], yid[:a.subset]
    Xm = Xm[:a.subset]

    out = []
    for seed in a.seeds:
        print('== seed', seed, flush=True)
        model, guide, _ = train_bnn(Xtr, ytr, seed)
        Pid = probs_samples(model, guide, Xid, NUM_SAMPLES, seed=seed)
        uid = decompose(Pid)
        rec = {'seed': seed, 'angles': [], 'mnist': None}
        for ang in ANGLES:
            Xr = rotate_batch(Xid, ang)
            Pr = probs_samples(model, guide, Xr, NUM_SAMPLES, seed=seed)
            ur = decompose(Pr)
            row = {'angle': ang,
                   'accuracy': float((ur['mean'].argmax(-1) == yid.numpy()).mean()),
                   'mean_total': float(ur['total'].mean()),
                   'mean_alea': float(ur['alea'].mean()),
                   'mean_epi': float(ur['epi'].mean())}
            if ang > 0:
                for key, sid, so in (('epistemic', uid['epi'], ur['epi']),
                                     ('total', uid['total'], ur['total']),
                                     ('aleatoric', uid['alea'], ur['alea']),
                                     ('msp', -uid['msp'], -ur['msp'])):
                    row[f'auroc_{key}'] = auroc(sid, so)
            rec['angles'].append(row)
            print(f"   {ang:3d}deg acc={row['accuracy']:.3f} "
                  f"epi={row['mean_epi']:.3f} "
                  f"auroc_epi={row.get('auroc_epistemic', float('nan')):.3f}", flush=True)
        Pm = probs_samples(model, guide, Xm, NUM_SAMPLES, seed=seed)
        um = decompose(Pm)
        rec['mnist'] = {k: auroc(s_id, s_ood) for k, (s_id, s_ood) in
                        {'epistemic': (uid['epi'], um['epi']),
                         'total': (uid['total'], um['total']),
                         'aleatoric': (uid['alea'], um['alea']),
                         'msp': (-uid['msp'], -um['msp'])}.items()}
        print('   MNIST', rec['mnist'], flush=True)
        out.append(rec)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'shift.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('saved')


if __name__ == '__main__':
    main()
