# -*- coding: utf-8 -*-
"""Продолжение аудита: скрещенный дизайн «повороты × режим обучения».

Лучше обученный режим (20 эпох, σ = 0,1) доводится до пяти сидов (сиды 0–1 считаются в
audit_runs.py), промежуточные режимы (5, 0,1) и (20, 1,0) — по пять сидов (0–4). Для
сидов, которые есть в results/regime.json, проверяется побитовая воспроизводимость.
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_runs import rotations, aurocs, save, S
from core import TRANSFORM, ROOT, tensors, probs_samples, decompose
from regime import train as train_regime
from torchvision.datasets import FashionMNIST, MNIST

RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results'))


def main(queue):
    Xtr, ytr = tensors(FashionMNIST(ROOT, train=True, transform=TRANSFORM, download=True))
    Xid, yid = tensors(FashionMNIST(ROOT, train=False, transform=TRANSFORM, download=True))
    Xm, _ = tensors(MNIST(ROOT, train=False, transform=TRANSFORM, download=True))
    yid_np = yid.numpy()
    REG = json.load(open(os.path.join(RES, 'regime.json')))
    for epochs, sigma, seed in queue:
        t0 = time.time()
        print(f'===== regime ep={epochs} sigma={sigma} seed {seed} =====', flush=True)
        model, guide = train_regime(Xtr, ytr, seed, sigma, epochs)
        rec = {'kind': 'regime', 'seed': seed, 'S': S,
               'config': {'epochs': epochs, 'sigma': sigma}}
        Pid = probs_samples(model, guide, Xid[:5000], S, seed=seed)
        Pood = probs_samples(model, guide, Xm[:5000], S, seed=seed)
        new = aurocs(decompose(Pid), decompose(Pood))
        ref = next((g for g in REG if g['seed'] == seed and g['epochs'] == epochs
                    and g['sigma'] == sigma), None)
        rec['repro_C'] = {'new': new, 'stored': ref['auroc'] if ref else None,
                          'max_abs_diff': (max(abs(new[k] - ref['auroc'][k]) for k in new)
                                           if ref else None)}
        rec['accuracy_5000'] = float((decompose(Pid)['mean'].argmax(-1) == yid_np[:5000]).mean())
        print('  repro C max|diff| =', rec['repro_C']['max_abs_diff'],
              ' acc5000 =', round(rec['accuracy_5000'], 4), flush=True)
        rows, mn = rotations(model, guide, Xid[:4000], yid_np[:4000], Xm[:4000], seed)
        rec['rotation'] = rows; rec['mnist_subset4000'] = mn
        rec['seconds'] = round(time.time() - t0)
        save(f'regime_e{epochs}_s{str(sigma).replace(".", "")}_seed{seed}.json', rec)
        print(f'  saved ({rec["seconds"]} s)', flush=True)


if __name__ == '__main__':
    q = ([(20, 0.1, s) for s in (2, 3, 4)] +
         [(5, 0.1, s) for s in range(5)] + [(20, 1.0, s) for s in range(5)])
    main(q)
    print('ALL DONE', flush=True)
