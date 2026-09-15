"""Experiment C — training-regime sensitivity.

Does the epistemic advantage survive when the Bayesian network is no longer
badly underfit?  Grid: epochs x prior scale.  For each cell we record the
classification quality of the Bayesian model (accuracy, ECE) alongside the
OOD-detection quality of all four uncertainty measures, so the two can be
plotted against each other.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
from torchvision.datasets import FashionMNIST, MNIST

from core import (TRANSFORM, ROOT, tensors, probs_samples, decompose, auroc,
                  fpr_at_tpr, ece, H, BATCH_SIZE, LR, NUM_SAMPLES)

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'results'))


class ScaledBayesianCNN(PyroModule):
    """Same network as the notebook, with a configurable prior scale."""
    def __init__(self, dataset_size, sigma=1.0, n_classes=10):
        super().__init__()
        self.dataset_size = dataset_size
        self.n_classes = n_classes
        def N(*shape):
            return PyroSample(dist.Normal(torch.zeros(*shape),
                                          sigma * torch.ones(*shape))
                              .to_event(len(shape)))
        self.conv1 = PyroModule[nn.Conv2d](1, 16, kernel_size=5)
        self.conv1.weight = N(16, 1, 5, 5); self.conv1.bias = N(16)
        self.conv2 = PyroModule[nn.Conv2d](16, 32, kernel_size=5)
        self.conv2.weight = N(32, 16, 5, 5); self.conv2.bias = N(32)
        self.fc1 = PyroModule[nn.Linear](32 * 4 * 4, 256)
        self.fc1.weight = N(256, 32 * 4 * 4); self.fc1.bias = N(256)
        self.fc2 = PyroModule[nn.Linear](256, n_classes)
        self.fc2.weight = N(n_classes, 256); self.fc2.bias = N(n_classes)

    def forward(self, x, y=None):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        logits = self.fc2(F.relu(self.fc1(x)))
        with pyro.plate("data", size=self.dataset_size, subsample_size=x.size(0)):
            pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits


def train(Xtr, ytr, seed, sigma, epochs):
    pyro.clear_param_store(); pyro.set_rng_seed(seed); torch.manual_seed(seed)
    model = ScaledBayesianCNN(len(Xtr), sigma)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, pyro.optim.Adam({"lr": LR}), loss=Trace_ELBO())
    g = torch.Generator().manual_seed(seed)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, n - BATCH_SIZE + 1, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            tot += svi.step(Xtr[idx], ytr[idx]); nb += 1
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"    ep{ep+1}/{epochs} ELBO/N {-tot/nb/n:.3f}", flush=True)
    return model, guide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--subset', type=int, default=5000)
    ap.add_argument('--out', default=RESULTS)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 2)

    Xtr, ytr = tensors(FashionMNIST(ROOT, train=True, transform=TRANSFORM, download=True))
    Xid, yid = tensors(FashionMNIST(ROOT, train=False, transform=TRANSFORM, download=True))
    Xood, _ = tensors(MNIST(ROOT, train=False, transform=TRANSFORM, download=True))
    Xid, yid, Xood = Xid[:a.subset], yid[:a.subset], Xood[:a.subset]
    yid_np = yid.numpy()

    grid = [(5, 1.0), (5, 0.1), (20, 1.0), (20, 0.1)]
    out = []
    for seed in a.seeds:
        for epochs, sigma in grid:
            print(f"== seed {seed} epochs={epochs} sigma={sigma}", flush=True)
            model, guide = train(Xtr, ytr, seed, sigma, epochs)
            Pid = probs_samples(model, guide, Xid, NUM_SAMPLES, seed=seed)
            Pood = probs_samples(model, guide, Xood, NUM_SAMPLES, seed=seed)
            ui, uo = decompose(Pid), decompose(Pood)
            m = {'msp': (-ui['msp'], -uo['msp']),
                 'total': (ui['total'], uo['total']),
                 'aleatoric': (ui['alea'], uo['alea']),
                 'epistemic': (ui['epi'], uo['epi'])}
            rec = {
                'seed': seed, 'epochs': epochs, 'sigma': sigma,
                'bnn_accuracy': float((ui['mean'].argmax(-1) == yid_np).mean()),
                'bnn_ece': ece(ui['mean'], yid_np),
                'mean_epi_id': float(ui['epi'].mean()),
                'mean_epi_ood': float(uo['epi'].mean()),
                'mean_alea_id': float(ui['alea'].mean()),
                'mean_alea_ood': float(uo['alea'].mean()),
                'auroc': {k: auroc(x, y) for k, (x, y) in m.items()},
                'fpr95': {k: fpr_at_tpr(x, y) for k, (x, y) in m.items()},
            }
            out.append(rec)
            print(f"    acc={rec['bnn_accuracy']:.3f} ece={rec['bnn_ece']:.3f} "
                  f"epi={rec['auroc']['epistemic']:.3f} tot={rec['auroc']['total']:.3f} "
                  f"alea={rec['auroc']['aleatoric']:.3f}", flush=True)
            with open(os.path.join(a.out, 'regime.json'), 'w') as f:
                json.dump(out, f, indent=2)
    print('done')


if __name__ == '__main__':
    main()
