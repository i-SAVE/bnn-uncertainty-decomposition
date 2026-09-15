"""Faithful replication of bnn_task_Kistin.ipynb (Task 1), plus everything the
paper claims but the notebook never computed.

Everything below mirrors the notebook exactly: the [-1, 1] normalisation, the
relu(maxpool(conv)) ordering, AutoDiagonalNormal, pyro.plate with
size=DATASET_SIZE, Adam lr=2e-3, batch 128, 5 epochs, S=50 posterior samples,
tau=0.7.  The only addition is an explicit seed, which the notebook lacks.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST, MNIST, KMNIST
from torchvision.transforms import v2
from sklearn.metrics import roc_auc_score, roc_curve

DEV = torch.device('cpu')
ROOT = os.path.join(os.path.dirname(__file__), 'data')
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'results'))
BATCH_SIZE, EPOCHS, LR = 128, 5, 2e-3
NUM_SAMPLES, CONF_THRESHOLD = 50, 0.7


def normalize(image):
    return (image * 2 - 1).to(torch.float32)


TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                        v2.Lambda(normalize)])


def tensors(ds):
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=2048, shuffle=False):
        xs.append(x); ys.append(y)
    return torch.cat(xs), torch.cat(ys)


class BayesianCNN(PyroModule):
    """Byte-for-byte the notebook's model."""
    def __init__(self, dataset_size, n_classes=10):
        super().__init__()
        self.dataset_size = dataset_size
        self.n_classes = n_classes
        def N(*shape):
            return PyroSample(dist.Normal(torch.zeros(*shape), torch.ones(*shape))
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
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        with pyro.plate("data", size=self.dataset_size, subsample_size=x.size(0)):
            pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits


class DetCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5); self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 256); self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


def train_bnn(Xtr, ytr, seed, n_classes=10, epochs=EPOCHS):
    pyro.clear_param_store(); pyro.set_rng_seed(seed); torch.manual_seed(seed)
    model = BayesianCNN(len(Xtr), n_classes)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, pyro.optim.Adam({"lr": LR}), loss=Trace_ELBO())
    g = torch.Generator().manual_seed(seed)
    hist = []
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - BATCH_SIZE + 1, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            hist.append(-svi.step(Xtr[idx], ytr[idx]) / BATCH_SIZE)
        print(f"  ep{ep+1} ELBO/batch-elem {np.mean(hist[-100:]):.1f}", flush=True)
    return model, guide, hist


@torch.no_grad()
def probs_samples(model, guide, X, S, batch=500, seed=0):
    """(S, N, K) softmax probabilities, exactly as the notebook's Predictive call."""
    pyro.set_rng_seed(10_000 + seed)
    pred = Predictive(model, guide=guide, num_samples=S, return_sites=("_RETURN",))
    out = []
    for i in range(0, len(X), batch):
        logits = pred(X[i:i + batch])["_RETURN"]
        out.append(F.softmax(logits, dim=-1).numpy().astype(np.float32))
    return np.concatenate(out, axis=1)


EPS = 1e-12
def H(p, axis=-1):
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log(p)).sum(axis)


def decompose(P):
    mean = P.mean(0)
    total = H(mean)
    alea = H(P).mean(0)
    return dict(mean=mean, total=total, alea=alea, epi=total - alea,
                msp=mean.max(-1))


def auroc(a, b):
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    return float(roc_auc_score(y, np.r_[a, b]))


def auroc_ci(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(a), len(a)); ib = rng.integers(0, len(b), len(b))
        vals.append(auroc(a[ia], b[ib]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_boot_diff(a1, b1, a2, b2, n_boot=2000, seed=0):
    """Bootstrap CI for AUROC(1) - AUROC(2) on the same objects (paired)."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(a1), len(a1)); ib = rng.integers(0, len(b1), len(b1))
        d.append(auroc(a1[ia], b1[ib]) - auroc(a2[ia], b2[ib]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fpr_at_tpr(a, b, target=0.95):
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    fpr, tpr, _ = roc_curve(y, np.r_[a, b])
    i = min(int(np.searchsorted(tpr, target, 'left')), len(fpr) - 1)
    return float(fpr[i])


def ece(probs, labels, bins=15):
    conf = probs.max(-1); pred = probs.argmax(-1)
    acc = (pred == labels).astype(float)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def threshold_stats(u, y, tau=CONF_THRESHOLD, labelled=True):
    """Reproduce the notebook's evaluate_with_threshold bookkeeping."""
    acc_mask = u['msp'] >= tau
    pred = u['mean'].argmax(-1)
    out = dict(total=int(len(pred)), refused=int((~acc_mask).sum()),
               accepted=int(acc_mask.sum()))
    if labelled:
        out['correct'] = int((pred[acc_mask] == y[acc_mask]).sum())
        out['incorrect'] = out['accepted'] - out['correct']
        out['acc_on_accepted'] = out['correct'] / max(out['accepted'], 1)
        out['overall_accuracy'] = float((pred == y).mean())
    out['rejection_rate'] = out['refused'] / out['total']
    return out


def run(seed, S=NUM_SAMPLES, outdir='results'):
    print(f"===== seed {seed} =====", flush=True)
    ftr = FashionMNIST(ROOT, train=True, transform=TRANSFORM, download=True)
    fte = FashionMNIST(ROOT, train=False, transform=TRANSFORM, download=True)
    mte = MNIST(ROOT, train=False, transform=TRANSFORM, download=True)
    Xtr, ytr = tensors(ftr); Xid, yid = tensors(fte); Xood, yood = tensors(mte)

    model, guide, hist = train_bnn(Xtr, ytr, seed)
    Pid = probs_samples(model, guide, Xid, S, seed=seed)
    Pood = probs_samples(model, guide, Xood, S, seed=seed)
    uid, uood = decompose(Pid), decompose(Pood)
    yid_np = yid.numpy()

    res = {'seed': seed, 'S': S}
    res['table1_fashion'] = threshold_stats(uid, yid_np)
    res['table1_mnist'] = threshold_stats(uood, None, labelled=False)
    res['mean_entropy'] = {'id': float(uid['total'].mean()), 'ood': float(uood['total'].mean())}
    res['mean_epistemic'] = {'id': float(uid['epi'].mean()), 'ood': float(uood['epi'].mean())}
    res['mean_aleatoric'] = {'id': float(uid['alea'].mean()), 'ood': float(uood['alea'].mean())}
    res['bnn_ece'] = ece(uid['mean'], yid_np)

    measures = {'msp': (-uid['msp'], -uood['msp']),
                'total': (uid['total'], uood['total']),
                'aleatoric': (uid['alea'], uood['alea']),
                'epistemic': (uid['epi'], uood['epi'])}
    res['auroc'] = {k: auroc(a, b) for k, (a, b) in measures.items()}
    res['auroc_ci95'] = {k: auroc_ci(a, b, seed=seed) for k, (a, b) in measures.items()}
    res['fpr95'] = {k: fpr_at_tpr(a, b) for k, (a, b) in measures.items()}
    res['delta_epi_vs_total'] = paired_boot_diff(*measures['epistemic'], *measures['total'], seed=seed)
    res['delta_epi_vs_msp'] = paired_boot_diff(*measures['epistemic'], *measures['msp'], seed=seed)

    # threshold sensitivity for the softmax-confidence rejection rule
    res['threshold_sweep'] = [
        {'tau': float(t),
         'rej_id': float((uid['msp'] < t).mean()),
         'rej_ood': float((uood['msp'] < t).mean()),
         'acc_accepted': float((uid['mean'].argmax(-1)[uid['msp'] >= t] ==
                                yid_np[uid['msp'] >= t]).mean()) if (uid['msp'] >= t).sum() else None}
        for t in np.arange(0.1, 1.0, 0.1)]

    # MC-sample sensitivity, reusing the same 50 draws
    res['S_sensitivity'] = {}
    for s in (1, 5, 10, 24, 50):
        ui, uo = decompose(Pid[:s]), decompose(Pood[:s])
        res['S_sensitivity'][s] = {
            'epistemic': auroc(ui['epi'], uo['epi']),
            'total': auroc(ui['total'], uo['total']),
            'aleatoric': auroc(ui['alea'], uo['alea']),
            'msp': auroc(-ui['msp'], -uo['msp']),
            'mean_epi_id': float(ui['epi'].mean()),
        }

    # deterministic reference
    torch.manual_seed(seed)
    det = DetCNN()
    opt = torch.optim.Adam(det.parameters(), lr=LR)
    lf = nn.CrossEntropyLoss(); g = torch.Generator().manual_seed(seed)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr) - BATCH_SIZE + 1, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad(); lf(det(Xtr[idx]), ytr[idx]).backward(); opt.step()
    det.eval()
    with torch.no_grad():
        dpi = torch.cat([F.softmax(det(Xid[i:i+1000]), -1) for i in range(0, len(Xid), 1000)]).numpy()
        dpo = torch.cat([F.softmax(det(Xood[i:i+1000]), -1) for i in range(0, len(Xood), 1000)]).numpy()
    res['deterministic'] = {
        'accuracy': float((dpi.argmax(-1) == yid_np).mean()),
        'ece': ece(dpi, yid_np),
        'auroc_entropy': auroc(H(dpi), H(dpo)),
        'auroc_msp': auroc(-dpi.max(-1), -dpo.max(-1)),
    }
    res['bnn_accuracy'] = float((uid['mean'].argmax(-1) == yid_np).mean())

    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, f'scores_seed{seed}.npz'),
                        **{f'id_{k}': v for k, v in
                           dict(total=uid['total'], alea=uid['alea'], epi=uid['epi'], msp=uid['msp']).items()},
                        **{f'ood_{k}': v for k, v in
                           dict(total=uood['total'], alea=uood['alea'], epi=uood['epi'], msp=uood['msp']).items()},
                        elbo=np.asarray(hist))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--out', default=RESULTS)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 2)
    allres = [run(s, outdir=a.out) for s in a.seeds]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'replication.json'), 'w') as f:
        json.dump(allres, f, indent=2, ensure_ascii=False)
    print(json.dumps(allres[-1]['auroc'], indent=2))
    print('saved', os.path.join(a.out, 'replication.json'))
