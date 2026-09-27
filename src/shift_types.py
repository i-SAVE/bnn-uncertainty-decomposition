# -*- coding: utf-8 -*-
"""Эксперименты D–F, закрывающие пункты аудита (дизайн зафиксирован до запуска).

Режимы обучения (оба — 5 сидов, 0–4):
  base    — 5 эпох, prior N(0, 1)        (исходная конфигурация)
  trained — 20 эпох, prior N(0, 0,1²)    (лучше обученная)

D. Семантически близкий сдвиг (ответ на замечание 3 рецензента).
   Сеть обучается на 8 классах FashionMNIST; классы «рубашка» (6) и «кроссовок» (7)
   исключены из обучения и служат внешним распределением. У каждого из них есть
   близкие классы во внутреннем распределении: рубашка — футболка, пуловер, пальто;
   кроссовок — сандалия, ботинок. Оценивание на полных тестовых подмножествах:
   8000 объектов внутреннего и 2000 внешнего распределения.

E. Второй далёкий межнаборный сдвиг: KMNIST (рукописные знаки кана), первые 4000
   объектов теста; внутреннее — первые 4000 объектов теста FashionMNIST (как в B).
   MNIST на тех же 4000 объектах — контроль.

F. Второй ковариатный сдвиг: аддитивный гауссов шум на тех же 4000 объектах,
   x' = clip(x + ε, −1, 1), ε ~ N(0, s²), s ∈ {0,1; 0,2; 0,4; 0,8} в единицах
   нормированного диапазона [−1, 1]. Шум фиксирован (генератор с постоянным сидом
   на каждый уровень), поэтому внешнее распределение одинаково для всех моделей.

Для каждой пары «внутр./внешн.»: AUROC четырёх мер при S = 50 и парный бутстреп
разностей Δ(эпист. − полная) и Δ(эпист. − max softmax), 1000 повторов.
"""
import json, os, sys, time
import numpy as np
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (TRANSFORM, ROOT, tensors, probs_samples, decompose,
                                BATCH_SIZE, LR)
from regime import ScaledBayesianCNN
from audit_runs import aurocs, paired_boot
from torchvision.datasets import FashionMNIST, MNIST, KMNIST

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'results', 'final'))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 2)
S = 50
REGIMES = {'base': (5, 1.0), 'trained': (20, 0.1)}
HELD_OUT = (6, 7)
NOISE = [0.1, 0.2, 0.4, 0.8]


def train(Xtr, ytr, seed, sigma, epochs, n_classes):
    """Как regime_experiment.train, но с числом классов."""
    pyro.clear_param_store(); pyro.set_rng_seed(seed); torch.manual_seed(seed)
    model = ScaledBayesianCNN(len(Xtr), sigma, n_classes=n_classes)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, pyro.optim.Adam({"lr": LR}), loss=Trace_ELBO())
    g = torch.Generator().manual_seed(seed)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - BATCH_SIZE + 1, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            svi.step(Xtr[idx], ytr[idx])
    return model, guide


def pair(uid, uood, seed):
    return {'auroc': aurocs(uid, uood),
            'd_epi_total': paired_boot(uid, uood, 'epistemic', 'total', seed),
            'd_epi_msp': paired_boot(uid, uood, 'epistemic', 'msp', seed)}


def save(name, obj):
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    Xtr, ytr = tensors(FashionMNIST(ROOT, train=True, transform=TRANSFORM, download=True))
    Xte, yte = tensors(FashionMNIST(ROOT, train=False, transform=TRANSFORM, download=True))
    Xm, _ = tensors(MNIST(ROOT, train=False, transform=TRANSFORM, download=True))
    Xk, _ = tensors(KMNIST(ROOT, train=False, transform=TRANSFORM, download=True))

    # --- D: 8 классов
    keep = [c for c in range(10) if c not in HELD_OUT]
    remap = {c: i for i, c in enumerate(keep)}
    mtr = torch.tensor([int(c) not in HELD_OUT for c in ytr])
    mte = torch.tensor([int(c) not in HELD_OUT for c in yte])
    Xtr8 = Xtr[mtr]; ytr8 = torch.tensor([remap[int(c)] for c in ytr[mtr]])
    Xid8 = Xte[mte]; yid8 = np.array([remap[int(c)] for c in yte[mte]])
    Xho = Xte[~mte]

    # --- E, F: подвыборка 4000, как в B
    Xid4, yid4 = Xte[:4000], yte[:4000].numpy()
    noisy = []
    for j, s in enumerate(NOISE):
        g = torch.Generator().manual_seed(777 + j)
        noisy.append(torch.clamp(Xid4 + s * torch.randn(Xid4.shape, generator=g), -1.0, 1.0))

    jobs = ([('D', r, s) for r in ('base', 'trained') for s in range(5)] +
            [('EF', r, s) for r in ('base', 'trained') for s in range(5)])
    for exp, reg, seed in jobs:
        fn = f'{exp}_{reg}_seed{seed}.json'
        if os.path.exists(os.path.join(OUT, fn)):
            print('skip', fn, flush=True); continue
        t0 = time.time()
        ep, sigma = REGIMES[reg]
        print(f'===== {exp} {reg} (ep={ep}, σ={sigma}) seed {seed}', flush=True)
        rec = {'exp': exp, 'regime': reg, 'epochs': ep, 'sigma': sigma, 'seed': seed, 'S': S}
        if exp == 'D':
            model, guide = train(Xtr8, ytr8, seed, sigma, ep, 8)
            uid = decompose(probs_samples(model, guide, Xid8, S, seed=seed))
            uho = decompose(probs_samples(model, guide, Xho, S, seed=seed))
            rec['held_out_classes'] = list(HELD_OUT)
            rec['n_id'], rec['n_ood'] = len(Xid8), len(Xho)
            rec['accuracy_id'] = float((uid['mean'].argmax(-1) == yid8).mean())
            rec['near_ood'] = pair(uid, uho, seed)
            a = rec['near_ood']['auroc']
            print(f"   acc={rec['accuracy_id']:.3f} epi={a['epistemic']:.4f} tot={a['total']:.4f} "
                  f"msp={a['msp']:.4f} alea={a['aleatoric']:.4f}", flush=True)
        else:
            model, guide = train(Xtr, ytr, seed, sigma, ep, 10)
            uid = decompose(probs_samples(model, guide, Xid4, S, seed=seed))
            rec['n_id'] = 4000
            rec['accuracy_id'] = float((uid['mean'].argmax(-1) == yid4).mean())
            rec['kmnist'] = pair(uid, decompose(probs_samples(model, guide, Xk[:4000], S, seed=seed)), seed)
            rec['mnist'] = pair(uid, decompose(probs_samples(model, guide, Xm[:4000], S, seed=seed)), seed)
            rec['noise'] = []
            for s, Xn in zip(NOISE, noisy):
                un = decompose(probs_samples(model, guide, Xn, S, seed=seed))
                row = pair(uid, un, seed)
                row['sigma_noise'] = s
                row['accuracy'] = float((un['mean'].argmax(-1) == yid4).mean())
                rec['noise'].append(row)
            print(f"   acc={rec['accuracy_id']:.3f} KMNIST epi={rec['kmnist']['auroc']['epistemic']:.4f} "
                  f"tot={rec['kmnist']['auroc']['total']:.4f} | noise Δ: "
                  + ' '.join(f"{r['d_epi_total']['point']:+.3f}" for r in rec['noise']), flush=True)
        rec['seconds'] = round(time.time() - t0)
        save(fn, rec)
        print(f'   saved ({rec["seconds"]} s)', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
