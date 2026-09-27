# -*- coding: utf-8 -*-
"""Независимый аудит: дополнительные реальные прогоны.

Очередь задач (выполняется последовательно, результаты пишутся после каждой):
  base  — конфигурация исходной статьи/ноутбука (5 эпох, prior N(0,1)):
          (a) проверка побитовой воспроизводимости сохранённых results/replication.json
          (b) MC-вариабельность AUROC при фиксированной обученной модели:
              независимые выборки весов при S=24 и S=50
          (c) повороты 15–180° (подвыборка 4000, как в shift.py) + проверка
              воспроизводимости results/shift.json + парный бутстреп разностей
  trained — лучше обученная модель (20 эпох, prior N(0, 0.1^2)):
          повороты + MNIST (скрещенный дизайн «тип сдвига × режим обучения»)
          + проверка воспроизводимости results/regime.json
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (TRANSFORM, ROOT, tensors, train_bnn, probs_samples,
                                decompose, auroc)
from regime import train as train_regime
from shift import rotate_batch, ANGLES
from torchvision.datasets import FashionMNIST, MNIST

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, os.pardir, 'results'))
OUT = os.path.join(RES, 'audit')
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 2)
S = 50
N_BOOT = 1000


def measures(u_id, u_ood):
    return {'epistemic': (u_id['epi'], u_ood['epi']),
            'total': (u_id['total'], u_ood['total']),
            'aleatoric': (u_id['alea'], u_ood['alea']),
            'msp': (-u_id['msp'], -u_ood['msp'])}


def aurocs(u_id, u_ood):
    return {k: auroc(a, b) for k, (a, b) in measures(u_id, u_ood).items()}


def paired_boot(u_id, u_ood, k1, k2, seed=0):
    m = measures(u_id, u_ood)
    a1, b1 = m[k1]; a2, b2 = m[k2]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        ia = rng.integers(0, len(a1), len(a1)); ib = rng.integers(0, len(b1), len(b1))
        d.append(auroc(a1[ia], b1[ib]) - auroc(a2[ia], b2[ib]))
    d = np.array(d)
    return {'point': auroc(a1, b1) - auroc(a2, b2),
            'lo': float(np.percentile(d, 2.5)), 'hi': float(np.percentile(d, 97.5))}


def rotations(model, guide, Xid4, yid4, Xm4, seed):
    Pid = probs_samples(model, guide, Xid4, S, seed=seed)
    uid = decompose(Pid)
    rows = []
    for ang in ANGLES:
        Pr = probs_samples(model, guide, rotate_batch(Xid4, ang), S, seed=seed)
        ur = decompose(Pr)
        row = {'angle': ang, 'accuracy': float((ur['mean'].argmax(-1) == yid4).mean())}
        if ang > 0:
            row['auroc'] = aurocs(uid, ur)
            row['d_epi_total'] = paired_boot(uid, ur, 'epistemic', 'total', seed)
            row['d_epi_msp'] = paired_boot(uid, ur, 'epistemic', 'msp', seed)
        rows.append(row)
        print(f'    rot {ang:3d}: acc={row["accuracy"]:.3f} '
              + (f'epi={row["auroc"]["epistemic"]:.4f} tot={row["auroc"]["total"]:.4f}'
                 if ang else ''), flush=True)
    Pm = probs_samples(model, guide, Xm4, S, seed=seed)
    um = decompose(Pm)
    mn = {'auroc': aurocs(uid, um),
          'd_epi_total': paired_boot(uid, um, 'epistemic', 'total', seed),
          'd_epi_msp': paired_boot(uid, um, 'epistemic', 'msp', seed)}
    return rows, mn


def save(name, obj):
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main(queue):
    ftr = FashionMNIST(ROOT, train=True, transform=TRANSFORM, download=True)
    fte = FashionMNIST(ROOT, train=False, transform=TRANSFORM, download=True)
    mte = MNIST(ROOT, train=False, transform=TRANSFORM, download=True)
    Xtr, ytr = tensors(ftr); Xid, yid = tensors(fte); Xm, _ = tensors(mte)
    yid_np = yid.numpy()
    REP = json.load(open(os.path.join(RES, 'replication.json')))
    SH = json.load(open(os.path.join(RES, 'shift.json')))
    REG = json.load(open(os.path.join(RES, 'regime.json')))

    for kind, seed in queue:
        t0 = time.time()
        print(f'===== {kind} seed {seed} =====', flush=True)
        rec = {'kind': kind, 'seed': seed, 'S': S}
        if kind == 'base':
            model, guide, _ = train_bnn(Xtr, ytr, seed)
            rec['config'] = {'epochs': 5, 'sigma': 1.0}
            # (a) воспроизводимость replication.json
            Pid = probs_samples(model, guide, Xid, S, seed=seed)
            Pood = probs_samples(model, guide, Xm, S, seed=seed)
            uid, uood = decompose(Pid), decompose(Pood)
            new = aurocs(uid, uood)
            s24 = aurocs(decompose(Pid[:24]), decompose(Pood[:24]))
            ref = next((r for r in REP if r['seed'] == seed), None)
            rec['repro_A'] = {'new': new, 'new_S24_nested': s24,
                              'stored': ref['auroc'] if ref else None,
                              'stored_S24': ref['S_sensitivity']['24'] if ref else None,
                              'max_abs_diff': (max(abs(new[k] - ref['auroc'][k]) for k in new)
                                               if ref else None)}
            print('  repro A max|diff| =', rec['repro_A']['max_abs_diff'], flush=True)
            # (b) MC-вариабельность при фиксированной модели (полные наборы 10000/10000)
            if seed in (0, 1):
                mc = {'S24': [], 'S50': [new]}
                for k in range(4):
                    a = probs_samples(model, guide, Xid, 24, seed=20000 + 100 * seed + k)
                    b = probs_samples(model, guide, Xm, 24, seed=20000 + 100 * seed + k)
                    mc['S24'].append(aurocs(decompose(a), decompose(b)))
                for k in range(2):
                    a = probs_samples(model, guide, Xid, 50, seed=30000 + 100 * seed + k)
                    b = probs_samples(model, guide, Xm, 50, seed=30000 + 100 * seed + k)
                    mc['S50'].append(aurocs(decompose(a), decompose(b)))
                rec['mc_variability'] = mc
                print('  MC S24 epi:', [round(x['epistemic'], 4) for x in mc['S24']],
                      ' S50 epi:', [round(x['epistemic'], 4) for x in mc['S50']], flush=True)
            del Pid, Pood
            # (c) повороты
            rows, mn = rotations(model, guide, Xid[:4000], yid_np[:4000], Xm[:4000], seed)
            rec['rotation'] = rows; rec['mnist_subset4000'] = mn
            ref = next((r for r in SH if r['seed'] == seed), None)
            if ref:
                diffs = []
                for r_new, r_old in zip(rows, ref['angles']):
                    if r_new['angle'] == 0:
                        continue
                    for k in ('epistemic', 'total', 'aleatoric', 'msp'):
                        diffs.append(abs(r_new['auroc'][k] - r_old[f'auroc_{k}']))
                rec['repro_B_max_abs_diff'] = max(diffs)
                print('  repro B max|diff| =', rec['repro_B_max_abs_diff'], flush=True)
        else:  # trained
            model, guide = train_regime(Xtr, ytr, seed, 0.1, 20)
            rec['config'] = {'epochs': 20, 'sigma': 0.1}
            Pid = probs_samples(model, guide, Xid[:5000], S, seed=seed)
            Pood = probs_samples(model, guide, Xm[:5000], S, seed=seed)
            new = aurocs(decompose(Pid), decompose(Pood))
            ref = next((g for g in REG if g['seed'] == seed and g['epochs'] == 20
                        and g['sigma'] == 0.1), None)
            rec['repro_C'] = {'new': new, 'stored': ref['auroc'] if ref else None,
                              'max_abs_diff': (max(abs(new[k] - ref['auroc'][k]) for k in new)
                                               if ref else None)}
            rec['accuracy_full'] = None
            print('  repro C max|diff| =', rec['repro_C']['max_abs_diff'], flush=True)
            rows, mn = rotations(model, guide, Xid[:4000], yid_np[:4000], Xm[:4000], seed)
            rec['rotation'] = rows; rec['mnist_subset4000'] = mn
        rec['seconds'] = round(time.time() - t0)
        save(f'{kind}_seed{seed}.json', rec)
        print(f'  saved ({rec["seconds"]} s)', flush=True)


if __name__ == '__main__':
    q = [('base', 0), ('trained', 0), ('base', 1), ('trained', 1),
         ('base', 2), ('base', 3), ('base', 4)]
    main(q)
    print('ALL DONE', flush=True)
