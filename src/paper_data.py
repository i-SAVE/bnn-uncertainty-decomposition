# -*- coding: utf-8 -*-
"""Все числа статьи — из сохранённых результатов (ничего не вписывается вручную).

A — results/replication.json (5 сидов, полные наборы 10 000 / 10 000)
B — сдвиги пяти видов × два режима обучения × 5 сидов:
      повороты      — results/audit/{base_seed*, trained_seed*, regime_e20_s01_seed*}.json
      MNIST, KMNIST, гауссов шум — results/final/EF_*.json
      отложенные классы — results/final/D_*.json
C — четыре режима обучения, MNIST и повороты — results/audit/*.json
"""
import glob, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, os.pardir, 'results'))
K = ('epistemic', 'total', 'msp', 'aleatoric')
ANG = [15, 30, 45, 60, 90, 120, 150, 180]

R = json.load(open(os.path.join(RES, 'replication.json')))


def mci(v):
    v = np.asarray(v, float)
    m = v.mean()
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return m, sd, m - h, m + h


def cls(d):
    return '+' if d['lo'] > 0 else ('−' if d['hi'] < 0 else '0')


# ------------------------------------------------------------ результаты аудита
def _a2():
    runs = {}
    for f in glob.glob(os.path.join(RES, 'audit', '*.json')):
        r = json.load(open(f))
        key = (r['config']['epochs'], r['config']['sigma'])
        runs.setdefault(key, {})[r['seed']] = r
    return runs


A2 = _a2()
REGIMES = {'base': (5, 1.0), 'trained': (20, 0.1)}
REGIME_ORDER = [(5, 1.0), (5, 0.1), (20, 1.0), (20, 0.1)]


def _final(prefix):
    out = {'base': {}, 'trained': {}}
    for f in glob.glob(os.path.join(RES, 'final', f'{prefix}_*.json')):
        r = json.load(open(f))
        out[r['regime']][r['seed']] = r
    return out


D = _final('D')
EF = _final('EF')


def rot_rows(run):
    return {x['angle']: x for x in run['rotation']}


def shift_units(shift, regime):
    """Для каждого сида — список пар «внутр./внешн.» (уровней сдвига) с AUROC и Δ."""
    out = {}
    if shift == 'rotation':
        for s, run in A2.get(REGIMES[regime], {}).items():
            rr = rot_rows(run)
            out[s] = [dict(level=a, auroc=rr[a]['auroc'], d=rr[a]['d_epi_total'],
                           dm=rr[a]['d_epi_msp'], acc=rr[a]['accuracy']) for a in ANG]
    elif shift in ('mnist', 'kmnist'):
        for s, run in EF[regime].items():
            x = run[shift]
            out[s] = [dict(level=None, auroc=x['auroc'], d=x['d_epi_total'], dm=x['d_epi_msp'])]
    elif shift == 'noise':
        for s, run in EF[regime].items():
            out[s] = [dict(level=x['sigma_noise'], auroc=x['auroc'], d=x['d_epi_total'],
                           dm=x['d_epi_msp'], acc=x['accuracy']) for x in run['noise']]
    elif shift == 'classes':
        for s, run in D[regime].items():
            x = run['near_ood']
            out[s] = [dict(level=None, auroc=x['auroc'], d=x['d_epi_total'], dm=x['d_epi_msp'])]
    return out


def shift_summary(shift, regime):
    u = shift_units(shift, regime)
    if not u:
        return None
    seeds = sorted(u)
    per_seed_d = [np.mean([x['d']['point'] for x in u[s]]) for s in seeds]
    per_seed = {k: [np.mean([x['auroc'][k] for x in u[s]]) for s in seeds] for k in K}
    cells = [cls(x['d']) for s in seeds for x in u[s]]
    cells_m = [cls(x['dm']) for s in seeds for x in u[s]]
    best = [max(K, key=lambda k: x['auroc'][k]) for s in seeds for x in u[s]]
    return dict(n=len(seeds), seeds=seeds, levels=len(u[seeds[0]]),
                auroc={k: mci(per_seed[k]) for k in K}, d=mci(per_seed_d), d_seeds=per_seed_d,
                pos=cells.count('+'), neg=cells.count('−'), zero=cells.count('0'),
                cells=len(cells), pos_m=cells_m.count('+'), neg_m=cells_m.count('−'),
                epi_best=best.count('epistemic'))


def level_summary(shift, regime):
    """По каждому уровню (угол, σ шума): Δ по сидам."""
    u = shift_units(shift, regime)
    seeds = sorted(u)
    if not seeds:
        return []
    levels = [x['level'] for x in u[seeds[0]]]
    out = []
    for i, lv in enumerate(levels):
        ds = [u[s][i]['d']['point'] for s in seeds]
        cs = [cls(u[s][i]['d']) for s in seeds]
        out.append(dict(level=lv, d=mci(ds), d_seeds=ds, pos=cs.count('+'), neg=cs.count('−'),
                        acc=mci([u[s][i].get('acc', np.nan) for s in seeds])))
    return out


def regime_table():
    """C: для каждого режима — точность, Δ на MNIST (4000) и средний Δ на поворотах."""
    out = []
    for key in REGIME_ORDER:
        runs = A2.get(key, {})
        if not runs:
            continue
        seeds = sorted(runs)
        acc = [rot_rows(runs[s])[0]['accuracy'] for s in seeds]
        dm = [runs[s]['mnist_subset4000']['d_epi_total']['point'] for s in seeds]
        dr = [np.mean([rot_rows(runs[s])[a]['d_epi_total']['point'] for a in ANG]) for s in seeds]
        am = {k: [runs[s]['mnist_subset4000']['auroc'][k] for s in seeds] for k in K}
        cells = [cls(rot_rows(runs[s])[a]['d_epi_total']) for s in seeds for a in ANG]
        out.append(dict(key=key, n=len(seeds), seeds=seeds, acc=mci(acc), acc_seeds=acc,
                        d_mnist=mci(dm), d_mnist_seeds=dm, d_rot=mci(dr), d_rot_seeds=dr,
                        auroc_mnist={k: mci(am[k]) for k in K},
                        rot_pos=cells.count('+'), rot_neg=cells.count('−'),
                        rot_cells=len(cells)))
    return out


def repro_checks():
    """Все проверки побитовой воспроизводимости сохранённых результатов."""
    vals = []
    for key, runs in A2.items():
        for r in runs.values():
            if r.get('repro_A'):
                vals.append(r['repro_A']['max_abs_diff'])
            if r.get('repro_B_max_abs_diff') is not None:
                vals.append(r['repro_B_max_abs_diff'])
            if r.get('repro_C', {}).get('max_abs_diff') is not None:
                vals.append(r['repro_C']['max_abs_diff'])
    return len(vals), max(vals) if vals else None


def mnist_consistency():
    """Контроль: Δ на MNIST, посчитанный в F-прогонах, совпадает с аудитом (те же модели)."""
    diffs = []
    for reg, key in REGIMES.items():
        for s, run in EF[reg].items():
            a2 = A2.get(key, {}).get(s)
            if a2:
                for k in K:
                    diffs.append(abs(run['mnist']['auroc'][k] - a2['mnist_subset4000']['auroc'][k]))
    return len(diffs), max(diffs) if diffs else None


def mc_range():
    rng = {'S24': [], 'S50': []}
    for r in A2.get((5, 1.0), {}).values():
        mc = r.get('mc_variability')
        if mc:
            for S in rng:
                v = [x['epistemic'] for x in mc[S]]
                rng[S].append(max(v) - min(v))
    return {S: max(v) for S, v in rng.items() if v}
