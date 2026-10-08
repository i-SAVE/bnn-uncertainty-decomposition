# -*- coding: utf-8 -*-
"""Рисунки статьи в редакционном формате: каждая часть (а, б) — отдельный файл.

Выход (каталог figures/article/):
  jpeg300/ — JPEG 300 dpi (в размере вёрстки);
  svg/     — векторный редактируемый формат (текст сохранён как текст);
  data/    — CSV с данными, по которым построена каждая часть рисунка.
Все значения читаются из results/; ничего не задаётся вручную.
"""
import csv, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import stats
from sklearn.metrics import roc_curve
from PIL import Image
import paper_data as P

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, os.pardir, 'results')
OUT = os.path.join(HERE, os.pardir, 'figures', 'article')
for sub in ('jpeg300', 'svg', 'data'):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.linewidth': 0.7, 'lines.linewidth': 1.2,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'svg.fonttype': 'none', 'savefig.bbox': 'standard',
    'figure.constrained_layout.use': True,
    'figure.constrained_layout.h_pad': 0.03, 'figure.constrained_layout.w_pad': 0.03})

CM = 1 / 2.54
PW, PH = 7.3 * CM, 5.7 * CM          # одна часть рисунка: 7,3 × 5,7 см (две в ширину полосы)
FW = 16.4 * CM                        # рисунок на всю ширину полосы
GREY = '0.45'
C = {'epistemic': '#1F4E79', 'total': '#C1651A', 'aleatoric': '#6E7B8B', 'msp': '#7A9A3B'}
LAB = {'epistemic': 'эпистемическая', 'total': 'полная энтропия',
       'aleatoric': 'алеаторная', 'msp': 'максимум softmax'}
MK = {'epistemic': 'o', 'total': 's', 'aleatoric': '^', 'msp': 'D'}
LS = {'epistemic': '-', 'total': (0, (5, 2)), 'aleatoric': (0, (1, 1.6)),
      'msp': (0, (4, 1.6, 1, 1.6))}
CB, CT = '#C1651A', '#1F4E79'         # исходный / лучше обученный режим

R = json.load(open(os.path.join(RES, 'replication.json')))
S5 = [np.load(f) for f in sorted(glob.glob(os.path.join(RES, 'scores_seed*.npz')))]
assert len(S5) == 5


class Comma(ScalarFormatter):
    """Десятичная запятая и знак минуса, как в тексте статьи."""
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace('.', ',').replace('-', '−')


def tidy(ax, xfmt=True, yfmt=True):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(alpha=.18, lw=.4); ax.set_axisbelow(True)
    if xfmt:
        ax.xaxis.set_major_formatter(Comma())
    if yfmt:
        ax.yaxis.set_major_formatter(Comma())


def fmt(x, n=2):
    return f'{x:.{n}f}'.replace('.', ',')


def save(fig, name):
    fig.savefig(os.path.join(OUT, 'svg', name + '.svg'))
    tmp = os.path.join(OUT, 'jpeg300', name + '_tmp.png')
    fig.savefig(tmp, dpi=300)
    Image.open(tmp).convert('RGB').save(os.path.join(OUT, 'jpeg300', name + '.jpg'),
                                        quality=95, dpi=(300, 300), subsampling=0)
    os.remove(tmp)
    # в SVG шрифт указывается списком: Liberation Serif метрически совпадает с Times New Roman
    p = os.path.join(OUT, 'svg', name + '.svg')
    s = open(p, encoding='utf-8').read()
    s = re.sub(r"font-family:\s*'?Liberation Serif'?", "font-family:'Times New Roman','Liberation Serif',serif", s)
    s = re.sub(r'font-family="Liberation Serif"', 'font-family="Times New Roman, Liberation Serif, serif"', s)
    open(p, 'w', encoding='utf-8').write(s)
    plt.close(fig)


def write_csv(name, header, rows):
    with open(os.path.join(OUT, 'data', name + '.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(header)
        for r in rows:
            w.writerow([f'{v:.6g}'.replace('.', ',') if isinstance(v, (float, np.floating)) else v
                        for v in r])


# ------------------------------------------------ рис. 1 а, б: распределения составляющих
for part, key, title in (('а', 'epi', 'Эпистемическая составляющая'),
                         ('б', 'alea', 'Алеаторная составляющая')):
    a = np.concatenate([s[f'id_{key}'] for s in S5])
    b = np.concatenate([s[f'ood_{key}'] for s in S5])
    hi = np.percentile(np.concatenate([a, b]), 99.5)
    bins = np.linspace(0, hi, 60)
    fig, ax = plt.subplots(figsize=(PW, PH))
    ha, _, _ = ax.hist(a, bins=bins, density=True, alpha=.62, color=C['epistemic'],
                       label=f'FashionMNIST, среднее {fmt(a.mean())}')
    hb, _, _ = ax.hist(b, bins=bins, density=True, alpha=.62, color=C['total'], hatch='///',
                       edgecolor=C['total'], linewidth=0.0, label=f'MNIST, среднее {fmt(b.mean())}')
    ax.set_xlabel(f'{title}, нат'); ax.set_ylabel('Плотность')
    ax.legend(frameon=False, loc='upper right', handlelength=1.4, borderaxespad=0.2)
    tidy(ax)
    save(fig, f'Рис_1{part}')
    write_csv(f'Рис_1{part}', ['левая граница интервала, нат', 'правая граница, нат',
                               'плотность FashionMNIST', 'плотность MNIST'],
              [(bins[i], bins[i + 1], ha[i], hb[i]) for i in range(len(ha))])

# ------------------------------------------------ рис. 2 а: ROC-кривые
fig, ax = plt.subplots(figsize=(PW, PH))
grid = np.linspace(0, 1, 201)
rows = []
curves_all = {}
for k, key in (('msp', 'msp'), ('total', 'total'), ('aleatoric', 'alea'), ('epistemic', 'epi')):
    sgn = -1 if key == 'msp' else 1
    cur = []
    for s in S5:
        y = np.r_[np.zeros(len(s[f'id_{key}'])), np.ones(len(s[f'ood_{key}']))]
        fpr, tpr, _ = roc_curve(y, sgn * np.r_[s[f'id_{key}'], s[f'ood_{key}']])
        cur.append(np.interp(grid, fpr, tpr))
    cur = np.array(cur); curves_all[k] = cur
    ax.plot(grid, cur.mean(0), color=C[k], ls=LS[k], label=LAB[k])
    ax.fill_between(grid, cur.min(0), cur.max(0), color=C[k], alpha=.13, lw=0)
ax.plot([0, 1], [0, 1], color=GREY, lw=.7, ls=(0, (3, 3)))
ax.set_xlabel('Доля ложных срабатываний'); ax.set_ylabel('Доля верных обнаружений')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
h_, l_ = ax.get_legend_handles_labels()
idx = [l_.index(LAB[k]) for k in ('epistemic', 'total', 'msp', 'aleatoric')]
leg = ax.legend([h_[i] for i in idx], [l_[i] for i in idx], loc='lower right', handlelength=2.2,
                borderaxespad=0.3, frameon=True, framealpha=0.92, facecolor='white', edgecolor='none')
tidy(ax)
save(fig, 'Рис_2а')
hdr = ['доля ложных срабатываний']
for k in ('epistemic', 'total', 'msp', 'aleatoric'):
    hdr += [f'{LAB[k]}: среднее', f'{LAB[k]}: минимум', f'{LAB[k]}: максимум']
write_csv('Рис_2а', hdr, [[g] + sum(([curves_all[k].mean(0)[i], curves_all[k].min(0)[i],
                                      curves_all[k].max(0)[i]]
                                     for k in ('epistemic', 'total', 'msp', 'aleatoric')), [])
                          for i, g in enumerate(grid)])

# ------------------------------------------------ рис. 2 б: AUROC по запускам
fig, ax = plt.subplots(figsize=(PW, PH))
order = ['epistemic', 'total', 'msp', 'aleatoric']
rows = []
for i, k in enumerate(order):
    v = np.array([r['auroc'][k] for r in R])
    m, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
    ax.scatter(v, [i] * len(v), s=17, facecolor='none', edgecolor=C[k], linewidth=.9,
               marker=MK[k], zorder=3)
    ax.plot([m - h, m + h], [i, i], color=C[k], lw=1.6, alpha=.55, zorder=2)
    ax.plot([m], [i], marker='|', color=C[k], ms=11, mew=1.8, zorder=4)
    rows.append([LAB[k]] + list(v) + [m, m - h, m + h])
ax.axvline(0.5, color=GREY, lw=.7, ls=(0, (3, 3)))
ax.set_yticks(range(len(order))); ax.set_yticklabels([LAB[k] for k in order])
ax.set_xlim(0.30, 1.0); ax.set_ylim(len(order) - 0.5, -0.5)
ax.set_xlabel('AUROC по пяти запускам')
tidy(ax, yfmt=False); ax.grid(alpha=.18, lw=.4, axis='x')
save(fig, 'Рис_2б')
write_csv('Рис_2б', ['мера'] + [f'запуск {r["seed"]}' for r in R] +
          ['среднее', '95 % ДИ: нижняя граница', '95 % ДИ: верхняя граница'], rows)


# ------------------------------------------------ рис. 3 а, б: профили по уровням сдвига
def profile(ax, shift, xlabel, xticks, logx=False):
    data = []
    for reg, col, mk, ls, lab in (('base', CB, 's', (0, (5, 2)), '5 эпох, σ = 1 (исходный)'),
                                  ('trained', CT, 'o', '-', '20 эпох, σ = 0,1')):
        lv = P.level_summary(shift, reg)
        x = [r['level'] for r in lv]
        per = np.array([r['d_seeds'] for r in lv])          # уровни × запуски
        for j in range(per.shape[1]):
            ax.plot(x, per[:, j], color=col, lw=0.6, alpha=0.35, ls=ls)
        ax.plot(x, per.mean(1), color=col, lw=1.8, ls=ls, marker=mk, ms=4,
                label=f'{lab}, n = {per.shape[1]}')
        for i, xv in enumerate(x):
            data.append([lab, xv] + list(per[i]) + [per[i].mean()])
    ax.axhline(0, color=GREY, lw=0.8)
    if logx:
        ax.set_xscale('log')
    ax.set_xticks(xticks); ax.set_xticklabels([str(t).replace('.', ',') for t in xticks])
    ax.minorticks_off()
    ax.set_xlabel(xlabel); ax.set_ylabel('Δ AUROC (эпист. − полная)')
    tidy(ax, xfmt=False)
    return data


axes = []
for part, shift, xl, xt, lg in (('а', 'rotation', 'Угол поворота, градусы', P.ANG, False),
                                ('б', 'noise', 'СКО гауссова шума', [0.1, 0.2, 0.4, 0.8], True)):
    fig, ax = plt.subplots(figsize=(PW, PH))
    d = profile(ax, shift, xl, xt, lg)
    axes.append((fig, ax, part, d, shift))
lo = min(a.get_ylim()[0] for _, a, _, _, _ in axes); hi = max(a.get_ylim()[1] for _, a, _, _, _ in axes)
for fig, ax, part, d, shift in axes:
    ax.set_ylim(lo, hi)
    if part == 'а':
        ax.legend(frameon=False, loc='lower left', handlelength=2.6)
    save(fig, f'Рис_3{part}')
    nrun = len(d[0]) - 3
    write_csv(f'Рис_3{part}', ['режим', 'угол, градусы' if shift == 'rotation' else 'СКО шума'] +
              [f'запуск {j}' for j in range(nrun)] + ['среднее'], d)

# ------------------------------------------------ рис. 4: режимы обучения
rt = P.regime_table()
fig, ax = plt.subplots(figsize=(FW, 6.4 * CM))
xs = np.arange(len(rt))
for off, key, col, mk, lab in ((-0.09, 'd_mnist', CT, 'o', 'межнаборный сдвиг (MNIST)'),
                               (0.09, 'd_rot', CB, 's', 'повороты, среднее по 8 углам')):
    for i, r in enumerate(rt):
        pts = r[key + '_seeds']
        ax.scatter(np.full(len(pts), i + off), pts, s=12, color=col, alpha=0.35, lw=0,
                   marker=mk, zorder=2)
    m = [r[key][0] for r in rt]; sd = [r[key][1] for r in rt]
    ax.errorbar(xs + off, m, yerr=sd, color=col, marker=mk, ms=5, lw=1.4, capsize=3,
                label=lab, zorder=3)
ax.axhline(0, color=GREY, lw=0.8)
ax.set_xticks(xs)


def sig(v):
    return '1' if v == 1.0 else str(v).replace('.', ',')


ax.set_xticklabels([f"{r['key'][0]} эпох, σ = {sig(r['key'][1])}\n"
                    f"точность {fmt(r['acc'][0] * 100, 1)} %, n = {r['n']}" for r in rt])
ax.set_ylabel('Δ AUROC (эпист. − полная)')
ax.set_xlim(-0.5, len(rt) - 0.5)
ax.legend(frameon=False, loc='upper right')
tidy(ax, xfmt=False)
save(fig, 'Рис_4')
rows = []
for r in rt:
    for j, sd_ in enumerate(r['seeds']):
        rows.append([f"{r['key'][0]} эпох, σ = {sig(r['key'][1])}", sd_, r['acc_seeds'][j],
                     r['d_mnist_seeds'][j], r['d_rot_seeds'][j]])
write_csv('Рис_4', ['режим', 'запуск', 'точность (первые 4000 тестовых объектов)',
                    'Δ AUROC, MNIST', 'Δ AUROC, повороты (среднее по 8 углам)'], rows)
print('рисунки:', sorted(os.listdir(os.path.join(OUT, 'jpeg300'))))
