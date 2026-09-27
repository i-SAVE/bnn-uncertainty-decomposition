# -*- coding: utf-8 -*-
"""Рисунки 1-4 статьи. Все значения читаются из results/*.json и
results/scores_seed*.npz; ничего не задаётся вручную.

Единый формат: все четыре рисунка построены на одну ширину полосы набора
(16,5 см), одним шрифтом и одной палитрой; серии различаются одновременно
цветом, типом линии и маркером, поэтому остаются различимыми в чёрно-белой
печати.
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import stats
import roc_utils as U

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, 'figures')
RES = os.path.join(HERE, os.pardir, 'results')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'DejaVu Serif'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.7, 'lines.linewidth': 1.2,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'savefig.dpi': 400, 'savefig.bbox': 'standard', 'figure.constrained_layout.use': True,
    'figure.constrained_layout.h_pad': 0.02, 'figure.constrained_layout.w_pad': 0.02,
})

W = 6.5                      # дюймы ≈ 16,5 см — ширина полосы набора
GREY = '0.45'
C = {'epistemic': '#1F4E79', 'total': '#C1651A',
     'aleatoric': '#6E7B8B', 'msp': '#7A9A3B'}
LAB = {'epistemic': 'эпистемическая', 'total': 'полная энтропия',
       'aleatoric': 'алеаторная', 'msp': 'максимум softmax'}
MK = {'epistemic': 'o', 'total': 's', 'aleatoric': '^', 'msp': 'D'}
LS = {'epistemic': '-', 'total': (0, (5, 2)), 'aleatoric': (0, (1, 1.6)),
      'msp': (0, (4, 1.6, 1, 1.6))}

R = json.load(open(os.path.join(RES, 'replication.json')))
S5 = [np.load(f) for f in sorted(glob.glob(os.path.join(RES, 'scores_seed*.npz')))]


def ci(v):
    v = np.asarray(v, float)
    sd = v.std(ddof=1)
    h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
    return v.mean(), sd, h


class CommaFormatter(ScalarFormatter):
    """Десятичная запятая на осях — как в тексте статьи."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace('.', ',')


def COMMA():
    return CommaFormatter()


def tidy(ax, xfmt=True, yfmt=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=.18, lw=.4)
    ax.set_axisbelow(True)
    if xfmt:                       # десятичная запятая — как в тексте статьи
        ax.xaxis.set_major_formatter(COMMA())
    if yfmt:
        ax.yaxis.set_major_formatter(COMMA())


def fmt(x, n=3):
    return f'{x:.{n}f}'.replace('.', ',')


# ------------------------------- рис. 1: составляющие, внутр. vs внешн.
fig, axes = plt.subplots(1, 2, figsize=(W, 2.35))
for ax, key, title in zip(axes, ['epi', 'alea'],
                          ['Эпистемическая составляющая',
                           'Алеаторная составляющая']):
    a = np.concatenate([s[f'id_{key}'] for s in S5])
    b = np.concatenate([s[f'ood_{key}'] for s in S5])
    hi = np.percentile(np.concatenate([a, b]), 99.5)
    bins = np.linspace(0, hi, 70)
    ax.hist(a, bins=bins, density=True, alpha=.62, color=C['epistemic'],
            label=f'FashionMNIST, среднее {fmt(a.mean(), 2)}')
    ax.hist(b, bins=bins, density=True, alpha=.62, color=C['total'],
            hatch='///', edgecolor=C['total'], linewidth=0.0,
            label=f'MNIST, среднее {fmt(b.mean(), 2)}')
    ax.set_xlabel(f'{title}, нат')
    ax.set_ylabel('Плотность')
    ax.legend(frameon=False, loc='upper right', handlelength=1.4,
              borderaxespad=0.2)
    tidy(ax)
fig.savefig(os.path.join(OUT, 'fig1_components.png'))
plt.close(fig)

# --------------------- рис. 2: ROC-кривые и AUROC по отдельным запускам
fig, axes = plt.subplots(1, 2, figsize=(W, 2.75))
ax = axes[0]
for k, key in [('msp', 'msp'), ('total', 'total'),
               ('aleatoric', 'alea'), ('epistemic', 'epi')]:
    sgn = -1 if key == 'msp' else 1
    grid = np.linspace(0, 1, 200)
    curves = []
    for s in S5:
        fpr, tpr = U.roc_points(sgn * s[f'id_{key}'], sgn * s[f'ood_{key}'])
        curves.append(np.interp(grid, fpr, tpr))
    curves = np.array(curves)
    ax.plot(grid, curves.mean(0), color=C[k], ls=LS[k], label=LAB[k])
    ax.fill_between(grid, curves.min(0), curves.max(0), color=C[k], alpha=.13, lw=0)
ax.plot([0, 1], [0, 1], color=GREY, lw=.7, ls=(0, (3, 3)))
ax.set_xlabel('Доля ложных срабатываний')
ax.set_ylabel('Доля верных обнаружений')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(frameon=False, loc='lower right', handlelength=2.2, borderaxespad=0.3)
tidy(ax)

ax = axes[1]
order = ['epistemic', 'total', 'msp', 'aleatoric']
for i, k in enumerate(order):
    v = [r['auroc'][k] for r in R]
    m, sd, h = ci(v)
    ax.scatter(v, [i] * len(v), s=17, facecolor='none',
               edgecolor=C[k], linewidth=.9, marker=MK[k], zorder=3)
    ax.plot([m - h, m + h], [i, i], color=C[k], lw=1.6, alpha=.55, zorder=2)
    ax.plot([m], [i], marker='|', color=C[k], ms=11, mew=1.8, zorder=4)
ax.axvline(0.5, color=GREY, lw=.7, ls=(0, (3, 3)))
ax.set_yticks(range(len(order)))
ax.set_yticklabels([LAB[k] for k in order])
ax.set_xlim(0.30, 1.0)
ax.set_ylim(len(order) - 0.5, -0.5)
ax.set_xlabel('AUROC по пяти запускам')
tidy(ax, yfmt=False)
ax.grid(alpha=.18, lw=.4, axis='x')
fig.savefig(os.path.join(OUT, 'fig2_roc_seeds.png'))
plt.close(fig)

import paper_data as P
CB, CT = '#C1651A', '#1F4E79'


class Comma(ScalarFormatter):
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace('.', ',').replace('-', '\u2212')


# ------------------------------------- рис. 3–4: виды сдвига и режимы обучения (эксп. B, C)
def profile(ax, shift, xlabel, xticks=None, logx=False):
    for reg, col, mk, ls, lab in (('base', CB, 's', (0, (5, 2)), '5 эпох, σ = 1 (исходный)'),
                                  ('trained', CT, 'o', '-', '20 эпох, σ = 0,1')):
        lv = P.level_summary(shift, reg)
        if not lv:
            continue
        x = [r['level'] for r in lv]
        per = np.array([r['d_seeds'] for r in lv])          # уровни × сиды
        for j in range(per.shape[1]):
            ax.plot(x, per[:, j], color=col, lw=0.6, alpha=0.35, ls=ls)
        ax.plot(x, per.mean(1), color=col, lw=1.8, ls=ls, marker=mk, ms=4,
                label=f'{lab}, n = {per.shape[1]}')
    ax.axhline(0, color=GREY, lw=0.8)
    if logx:
        ax.set_xscale('log')
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t).replace('.', ',') for t in xticks])
        ax.minorticks_off()
    ax.set_xlabel(xlabel)
    tidy(ax, xfmt=False)


fig, axes = plt.subplots(1, 2, figsize=(W, 2.75), gridspec_kw={'width_ratios': [1.55, 1]})
profile(axes[0], 'rotation', 'Угол поворота, градусы', xticks=P.ANG)
axes[0].set_ylabel('Δ AUROC (эпист. − полная)')
axes[0].legend(frameon=False, loc='lower left', handlelength=2.6)
profile(axes[1], 'noise', 'СКО гауссова шума', xticks=[0.1, 0.2, 0.4, 0.8], logx=True)
lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
for a in axes:
    a.set_ylim(lo, hi)
fig.savefig(os.path.join(OUT, 'fig3_shift_profiles.png'))
plt.close(fig)

# ------------------------------------------------------------- рис. 4: режимы
rt = P.regime_table()
fig, ax = plt.subplots(figsize=(W, 2.55))
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
ax.set_xticklabels([f"{r['key'][0]} эпох, σ = {str(r['key'][1]).replace('.', ',')}\n"
                    f"точность {r['acc'][0] * 100:.1f} %, n = {r['n']}".replace('.', ',', 1)
                    .replace('.', ',') for r in rt])
ax.set_ylabel('Δ AUROC (эпист. − полная)')
ax.set_xlim(-0.5, len(rt) - 0.5)
ax.legend(frameon=False, loc='upper right')
tidy(ax, xfmt=False)
fig.savefig(os.path.join(OUT, 'fig4_regime_interaction.png'))
plt.close(fig)
print('рисунки:', sorted(os.listdir(OUT)))
