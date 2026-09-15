"""Рисунки 1-4 статьи. Все значения читаются из results/*.json и
results/scores_seed*.npz; ничего не задаётся вручную."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import roc_utils as U  # local helpers

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
    'savefig.dpi': 400, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
W = 6.5           # inches ≈ 16.5 cm text width
GREY = '0.45'
C = {'epistemic': '#1F4E79', 'total': '#C1651A',
     'aleatoric': '#6E7B8B', 'msp': '#7A9A3B'}
LAB = {'epistemic': 'эпистемическая', 'total': 'полная энтропия',
       'aleatoric': 'алеаторная', 'msp': 'максимум softmax'}
MK = {'epistemic': 'o', 'total': 's', 'aleatoric': '^', 'msp': 'D'}

R = json.load(open(os.path.join(RES, 'replication.json')))
S5 = [np.load(f) for f in sorted(glob.glob(os.path.join(RES, 'scores_seed*.npz')))]


def ci(v):
    v = np.asarray(v, float)
    sd = v.std(ddof=1)
    h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
    return v.mean(), sd, h


# ------------------------------- рис. 1: составляющие, внутр. vs внешн.
fig, axes = plt.subplots(1, 2, figsize=(W, 2.3))
for ax, key, title in zip(axes, ['epi', 'alea'],
                          ['Эпистемическая составляющая',
                           'Алеаторная составляющая']):
    a = np.concatenate([s[f'id_{key}'] for s in S5])
    b = np.concatenate([s[f'ood_{key}'] for s in S5])
    hi = np.percentile(np.concatenate([a, b]), 99.5)
    bins = np.linspace(0, hi, 70)
    ax.hist(a, bins=bins, density=True, alpha=.62, color=C['epistemic'],
            label=f'FashionMNIST, μ = {a.mean():.2f}')
    ax.hist(b, bins=bins, density=True, alpha=.62, color=C['total'],
            label=f'MNIST, μ = {b.mean():.2f}')
    ax.set_xlabel(f'{title}, нат')
    ax.set_ylabel('Плотность')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(alpha=.2, lw=.4)
fig.tight_layout(w_pad=1.6)
fig.savefig(os.path.join(OUT, 'fig1_components.png'))
plt.close(fig)

# ------------- рис. 2: ROC-кривые и AUROC по отдельным запускам
fig, axes = plt.subplots(1, 2, figsize=(W, 2.7))
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
    ax.plot(grid, curves.mean(0), color=C[k], label=LAB[k])
    ax.fill_between(grid, curves.min(0), curves.max(0), color=C[k], alpha=.13, lw=0)
ax.plot([0, 1], [0, 1], color=GREY, lw=.7, ls=(0, (4, 3)))
ax.set_xlabel('Доля ложных срабатываний')
ax.set_ylabel('Доля верных обнаружений')
ax.legend(frameon=False, loc='lower right')
ax.grid(alpha=.2, lw=.4)

ax = axes[1]
order = ['epistemic', 'total', 'msp', 'aleatoric']
for i, k in enumerate(order):
    v = [r['auroc'][k] for r in R]
    m, sd, h = ci(v)
    ax.scatter(v, [i] * len(v), s=17, facecolor='none',
               edgecolor=C[k], linewidth=.9, zorder=3)
    ax.plot([m - h, m + h], [i, i], color=C[k], lw=1.6, alpha=.55, zorder=2)
    ax.plot([m], [i], marker='|', color=C[k], ms=11, mew=1.8, zorder=4)
ax.axvline(0.5, color=GREY, lw=.7, ls=(0, (4, 3)))
ax.set_yticks(range(len(order)))
ax.set_yticklabels([LAB[k] for k in order])
ax.set_xlim(0.30, 1.0)
ax.set_xlabel('AUROC по пяти сидам')
ax.invert_yaxis()
ax.grid(alpha=.2, lw=.4, axis='x')
fig.tight_layout(w_pad=1.8)
fig.savefig(os.path.join(OUT, 'fig2_roc_seeds.png'))
plt.close(fig)

# ---------------------------------- рис. 3: повороты (эксперимент B)
SH = json.load(open(os.path.join(RES, 'shift.json')))
angles = [a['angle'] for a in SH[0]['angles']]
fig, ax = plt.subplots(figsize=(W * 0.72, 2.6))
for k, key in [('epistemic', 'auroc_epistemic'), ('total', 'auroc_total'),
               ('msp', 'auroc_msp'), ('aleatoric', 'auroc_aleatoric')]:
    xs, ys, lo, hi = [], [], [], []
    for i, ang in enumerate(angles):
        vals = [r['angles'][i].get(key) for r in SH]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        xs.append(ang); ys.append(np.mean(vals))
        lo.append(min(vals)); hi.append(max(vals))
    ax.plot(xs, ys, MK[k] + '-', color=C[k], ms=3.6, label=LAB[k])
    ax.fill_between(xs, lo, hi, color=C[k], alpha=.12, lw=0)
mn = np.mean([r['mnist']['epistemic'] for r in SH])
ax.axhline(mn, color=C['epistemic'], lw=.9, ls=(0, (5, 2)))
ax.text(182, mn, '  MNIST\n  (эпист.)', va='center', fontsize=7.5,
        color=C['epistemic'])
mt = np.mean([r['mnist']['total'] for r in SH])
ax.axhline(mt, color=C['total'], lw=.9, ls=(0, (5, 2)))
ax.text(182, mt, '  MNIST\n  (полная)', va='center', fontsize=7.5, color=C['total'])
ax.axhline(0.5, color=GREY, lw=.7, ls=(0, (2, 2)))
ax.set_xlabel('Угол поворота, градусы')
ax.set_ylabel('AUROC')
ax.set_xticks(angles)
ax.set_xlim(5, 215)
ax.set_ylim(0.45, 1.0)
ax.legend(frameon=False, loc='upper left', ncol=2, columnspacing=1.0)
ax.grid(alpha=.2, lw=.4)
fig.savefig(os.path.join(OUT, 'fig3_rotation.png'))
plt.close(fig)

# ----------------------------- рис. 4: режим обучения (эксперимент C)
rp = os.path.join(RES, 'regime.json')
if os.path.exists(rp):
    G = json.load(open(rp))
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.4))
    cells = sorted({(g['epochs'], g['sigma']) for g in G})
    for ep, sg in cells:
        sub = [g for g in G if g['epochs'] == ep and g['sigma'] == sg]
        x = np.mean([g['bnn_accuracy'] for g in sub])
        y = np.mean([g['auroc']['epistemic'] for g in sub])
        y2 = np.mean([g['auroc']['total'] for g in sub])
        ax.scatter([x], [y], s=34, color=C['epistemic'], zorder=3)
        ax.scatter([x], [y2], s=34, marker='s', color=C['total'], zorder=3)
        ax.annotate(f'{ep} эп., σ={sg}', (x, y), textcoords='offset points',
                    xytext=(5, 5), fontsize=7)
    ax.scatter([], [], s=34, color=C['epistemic'], label='эпистемическая')
    ax.scatter([], [], s=34, marker='s', color=C['total'], label='полная энтропия')
    ax.set_xlabel('Общая точность байесовской сети')
    ax.set_ylabel('AUROC обнаружения MNIST')
    ax.legend(frameon=False, loc='lower left')
    ax.grid(alpha=.2, lw=.4)
    fig.savefig(os.path.join(OUT, 'fig4_regime.png'))
    plt.close(fig)
    print('рис. 4 построен')
else:
    print('рис. 4 пропущен — нет results/regime.json')

print('figures:', sorted(os.listdir(OUT)))
