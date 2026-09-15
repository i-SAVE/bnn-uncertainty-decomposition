# -*- coding: utf-8 -*-
"""Печатает таблицы 2-7 и П1-П3 статьи из results/*.json.

Значения и формат вывода (три знака после запятой, десятичная запятая,
«среднее ± СКО») совпадают с теми, что напечатаны в статье: и таблицы статьи,
и этот скрипт строятся одним и тем же кодом из одних и тех же файлов
результатов, поэтому вывод можно сравнивать со статьёй построчно.
"""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, os.pardir, 'results'))

R = json.load(open(os.path.join(RES, 'replication.json'), encoding='utf-8'))
SH = json.load(open(os.path.join(RES, 'shift.json'), encoding='utf-8'))
REG = json.load(open(os.path.join(RES, 'regime.json'), encoding='utf-8'))

KEYS = ('msp', 'total', 'aleatoric', 'epistemic')


# ------------------------------------------------------------- форматирование
def agg(v):
    """Среднее, СКО и границы 95 % доверительного интервала для среднего."""
    v = np.asarray(v, float)
    m, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
    return m, sd, m - h, m + h


def f3(x):
    return f'{x:.3f}'.replace('.', ',')


def f2(x):
    return f'{x:.2f}'.replace('.', ',')


def pct(x, n=1):
    return f'{x * 100:.{n}f}'.replace('.', ',')


def pm(v, fmt=f3):
    m, sd, _, _ = agg(v)
    return f'{fmt(m)} ± {fmt(sd)}'


def ci(v, fmt=f3):
    _, _, lo, hi = agg(v)
    return f'[{fmt(lo)}; {fmt(hi)}]'


def show(caption, header, rows, note=None):
    cols = len(header)
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(cols)]
    rule = '  '.join('-' * x for x in w)
    print()
    print(caption)
    print(rule)
    print('  '.join(str(header[i]).ljust(w[i]) for i in range(cols)))
    print(rule)
    for r in rows:
        print('  '.join(str(r[i]).ljust(w[i]) for i in range(cols)))
    print(rule)
    if note:
        print(note)


# ------------------------------------------------------------- эксперимент A
AU = {k: [r['auroc'][k] for r in R] for k in KEYS}
FP = {k: [r['fpr95'][k] for r in R] for k in KEYS}
MEAN = {k: ([r[f'mean_{n}']['id'] for r in R], [r[f'mean_{n}']['ood'] for r in R])
        for k, n in (('total', 'entropy'), ('aleatoric', 'aleatoric'),
                     ('epistemic', 'epistemic'))}
DT = [r['delta_epi_vs_total'] for r in R]

print(f'Источник: {RES}')
print(f'Эксперимент A: {len(R)} запусков, сиды {[r["seed"] for r in R]}')

show('Таблица 2. Качество классификации на FashionMNIST и обнаружение MNIST '
     'недекомпозированными мерами',
     ['Показатель', 'Байесовская сеть', 'Детерминированная сеть'],
     [['Общая точность, %',
       pm([r['table1_fashion']['overall_accuracy'] for r in R], pct),
       pm([r['deterministic']['accuracy'] for r in R], pct)],
      ['ECE', pm([r['bnn_ece'] for r in R]),
       pm([r['deterministic']['ece'] for r in R])],
      ['AUROC по полной энтропии', pm(AU['total']),
       pm([r['deterministic']['auroc_entropy'] for r in R])],
      ['AUROC по максимуму softmax', pm(AU['msp']),
       pm([r['deterministic']['auroc_msp'] for r in R])]],
     'Среднее ± СКО по пяти сидам.')

show('Таблица 3. Обнаружение MNIST четырьмя мерами неопределённости, пять сидов',
     ['Мера неопределённости', 'AUROC', '95 % ДИ', 'FPR@95TPR'],
     [['Максимум softmax', pm(AU['msp']), ci(AU['msp']), pm(FP['msp'])],
      ['Полная энтропия', pm(AU['total']), ci(AU['total']), pm(FP['total'])],
      ['Алеаторная составляющая', pm(AU['aleatoric']), ci(AU['aleatoric']),
       pm(FP['aleatoric'])],
      ['Эпистемическая составляющая', pm(AU['epistemic']), ci(AU['epistemic']),
       pm(FP['epistemic'])]],
     'Доверительный интервал — для среднего по сидам (t-распределение, '
     '4 степени свободы).')

show('Таблица 4. Средние значения составляющих неопределённости',
     ['Составляющая', 'FashionMNIST (внутр.), нат', 'MNIST (внешн.), нат'],
     [['Полная предсказательная энтропия', pm(MEAN['total'][0]), pm(MEAN['total'][1])],
      ['Алеаторная составляющая', pm(MEAN['aleatoric'][0]), pm(MEAN['aleatoric'][1])],
      ['Эпистемическая составляющая', pm(MEAN['epistemic'][0]),
       pm(MEAN['epistemic'][1])]],
     'Среднее ± СКО по пяти сидам.')

# ------------------------------------------------------------- эксперимент B
angles = [a['angle'] for a in SH[0]['angles']]
rows, unstable = [], []
for i, ang in enumerate(angles):
    rr = [x['angles'][i] for x in SH]
    acc = np.mean([x['accuracy'] for x in rr])
    if ang == 0:
        rows.append([f'{ang}°', f2(acc), '—', '—', '—', '—'])
        continue
    v = {k: np.mean([x[f'auroc_{k}'] for x in rr]) for k in KEYS}
    dd = [x['auroc_epistemic'] - x['auroc_total'] for x in rr]
    mark = '' if dd[0] * dd[1] > 0 else ' *'
    if mark:
        unstable.append(ang)
    rows.append([f'{ang}°{mark}', f2(acc), f3(v['epistemic']), f3(v['total']),
                 f3(v['aleatoric']), f3(v['msp'])])
mn = {k: np.mean([x['mnist'][k] for x in SH]) for k in KEYS}
rows.append(['MNIST', '—', f3(mn['epistemic']), f3(mn['total']),
             f3(mn['aleatoric']), f3(mn['msp'])])

show('Таблица 5. AUROC обнаружения внутридоменного сдвига в зависимости от угла '
     'поворота',
     ['Сдвиг', 'Точность', 'Эпистем.', 'Полная', 'Алеатор.', 'softmax'], rows,
     'Два запуска, по 4000 объектов в каждом распределении. Звёздочкой отмечены '
     'углы (' + ', '.join(f'{a}°' for a in unstable) + '), на которых знак разности '
     'AUROC эпистемической составляющей и полной энтропии различается между '
     'запусками.')

# ------------------------------------------------------------- эксперимент C
cells, gaps = [], {}
for ep in sorted({g['epochs'] for g in REG}):
    for sg in sorted({g['sigma'] for g in REG}, reverse=True):
        sub = [g for g in REG if g['epochs'] == ep and g['sigma'] == sg]
        if not sub:
            continue
        gp = np.mean([g['auroc']['epistemic'] - g['auroc']['total'] for g in sub])
        gaps[(ep, sg)] = gp
        cells.append([f"{ep} / {str(sg).replace('.', ',')}",
                      pct(np.mean([g['bnn_accuracy'] for g in sub])),
                      f3(np.mean([g['bnn_ece'] for g in sub])),
                      f3(np.mean([g['auroc']['epistemic'] for g in sub])),
                      f3(np.mean([g['auroc']['total'] for g in sub])),
                      f3(np.mean([g['auroc']['aleatoric'] for g in sub])),
                      f3(gp), str(len(sub))])

show('Таблица 6. Зависимость качества модели и обнаружения MNIST от режима обучения',
     ['Эпохи / σ', 'Точн., %', 'ECE', 'Эпистем.', 'Полная', 'Алеатор.', 'Разрыв', 'n'],
     cells,
     '«Разрыв» — разность AUROC эпистемической составляющей и полной энтропии. '
     'n — число запусков; оценивание на подвыборке 5000 объектов из каждого '
     'распределения.')

# ---------------------------------------------------------- правило отказа
ts = R[0]['threshold_sweep']
rows = []
for tv in (0.3, 0.5, 0.7, 0.9):
    i = min(range(len(ts)), key=lambda j: abs(ts[j]['tau'] - tv))
    rows.append([f2(ts[i]['tau']),
                 pct(np.mean([x['threshold_sweep'][i]['rej_id'] for x in R])),
                 pct(np.mean([x['threshold_sweep'][i]['rej_ood'] for x in R])),
                 pct(np.mean([x['threshold_sweep'][i]['acc_accepted'] for x in R]))])

show('Таблица 7. Правило отказа по максимуму апостериорной вероятности',
     ['τ', 'Отказы, FashionMNIST, %', 'Отказы, MNIST, %', 'Точность на принятых, %'],
     rows, 'Среднее по пяти сидам.')

# ------------------------------------------------------------------ приложение
show('Таблица П1. Эксперимент A: AUROC и парный бутстреп разности по запускам',
     ['Сид', 'Эпистем.', 'Полная', 'Алеатор.', 'softmax',
      'Разность эпистем. − полная, 95 % ДИ'],
     [[str(r['seed']), f3(r['auroc']['epistemic']), f3(r['auroc']['total']),
       f3(r['auroc']['aleatoric']), f3(r['auroc']['msp']),
       f'+{f3(DT[i][0])} [+{f3(DT[i][1])}; +{f3(DT[i][2])}]']
      for i, r in enumerate(R)],
     'Парный бутстреп, 2000 повторных выборок на одних и тех же объектах.')

rows = []
for i, ang in enumerate(angles):
    if ang == 0:
        continue
    dd = [x['angles'][i]['auroc_epistemic'] - x['angles'][i]['auroc_total'] for x in SH]
    rows.append([f'{ang}°',
                 f"{'+' if dd[0] > 0 else ''}{f3(dd[0])}",
                 f"{'+' if dd[1] > 0 else ''}{f3(dd[1])}",
                 'да' if dd[0] * dd[1] > 0 else 'нет'])

show('Таблица П2. Эксперимент B: разность AUROC эпистемической составляющей и '
     'полной энтропии по запускам',
     ['Угол', 'Сид 0', 'Сид 1', 'Знак совпадает'], rows)

rows = []
for g in sorted(REG, key=lambda x: (x['epochs'], -x['sigma'], x['seed'])):
    dd = g['auroc']['epistemic'] - g['auroc']['total']
    rows.append([f"{g['epochs']} / {str(g['sigma']).replace('.', ',')}",
                 str(g['seed']), pct(g['bnn_accuracy']),
                 f3(g['auroc']['epistemic']), f3(g['auroc']['total']),
                 f"{'+' if dd > 0 else ''}{f3(dd)}"])

show('Таблица П3. Эксперимент C: значения по отдельным запускам',
     ['Эпохи / σ', 'Сид', 'Точн., %', 'Эпистем.', 'Полная', 'Разрыв'], rows)
print()
