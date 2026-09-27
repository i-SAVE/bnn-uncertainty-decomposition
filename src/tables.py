# -*- coding: utf-8 -*-
"""Печатает таблицы 2–5 и П1–П3 статьи, а также данные рис. 4, из results/.

Таблицы статьи строятся тем же расчётом из тех же файлов, поэтому вывод можно сравнивать
со статьёй построчно (десятичная запятая, типографский минус).
"""
import numpy as np
import paper_data as P

R = P.R
MINUS = '−'


def f3(x): return f'{x:.3f}'.replace('.', ',').replace('-', MINUS)
def sg(x): return f'{x:+.3f}'.replace('.', ',').replace('-', MINUS)
def pct(x): return f'{x * 100:.1f}'.replace('.', ',').replace('-', MINUS)


def pm(v, fmt=f3):
    m, sd, _, _ = P.mci(v)
    return f'{fmt(m)} ± {fmt(sd)}'


def show(caption, header, rows, note=None):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    rule = '  '.join('-' * x for x in w)
    print('\n' + caption); print(rule)
    print('  '.join(str(header[i]).ljust(w[i]) for i in range(len(header)))); print(rule)
    for r in rows:
        print('  '.join(str(r[i]).ljust(w[i]) for i in range(len(header))))
    print(rule)
    if note:
        print(note)


AU = {k: [r['auroc'][k] for r in R] for k in P.K}
FP = {k: [r['fpr95'][k] for r in R] for k in P.K}
show('Таблица 2. Качество классификации и обнаружение MNIST недекомпозированными мерами',
     ['Показатель', 'Байесовская сеть', 'Детерминированная сеть'],
     [['Общая точность, %', pm([r['table1_fashion']['overall_accuracy'] for r in R], pct),
       pm([r['deterministic']['accuracy'] for r in R], pct)],
      ['ECE', pm([r['bnn_ece'] for r in R]), pm([r['deterministic']['ece'] for r in R])],
      ['AUROC по полной энтропии', pm(AU['total']), pm([r['deterministic']['auroc_entropy'] for r in R])],
      ['AUROC по максимуму softmax', pm(AU['msp']), pm([r['deterministic']['auroc_msp'] for r in R])]])


def ci(v):
    _, _, lo, hi = P.mci(v)
    return f'[{f3(lo)}; {f3(hi)}]'


show('Таблица 3. Обнаружение MNIST четырьмя мерами, пять сидов',
     ['Мера', 'AUROC', '95 % ДИ', 'FPR@95TPR'],
     [[n, pm(AU[k]), ci(AU[k]), pm(FP[k])] for k, n in
      (('msp', 'Максимум softmax'), ('total', 'Полная энтропия'),
       ('aleatoric', 'Алеаторная'), ('epistemic', 'Эпистемическая'))])

MEAN = {k: ([r[f'mean_{n}']['id'] for r in R], [r[f'mean_{n}']['ood'] for r in R])
        for k, n in (('total', 'entropy'), ('aleatoric', 'aleatoric'), ('epistemic', 'epistemic'))}
show('Таблица 4. Средние значения составляющих, нат',
     ['Составляющая', 'FashionMNIST', 'MNIST'],
     [[n, pm(MEAN[k][0]), pm(MEAN[k][1])] for k, n in
      (('total', 'Полная энтропия'), ('aleatoric', 'Алеаторная'), ('epistemic', 'Эпистемическая'))])

SH = [('mnist', 'Межнаборный: MNIST'), ('kmnist', 'Межнаборный: KMNIST'),
      ('classes', 'Близкий: отложенные классы'), ('rotation', 'Ковариатный: повороты'),
      ('noise', 'Ковариатный: гауссов шум')]
rows = []
for k, name in SH:
    row = [name]
    for rg in ('base', 'trained'):
        sm = P.shift_summary(k, rg)
        row += [f3(sm['auroc']['epistemic'][0]), f3(sm['auroc']['total'][0]),
                f"{sg(sm['d'][0])} ± {f3(sm['d'][1])}", f"{sm['pos']}/{sm['neg']}/{sm['cells']}"]
    rows.append(row)
show('Таблица 5. Пять видов сдвига в исходном (5 эпох, σ = 1) и лучше обученном (20 эпох, σ = 0,1) режимах',
     ['Сдвиг', 'эпист.', 'полная', 'Δ', '+/−/n', 'эпист.', 'полная', 'Δ', '+/−/n'], rows,
     '«+/−/n» — число комбинаций «уровень × запуск» с ДИ разности выше / ниже нуля из n.')

show('Таблица П4 (данные рис. 4). Режимы обучения: Δ(эпист. − полная)',
     ['Режим', 'n', 'Точность, %', 'MNIST', 'Повороты (среднее)', 'Повороты: + / n'],
     [[f"{r['key'][0]} эпох, σ = {str(r['key'][1]).replace('.', ',')}", r['n'], pct(r['acc'][0]),
       f"{sg(r['d_mnist'][0])} ± {f3(r['d_mnist'][1])}", f"{sg(r['d_rot'][0])} ± {f3(r['d_rot'][1])}",
       f"{r['rot_pos']}/{r['rot_cells']}"] for r in P.regime_table()])

DT = [r['delta_epi_vs_total'] for r in R]
show('Таблица П1. Эксперимент A по запускам',
     ['Сид', 'Эпистем.', 'Полная', 'Алеатор.', 'softmax', 'Разность, 95 % ДИ'],
     [[r['seed'], f3(r['auroc']['epistemic']), f3(r['auroc']['total']), f3(r['auroc']['aleatoric']),
       f3(r['auroc']['msp']), f'{sg(d[0])} [{sg(d[1])}; {sg(d[2])}]'] for r, d in zip(R, DT)])

ub, ut = P.shift_units('rotation', 'base'), P.shift_units('rotation', 'trained')
rows = []
for j, ang in enumerate(P.ANG):
    cell = lambda u, s: sg(u[s][j]['d']['point']) + ('°' if P.cls(u[s][j]['d']) == '0' else '')
    rows.append([f'{ang}°'] + [cell(ub, s) for s in sorted(ub)] + [cell(ut, s) for s in sorted(ut)])
show('Таблица П2. Повороты: разность по углам и запускам (и — исходный режим, л — лучше обученный)',
     ['Угол'] + [f'и{s}' for s in sorted(ub)] + [f'л{s}' for s in sorted(ut)], rows,
     '° — ДИ парного бутстрепа содержит ноль.')

rows = []
for rg, lab in (('base', 'исх.'), ('trained', 'лучше')):
    for s in sorted(P.shift_units('mnist', rg)):
        row = [f'{lab}, {s}']
        for k, _ in SH:
            u = P.shift_units(k, rg).get(s)
            row.append(sg(np.mean([x['d']['point'] for x in u])) if u else '—')
        rows.append(row)
show('Таблица П3. Разность по видам сдвига и запускам',
     ['Режим, сид', 'MNIST', 'KMNIST', 'Классы', 'Пов.', 'Шум'], rows)
n, mx = P.repro_checks()
print(f'\nПроверок побитовой воспроизводимости: {n}, максимальное расхождение AUROC: {mx}')

# представительность подвыборок 4000: те же модели эксперимента A, полный набор и подвыборка
A2b = P.A2[(5, 1.0)]
sub = [(r['auroc'], A2b[r['seed']]['mnist_subset4000']['auroc']) for r in R if r['seed'] in A2b]
subc = max(abs(f[k] - q[k]) for f, q in sub for k in P.K)
subd = max(abs((f['epistemic'] - f['total']) - (q['epistemic'] - q['total'])) for f, q in sub)
dfull = [f['epistemic'] - f['total'] for f, q in sub]
print(f'Подвыборка 4000 против полного набора MNIST ({len(sub)} моделей): '
      f'макс. |ΔAUROC| по мерам {f3(subc)}; макс. различие разности «эпист. − полная» {f3(subd)} '
      f'при её величине от {f3(min(dfull))} до {f3(max(dfull))}')
nc, mc = P.mnist_consistency()
print(f'Совпадение моделей в независимых прогонах (одинаковые конфигурация и сид; MNIST, подвыборка 4000): '
      f'{nc} сравнений, максимальное расхождение {mc}')
